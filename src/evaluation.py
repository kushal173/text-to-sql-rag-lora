"""SQL extraction and comprehensive deterministic evaluation for Chinook."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from .retrieval_evaluation import extract_sql_tables
from .sql_quality import (
    contains_unsafe_sql,
    grounding_metrics,
    is_read_only_sql,
    multiset_prf,
    prf_from_sets,
    single_statement_sql,
)


class SQLEvaluator:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        self.schema_tables = {name.lower() for (name,) in rows}

    @staticmethod
    def normalize(sql: str) -> str:
        return " ".join((sql or "").strip().replace("\n", " ").lower().split()).rstrip(";")

    @staticmethod
    def extract_first_statement(text: str) -> str:
        """Keep only the first SQL statement, removing model explanations."""
        if not text:
            return ""
        cleaned = re.sub(r"^\s*```(?:sql)?\s*", "", text, flags=re.I)
        match = re.search(r"\b(select|with)\b", cleaned, flags=re.I)
        if not match:
            return ""
        cleaned = cleaned[match.start():]
        end = cleaned.find(";")
        return (cleaned[: end + 1] if end >= 0 else cleaned).strip().replace("```", "").strip()

    def compare(
        self,
        predicted_sql: str,
        gold_sql: str,
        extract_first_statement: bool = True,
        retrieved_context=None,
    ) -> dict:
        if extract_first_statement:
            predicted_sql = self.extract_first_statement(predicted_sql)

        exact = self.normalize(predicted_sql) == self.normalize(gold_sql)
        error = None
        execution = False
        predicted_rows = []
        gold_rows = []
        try:
            with sqlite3.connect(self.db_path) as connection:
                predicted_rows = connection.execute(predicted_sql).fetchall()
                gold_rows = connection.execute(gold_sql).fetchall()
            execution = predicted_rows == gold_rows
        except Exception as exc:
            error = str(exc)

        sql_valid = bool(predicted_sql) and error is None
        result_precision = result_recall = result_f1 = 0.0
        if sql_valid:
            result_precision, result_recall, result_f1 = multiset_prf(predicted_rows, gold_rows)

        predicted_tables = extract_sql_tables(predicted_sql)
        gold_tables = extract_sql_tables(gold_sql)
        table_precision, table_recall, table_f1 = prf_from_sets(predicted_tables, gold_tables)
        schema_compliant = bool(predicted_sql) and predicted_tables.issubset(self.schema_tables)
        grounding_score, fully_grounded = grounding_metrics(predicted_sql, retrieved_context)

        return {
            "generated_sql": predicted_sql,
            "exact_match": exact,
            "execution_match": execution,
            "sql_valid": sql_valid,
            "sql_extraction_success": bool(predicted_sql),
            "single_statement_compliance": single_statement_sql(predicted_sql),
            "read_only_compliance": is_read_only_sql(predicted_sql),
            "unsafe_sql": contains_unsafe_sql(predicted_sql),
            "schema_compliance": schema_compliant,
            "predicted_tables": sorted(predicted_tables),
            "gold_tables": sorted(gold_tables),
            "table_selection_precision": table_precision,
            "table_selection_recall": table_recall,
            "table_selection_f1": table_f1,
            "result_set_precision": result_precision,
            "result_set_recall": result_recall,
            "result_set_f1": result_f1,
            "context_grounding_score": grounding_score,
            "fully_grounded_query": fully_grounded,
            "error": error,
        }

    def evaluate(
        self,
        predictions: list[dict],
        extract_first_statement: bool = True,
    ) -> tuple[list[dict], dict]:
        rows = []
        for item in predictions:
            gold = item.get("ground_truth", item.get("gold_sql", item.get("query", "")))
            result = self.compare(
                item.get("generated_sql", ""),
                gold,
                extract_first_statement=extract_first_statement,
                retrieved_context=item.get("retrieved_context"),
            )
            rows.append({**item, **result})

        total = len(rows) or 1
        has_context = any(item.get("retrieved_context") is not None for item in predictions)
        metrics = {
            "n": len(rows),
            "Exact Match Accuracy": round(100 * sum(r["exact_match"] for r in rows) / total, 2),
            "Execution Match Accuracy": round(100 * sum(r["execution_match"] for r in rows) / total, 2),
            "SQL Validity Rate": round(100 * sum(r["sql_valid"] for r in rows) / total, 2),
            "SQL Extraction Success Rate": round(100 * sum(r["sql_extraction_success"] for r in rows) / total, 2),
            "Single-Statement Compliance Rate": round(100 * sum(r["single_statement_compliance"] for r in rows) / total, 2),
            "Read-Only Compliance Rate": round(100 * sum(r["read_only_compliance"] for r in rows) / total, 2),
            "Unsafe SQL Rate": round(100 * sum(r["unsafe_sql"] for r in rows) / total, 2),
            "Schema Compliance Rate": round(100 * sum(r["schema_compliance"] for r in rows) / total, 2),
            "Mean Table Selection Precision": round(sum(r["table_selection_precision"] for r in rows) / total, 4),
            "Mean Table Selection Recall": round(sum(r["table_selection_recall"] for r in rows) / total, 4),
            "Mean Table Selection F1": round(sum(r["table_selection_f1"] for r in rows) / total, 4),
            "Mean Result-Set Precision": round(sum(r["result_set_precision"] for r in rows) / total, 4),
            "Mean Result-Set Recall": round(sum(r["result_set_recall"] for r in rows) / total, 4),
            "Mean Result-Set F1": round(sum(r["result_set_f1"] for r in rows) / total, 4),
        }
        if has_context:
            metrics.update({
                "Mean Context Grounding Score": round(sum(r["context_grounding_score"] for r in rows) / total, 4),
                "Fully Grounded Query Rate": round(100 * sum(r["fully_grounded_query"] for r in rows) / total, 2),
            })
        return rows, metrics
