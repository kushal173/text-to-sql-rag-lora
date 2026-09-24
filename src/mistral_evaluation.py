"""Evaluation helpers for Mistral datasets with RAG, quality, and safety metrics."""

from __future__ import annotations

import json
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


class MistralSQLEvaluator:
    def __init__(self, datasets_root: str | Path):
        self.datasets_root = Path(datasets_root)

    @staticmethod
    def normalize(sql: str) -> str:
        sql = re.sub(r"\s+", " ", (sql or "").strip().lower())
        return sql.replace(";", "").replace(" ,", ",")

    def db_path(self, dataset: str, db_id: str) -> Path:
        folder = self.datasets_root / dataset
        nested = folder / db_id / f"{db_id}.sqlite"
        if nested.is_file():
            return nested
        direct = folder / f"{db_id}.sqlite"
        if direct.is_file():
            return direct
        direct_db = folder / f"{db_id}.db"
        if direct_db.is_file():
            return direct_db
        raise FileNotFoundError(f"SQLite database not found for {db_id}")

    @staticmethod
    def _schema_tables(db_path: Path) -> set[str]:
        with sqlite3.connect(db_path) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        return {name.lower() for (name,) in rows}

    def evaluate(self, rows: list[dict], dataset: str) -> tuple[list[dict], dict]:
        evaluated = []
        for row in rows:
            raw_predicted = row.get("clean_sql_extracted", "")
            predicted = self.normalize(raw_predicted)
            gold = self.normalize(row["gold_sql"])
            exact = predicted == gold
            execution = False
            error = None
            predicted_result = []
            gold_result = []
            db_path = self.db_path(dataset, row["db_id"])
            try:
                with sqlite3.connect(db_path) as connection:
                    predicted_result = connection.execute(predicted).fetchall()
                    gold_result = connection.execute(gold).fetchall()
                execution = predicted_result == gold_result
            except Exception as exc:
                error = str(exc)

            sql_valid = bool(predicted) and error is None
            result_precision = result_recall = result_f1 = 0.0
            if sql_valid:
                result_precision, result_recall, result_f1 = multiset_prf(
                    predicted_result, gold_result
                )

            predicted_tables = extract_sql_tables(predicted)
            gold_tables = extract_sql_tables(gold)
            table_precision, table_recall, table_f1 = prf_from_sets(
                predicted_tables, gold_tables
            )
            schema_tables = self._schema_tables(db_path)
            schema_compliant = bool(predicted) and predicted_tables.issubset(schema_tables)
            grounding_score, fully_grounded = grounding_metrics(
                predicted, row.get("retrieved_context")
            )

            evaluated.append(
                {
                    **row,
                    "exact_match": exact,
                    "execution_match": execution,
                    "sql_valid": sql_valid,
                    "sql_extraction_success": bool(predicted),
                    "single_statement_compliance": single_statement_sql(predicted),
                    "read_only_compliance": is_read_only_sql(predicted),
                    "unsafe_sql": contains_unsafe_sql(predicted),
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
            )

        total = len(rows) or 1
        has_context = any(row.get("retrieved_context") is not None for row in rows)
        metrics = {
            "n": len(rows),
            "Exact Match Accuracy": round(
                100 * sum(row["exact_match"] for row in evaluated) / total, 2
            ),
            "Execution Match Accuracy": round(
                100 * sum(row["execution_match"] for row in evaluated) / total, 2
            ),
            "SQL Validity Rate": round(100 * sum(row["sql_valid"] for row in evaluated) / total, 2),
            "SQL Extraction Success Rate": round(100 * sum(row["sql_extraction_success"] for row in evaluated) / total, 2),
            "Single-Statement Compliance Rate": round(100 * sum(row["single_statement_compliance"] for row in evaluated) / total, 2),
            "Read-Only Compliance Rate": round(100 * sum(row["read_only_compliance"] for row in evaluated) / total, 2),
            "Unsafe SQL Rate": round(100 * sum(row["unsafe_sql"] for row in evaluated) / total, 2),
            "Schema Compliance Rate": round(100 * sum(row["schema_compliance"] for row in evaluated) / total, 2),
            "Mean Table Selection Precision": round(sum(row["table_selection_precision"] for row in evaluated) / total, 4),
            "Mean Table Selection Recall": round(sum(row["table_selection_recall"] for row in evaluated) / total, 4),
            "Mean Table Selection F1": round(sum(row["table_selection_f1"] for row in evaluated) / total, 4),
            "Mean Result-Set Precision": round(sum(row["result_set_precision"] for row in evaluated) / total, 4),
            "Mean Result-Set Recall": round(sum(row["result_set_recall"] for row in evaluated) / total, 4),
            "Mean Result-Set F1": round(sum(row["result_set_f1"] for row in evaluated) / total, 4),
        }
        if has_context:
            metrics.update({
                "Mean Context Grounding Score": round(sum(row["context_grounding_score"] for row in evaluated) / total, 4),
                "Fully Grounded Query Rate": round(100 * sum(row["fully_grounded_query"] for row in evaluated) / total, 2),
            })
        return evaluated, metrics

    @staticmethod
    def load_saved_csv(path: str | Path) -> list[dict]:
        import pandas as pd

        return pd.read_csv(path).fillna("").to_dict(orient="records")

    @staticmethod
    def save(rows: list[dict], metrics: dict, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"results": rows, "metrics": metrics}, indent=2),
            encoding="utf-8",
        )
