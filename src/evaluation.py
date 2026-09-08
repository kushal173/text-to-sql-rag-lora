"""SQL extraction and evaluation for Chinook."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path


class SQLEvaluator:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)

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
    ) -> dict:
        if extract_first_statement:
            predicted_sql = self.extract_first_statement(predicted_sql)
        exact = self.normalize(predicted_sql) == self.normalize(gold_sql)
        error = None
        execution = False
        try:
            with sqlite3.connect(self.db_path) as connection:
                predicted_rows = connection.execute(predicted_sql).fetchall()
                gold_rows = connection.execute(gold_sql).fetchall()
            execution = predicted_rows == gold_rows
        except Exception as exc:
            error = str(exc)
        return {
            "generated_sql": predicted_sql,
            "exact_match": exact,
            "execution_match": execution,
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
            )
            rows.append({**item, **result})
        total = len(rows) or 1
        metrics = {
            "Exact Match Accuracy": round(100 * sum(r["exact_match"] for r in rows) / total, 2),
            "Execution Match Accuracy": round(100 * sum(r["execution_match"] for r in rows) / total, 2),
        }
        return rows, metrics
