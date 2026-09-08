"""Evaluation helpers for both Mistral datasets."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


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

    def evaluate(self, rows: list[dict], dataset: str) -> tuple[list[dict], dict]:
        evaluated = []
        for row in rows:
            predicted = self.normalize(row["clean_sql_extracted"])
            gold = self.normalize(row["gold_sql"])
            exact = predicted == gold
            execution = False
            error = None
            try:
                with sqlite3.connect(self.db_path(dataset, row["db_id"])) as connection:
                    predicted_result = connection.execute(predicted).fetchall()
                    gold_result = connection.execute(gold).fetchall()
                execution = predicted_result == gold_result
            except Exception as exc:
                error = str(exc)
            evaluated.append(
                {
                    **row,
                    "exact_match": exact,
                    "execution_match": execution,
                    "error": error,
                }
            )
        total = len(rows) or 1
        metrics = {
            "n": len(rows),
            "Exact Match Accuracy": round(
                100 * sum(row["exact_match"] for row in evaluated) / total, 2
            ),
            "Execution Match Accuracy": round(
                100 * sum(row["execution_match"] for row in evaluated) / total, 2
            ),
        }
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
