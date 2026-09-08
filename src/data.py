"""Chinook data loading, schema extraction and fine-tuning preparation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ChinookData:
    """Load Chinook examples and extract the SQLite schema."""

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.db_path = self.dataset_dir / "chinook.db"
        self.train_path = self.dataset_dir / "chinook_data.json"
        self.test_path = self.dataset_dir / "chinook_test_30.json"

    @staticmethod
    def load_json(path: str | Path) -> list[dict[str, Any]]:
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)

    def load_train(self) -> list[dict[str, Any]]:
        return self.load_json(self.train_path)

    def load_test(self) -> list[dict[str, Any]]:
        return self.load_json(self.test_path)

    def schema_dict(self) -> dict[str, str]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return {name: statement for name, statement in rows if statement}

    def full_schema_text(self) -> str:
        return "\n".join(
            f"{name}: {sql}" for name, sql in self.schema_from_training().items()
        )

    def schema_from_training(self) -> dict[str, str]:
        """Return the embedded schema while preserving table order."""
        return self.load_train()[0]["schema"]

    @staticmethod
    def make_finetune_rows(examples: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Create instruction/input/output records for causal-LM fine-tuning."""
        rows = []
        for example in examples:
            rows.append(
                {
                    "instruction": "Write an SQL query for the Chinook database.",
                    "input": example["question"],
                    "output": example.get("query", example.get("gold_sql", "")),
                }
            )
        return rows
