"""Lightweight retrieval evaluation for the existing Text-to-SQL RAG pipeline.

These metrics do not change retrieval or generation. They use each gold SQL query
as a source of required table names, then measure whether the existing retriever's
Top-K context covers those tables.
"""

from __future__ import annotations

import re
from typing import Any


_TABLE_PATTERN = re.compile(
    r"\b(?:from|join|update|into)\s+[`\"\[]?([A-Za-z_][\w$]*)[`\"\]]?",
    flags=re.IGNORECASE,
)
_TABLE_LABEL_PATTERN = re.compile(
    r"\bTable\s*:\s*[`\"\[]?([A-Za-z_][\w$]*)[`\"\]]?",
    flags=re.IGNORECASE,
)
_CREATE_TABLE_PATTERN = re.compile(
    r"\bcreate\s+table\s+(?:if\s+not\s+exists\s+)?[`\"\[]?([A-Za-z_][\w$]*)[`\"\]]?",
    flags=re.IGNORECASE,
)


def extract_sql_tables(sql: str) -> set[str]:
    """Extract table names referenced by common SQL FROM/JOIN/UPDATE/INTO clauses."""
    return {match.lower() for match in _TABLE_PATTERN.findall(sql or "")}


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        parts = []
        for key in ("question", "query", "gold_sql", "sql", "schema"):
            value = chunk.get(key)
            if value:
                parts.append(str(value))
        return "\n".join(parts)
    return str(chunk)


def extract_chunk_tables(chunk: Any) -> set[str]:
    """Extract table names represented inside one retrieved knowledge chunk."""
    text = _chunk_text(chunk)
    tables = extract_sql_tables(text)
    tables.update(match.lower() for match in _TABLE_LABEL_PATTERN.findall(text))
    tables.update(match.lower() for match in _CREATE_TABLE_PATTERN.findall(text))
    return tables


def evaluate_retriever(examples: list[dict], retriever, top_k: int = 3, limit: int | None = None):
    """Evaluate the existing retriever without modifying its ranking logic.

    Metrics are table-grounded proxies derived from each example's gold SQL:
    - Table Recall@K: fraction of gold-required tables found in retrieved context.
    - Table Precision@K: fraction of distinct retrieved tables that are gold-required.
    - MRR: reciprocal rank of the first retrieved chunk containing any required table.
    """
    selected = examples if limit is None else examples[:limit]
    rows = []

    for example in selected:
        question = example.get("question", example.get("natural_language_question", ""))
        gold_sql = example.get("gold_sql", example.get("query", example.get("ground_truth", "")))
        gold_tables = extract_sql_tables(gold_sql)
        retrieved = retriever.retrieve(question, top_k=top_k)
        chunk_tables = [extract_chunk_tables(chunk) for chunk in retrieved]
        retrieved_tables = set().union(*chunk_tables) if chunk_tables else set()

        overlap = gold_tables & retrieved_tables
        recall = len(overlap) / len(gold_tables) if gold_tables else 0.0
        precision = len(overlap) / len(retrieved_tables) if retrieved_tables else 0.0

        first_relevant_rank = next(
            (rank for rank, tables in enumerate(chunk_tables, start=1) if gold_tables & tables),
            None,
        )
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

        rows.append(
            {
                "question": question,
                "gold_sql": gold_sql,
                "gold_tables": sorted(gold_tables),
                "retrieved_tables": sorted(retrieved_tables),
                f"table_recall@{top_k}": recall,
                f"table_precision@{top_k}": precision,
                "reciprocal_rank": reciprocal_rank,
                "retrieved_context": retrieved,
            }
        )

    total = len(rows) or 1
    metrics = {
        f"Table Recall@{top_k}": round(sum(r[f"table_recall@{top_k}"] for r in rows) / total, 4),
        f"Table Precision@{top_k}": round(sum(r[f"table_precision@{top_k}"] for r in rows) / total, 4),
        "MRR": round(sum(r["reciprocal_rank"] for r in rows) / total, 4),
        "Evaluated Questions": len(rows),
    }
    return rows, metrics
