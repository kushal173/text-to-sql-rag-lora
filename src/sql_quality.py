"""Deterministic quality, grounding, and safety helpers for Text-to-SQL evaluation."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from .retrieval_evaluation import extract_chunk_tables, extract_sql_tables


_UNSAFE_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma|vacuum|reindex)\b",
    flags=re.IGNORECASE,
)
_READ_ONLY_START = re.compile(r"^\s*(select|with)\b", flags=re.IGNORECASE)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def prf_from_sets(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    overlap = len(predicted & gold)
    precision = safe_div(overlap, len(predicted))
    recall = safe_div(overlap, len(gold))
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def multiset_prf(predicted_rows: Iterable[Any], gold_rows: Iterable[Any]) -> tuple[float, float, float]:
    """Precision/recall/F1 over result rows while preserving duplicate counts."""
    predicted_counter = Counter(tuple(row) if isinstance(row, list) else row for row in predicted_rows)
    gold_counter = Counter(tuple(row) if isinstance(row, list) else row for row in gold_rows)
    overlap = sum((predicted_counter & gold_counter).values())
    predicted_total = sum(predicted_counter.values())
    gold_total = sum(gold_counter.values())
    precision = safe_div(overlap, predicted_total)
    recall = safe_div(overlap, gold_total)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def is_read_only_sql(sql: str) -> bool:
    """True for SELECT/WITH queries that contain no obvious mutating SQL command."""
    text = (sql or "").strip()
    return bool(_READ_ONLY_START.search(text)) and not bool(_UNSAFE_SQL_PATTERN.search(text))


def contains_unsafe_sql(sql: str) -> bool:
    return bool(_UNSAFE_SQL_PATTERN.search(sql or ""))


def single_statement_sql(sql: str) -> bool:
    """Conservative format check used as a system-prompt compliance proxy."""
    text = (sql or "").strip()
    if not text:
        return False
    # A trailing semicolon is fine; any additional non-empty statement is not.
    parts = [part.strip() for part in text.split(";") if part.strip()]
    return len(parts) == 1 and bool(_READ_ONLY_START.search(parts[0]))


def context_tables(context: Any) -> set[str]:
    if context is None:
        return set()
    if isinstance(context, (str, dict)):
        context = [context]
    tables: set[str] = set()
    for chunk in context:
        tables.update(extract_chunk_tables(chunk))
    return tables


def grounding_metrics(predicted_sql: str, retrieved_context: Any) -> tuple[float, float]:
    """Return table-level grounding score and all-tables-grounded indicator."""
    predicted_tables = extract_sql_tables(predicted_sql)
    if not predicted_tables:
        return 0.0, 0.0
    retrieved_tables = context_tables(retrieved_context)
    grounded = predicted_tables & retrieved_tables
    score = safe_div(len(grounded), len(predicted_tables))
    fully_grounded = 1.0 if predicted_tables.issubset(retrieved_tables) else 0.0
    return score, fully_grounded
