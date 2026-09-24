"""Retrieval evaluation for the existing Text-to-SQL RAG pipeline.

The module is intentionally evaluation-only: it does not change embeddings,
indexing, ranking, prompts, or generation. Gold SQL supplies table-level
relevance labels, which lets the original retriever be measured without a
separate manual chunk-annotation set.
"""

from __future__ import annotations

import math
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


def _safe_f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _average_precision_at_k(relevance: list[int], total_relevant: int, k: int) -> float:
    """Standard binary AP@K; denominator is min(total relevant corpus items, K)."""
    if total_relevant <= 0:
        return 0.0
    running_relevant = 0
    score = 0.0
    for rank, relevant in enumerate(relevance[:k], start=1):
        if relevant:
            running_relevant += 1
            score += running_relevant / rank
    return score / min(total_relevant, k)


def _ndcg_at_k(relevance: list[int], total_relevant: int, k: int) -> float:
    """Binary nDCG@K based on whether each retrieved chunk contains a gold table."""
    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevance[:k], start=1))
    ideal_relevant = min(total_relevant, k)
    if ideal_relevant <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_retriever(
    examples: list[dict],
    retriever,
    top_k: int = 3,
    limit: int | None = None,
):
    """Evaluate the existing retriever without modifying its ranking logic.

    Table-grounded metrics:
    - Table Recall@K: fraction of required gold tables represented in Top-K context.
    - Table Precision@K: fraction of distinct retrieved tables that are required.
    - Table F1@K: harmonic mean of table precision and recall.
    - Hit Rate@K: fraction of questions with at least one relevant chunk retrieved.
    - Full Table Coverage@K: fraction of questions where every required table appears.
    - MRR: reciprocal rank of the first relevant retrieved chunk.
    - MAP@K: mean average precision over binary chunk relevance.
    - nDCG@K: rank-sensitive gain for relevant chunks.

    A retrieved chunk is considered relevant when it contains at least one table
    required by the example's gold SQL. MAP/nDCG use ``retriever.corpus`` to count
    the total number of relevant chunks when the corpus is exposed (as it is for
    this repository's retrievers).
    """
    selected = examples if limit is None else examples[:limit]
    rows = []
    corpus = getattr(retriever, "corpus", None)

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
        f1 = _safe_f1(precision, recall)

        relevance = [1 if gold_tables & tables else 0 for tables in chunk_tables]
        first_relevant_rank = next(
            (rank for rank, relevant in enumerate(relevance, start=1) if relevant),
            None,
        )
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        hit = 1.0 if first_relevant_rank is not None else 0.0
        full_coverage = 1.0 if gold_tables and gold_tables.issubset(retrieved_tables) else 0.0

        if corpus is not None:
            corpus_relevance = [
                1 if gold_tables & extract_chunk_tables(chunk) else 0 for chunk in corpus
            ]
            total_relevant = sum(corpus_relevance)
        else:
            # Fallback keeps the metric computable for third-party retrievers, but
            # the repository's own retrievers always expose a corpus.
            total_relevant = sum(relevance)

        average_precision = _average_precision_at_k(relevance, total_relevant, top_k)
        ndcg = _ndcg_at_k(relevance, total_relevant, top_k)

        rows.append(
            {
                "question": question,
                "gold_sql": gold_sql,
                "gold_tables": sorted(gold_tables),
                "retrieved_tables": sorted(retrieved_tables),
                f"table_recall@{top_k}": recall,
                f"table_precision@{top_k}": precision,
                f"table_f1@{top_k}": f1,
                f"hit@{top_k}": hit,
                f"full_table_coverage@{top_k}": full_coverage,
                "reciprocal_rank": reciprocal_rank,
                f"average_precision@{top_k}": average_precision,
                f"ndcg@{top_k}": ndcg,
                "retrieved_context": retrieved,
            }
        )

    total = len(rows) or 1
    metrics = {
        f"Table Recall@{top_k}": round(sum(r[f"table_recall@{top_k}"] for r in rows) / total, 4),
        f"Table Precision@{top_k}": round(sum(r[f"table_precision@{top_k}"] for r in rows) / total, 4),
        f"Table F1@{top_k}": round(sum(r[f"table_f1@{top_k}"] for r in rows) / total, 4),
        f"Hit Rate@{top_k}": round(sum(r[f"hit@{top_k}"] for r in rows) / total, 4),
        f"Full Table Coverage@{top_k}": round(
            sum(r[f"full_table_coverage@{top_k}"] for r in rows) / total, 4
        ),
        "MRR": round(sum(r["reciprocal_rank"] for r in rows) / total, 4),
        f"MAP@{top_k}": round(sum(r[f"average_precision@{top_k}"] for r in rows) / total, 4),
        f"nDCG@{top_k}": round(sum(r[f"ndcg@{top_k}"] for r in rows) / total, 4),
        "Evaluated Questions": len(rows),
    }
    return rows, metrics
