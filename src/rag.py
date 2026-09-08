"""FAISS retrieval for schema and example augmentation."""

from __future__ import annotations

from typing import Any

import numpy as np


class ChinookRetriever:
    """SentenceTransformer embeddings plus an unnormalised FAISS L2 index."""

    def __init__(self, corpus: list[str], model_name: str = "all-MiniLM-L6-v2"):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.corpus = corpus
        self.model = SentenceTransformer(model_name)
        embeddings = self.model.encode(corpus, convert_to_numpy=True)
        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.index: Any = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    @staticmethod
    def build_corpus(
        schema: dict[str, str],
        training_examples: list[dict],
        example_limit: int = 50,
    ) -> list[str]:
        # Keep schema chunks before example chunks for deterministic indexing.
        corpus = ["Schema:"]
        corpus.extend(schema.values())
        for index, item in enumerate(training_examples[:example_limit], 1):
            sql = item.get("query", item.get("gold_sql", ""))
            corpus.append(
                f"Example {index} - Question: {item['question']} SQL: {sql}"
            )
        return corpus

    def retrieve(self, question: str, top_k: int = 4) -> list[str]:
        query = self.model.encode([question], convert_to_numpy=True)
        _, indices = self.index.search(np.asarray(query, dtype="float32"), top_k)
        return [self.corpus[index] for index in indices[0]]
