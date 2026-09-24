"""Re-run the original Mistral + Chinook RAG experiment with expanded metrics.

This script intentionally preserves the old experiment settings:
- model: mistralai/Mistral-7B-Instruct-v0.2
- embeddings: all-MiniLM-L6-v2
- FAISS L2 retriever
- original knowledge base
- original prompt builder
- top_k = 3
- same 30 held-out Chinook questions

Only evaluation instrumentation has been added.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import (
    MistralExperiment,
    MistralKnowledgeRetriever,
    MistralRunner,
    MistralSQLEvaluator,
    evaluate_retriever,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test subset; omit for all 30")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    prompts = MistralExperiment.load_prompts(root / "prompts/Mistral7b/chinook_prompts.json")
    selected_count = len(prompts) if args.limit is None else min(args.limit, len(prompts))

    retriever = MistralKnowledgeRetriever(
        root / "knowledge/chinook_knowledge_base.txt",
        dataset="chinook",
        model_name="all-MiniLM-L6-v2",
    )

    retrieval_rows, retrieval_metrics = evaluate_retriever(
        prompts,
        retriever,
        top_k=3,
        limit=args.limit,
    )

    runner = MistralRunner.load_4bit()
    experiment = MistralExperiment(runner)
    evaluator = MistralSQLEvaluator(root / "datasets")

    generation_rows = experiment.run(
        prompts,
        retriever=retriever,
        top_k=3,
        limit=args.limit,
    )
    evaluated_rows, generation_metrics = evaluator.evaluate(
        generation_rows,
        dataset="chinook",
    )

    payload = {
        "experiment": {
            "name": "Original Mistral-7B-Instruct-v0.2 + old Chinook RAG, expanded evaluation",
            "questions": selected_count,
            "model": MistralRunner.MODEL_NAME,
            "embedding_model": "all-MiniLM-L6-v2",
            "retriever": "FAISS IndexFlatL2",
            "top_k": 3,
            "pipeline_changed": False,
            "note": "Only evaluation metrics/instrumentation were added.",
        },
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "retrieval_results": retrieval_rows,
        "generation_results": evaluated_rows,
    }

    output = root / "results/mistral_chinook_old_rag_complete_metrics.json"
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n=== RETRIEVAL METRICS ===")
    print(json.dumps(retrieval_metrics, indent=2))
    print("\n=== GENERATION / RAG METRICS ===")
    print(json.dumps(generation_metrics, indent=2))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
