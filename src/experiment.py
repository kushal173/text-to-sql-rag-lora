"""Batch generation orchestration shared by baseline and RAG experiments."""

from __future__ import annotations

from .prompts import PromptBuilder


class ExperimentRunner:
    def __init__(self, model_runner, evaluator):
        self.model_runner = model_runner
        self.evaluator = evaluator

    @staticmethod
    def clean_output(text: str) -> str:
        """Normalize decoded output before SQL extraction and evaluation."""
        return text.strip().replace("\n", " ").strip("`sql").strip()

    def run(
        self,
        examples: list[dict],
        schema_text: str | None = None,
        retriever=None,
        top_k: int = 4,
        max_new_tokens: int = 256,
        corrected_sql_extraction: bool = False,
    ) -> tuple[list[dict], dict]:
        predictions = []
        for number, example in enumerate(examples, 1):
            question = example["question"]
            if retriever is None:
                if schema_text is None:
                    prompt = PromptBuilder.finetune(question)
                else:
                    prompt = PromptBuilder.schema_or_rag(question, schema_text)
            else:
                retrieved_context = retriever.retrieve(question, top_k=top_k)
                prompt = PromptBuilder.schema_or_rag(question, retrieved_context)
            raw = self.model_runner.generate(prompt, max_new_tokens=max_new_tokens)
            generated = self.clean_output(raw)
            predictions.append(
                {
                    "question": question,
                    "ground_truth": example.get("query", example.get("gold_sql", "")),
                    "raw_completion": raw,
                    "generated_sql": generated,
                    "retrieved_context": retrieved_context if retriever is not None else None,
                }
            )
            print(f"{number}/{len(examples)} generated")
        return self.evaluator.evaluate(
            predictions, extract_first_statement=corrected_sql_extraction
        )
