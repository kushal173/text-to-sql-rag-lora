"""Prompt templates for CodeLlama generation and fine-tuning."""

from __future__ import annotations

from collections.abc import Iterable


class PromptBuilder:
    @staticmethod
    def finetune(question: str) -> str:
        return (
            "### Instruction:\n"
            "Write an SQL query for the Chinook database.\n\n"
            f"### Input:\n{question}\n\n"
            "### Response:\n"
        )

    @staticmethod
    def training(instruction: str, input_text: str, output: str) -> str:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )

    @staticmethod
    def schema_or_rag(question: str, context: str | Iterable[str]) -> str:
        if not isinstance(context, str):
            context = "\n".join(context)
        return (
            "You are a SQL expert. Use the following context to write an accurate SQL query.\n\n"
            f"### Context:\n{context}\n\n"
            f"### Question:\n{question}\n\n"
            "### SQL:"
        )
