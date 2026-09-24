"""Reusable Mistral-7B generation and retrieval components."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any


class MistralPromptBuilder:
    """Prompt strings shared by the baseline and RAG experiments."""

    @staticmethod
    def baseline(schema: Any, question: str) -> str:
        return (
            f"Database Schema {schema}, Based on the provided database schema "
            f"information, {question}\n ### SQL:"
        )

    @staticmethod
    def rag(schema: Any, context: Any, question: str) -> str:
        return (
            f"Database Schema {schema}, ###Context {context} Based on the provided "
            f"database schema information, {question}\n ### SQL:"
        )


class MistralRunner:
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load_4bit(cls, model_name: str = MODEL_NAME) -> "MistralRunner":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return cls(model, tokenizer)

    def generate(self, prompt: str, max_new_tokens: int = 500) -> str:
        """Decode the complete generated sequence for subsequent SQL extraction."""
        import torch

        device = self.model.get_input_embeddings().weight.device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @staticmethod
    def extract_sql(text: str) -> str:
        """Extract the first fenced SQL block when one is present."""
        matches = re.findall(r"```(.*?)\n(.*?)(```|$)", text, re.DOTALL)
        if matches:
            clean_sql = matches[0][1]
        else:
            completion = text.rsplit("### SQL:", 1)[-1]
            start = re.search(r"\b(SELECT|WITH)\b", completion, flags=re.I)
            if not start:
                return ""
            clean_sql = completion[start.start():]
            semicolon = clean_sql.find(";")
            if semicolon >= 0:
                clean_sql = clean_sql[: semicolon + 1]
        clean_sql = clean_sql.strip().replace("\n", " ").replace("\t", " ")
        return re.sub(r"\s+", " ", clean_sql)


class MistralKnowledgeRetriever:
    """FAISS L2 retriever over complete knowledge chunks."""

    def __init__(
        self,
        knowledge_path: str | Path,
        dataset: str,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        text = Path(knowledge_path).read_text(encoding="utf-8")
        if dataset == "chinook":
            self.corpus = ast.literal_eval(text)
            embedding_texts = [str(item) for item in self.corpus]
        else:
            self.corpus = ast.literal_eval(text)
            embedding_texts = list(self.corpus)

        self.dataset = dataset
        self.model = SentenceTransformer(model_name)
        embeddings = self.model.encode(embedding_texts, convert_to_numpy=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, question: str, top_k: int = 3) -> list[Any]:
        import numpy as np

        query = self.model.encode([question], convert_to_numpy=True)
        _, indices = self.index.search(np.asarray(query, dtype="float32"), top_k)
        return [self.corpus[index] for index in indices[0] if index >= 0]


class MistralExperiment:
    def __init__(self, runner: MistralRunner):
        self.runner = runner

    @staticmethod
    def load_prompts(path: str | Path) -> list[dict]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def run(
        self,
        prompts: list[dict],
        retriever: MistralKnowledgeRetriever | None = None,
        top_k: int = 3,
        limit: int | None = None,
    ) -> list[dict]:
        rows = []
        selected = prompts if limit is None else prompts[:limit]
        for index, example in enumerate(selected, 1):
            if retriever is None:
                prompt = MistralPromptBuilder.baseline(
                    example["db_schema"], example["question"]
                )
            else:
                context = retriever.retrieve(example["question"], top_k=top_k)
                prompt = MistralPromptBuilder.rag(
                    example["db_schema"], context, example["question"]
                )
            raw = self.runner.generate(prompt, max_new_tokens=500)
            rows.append(
                {
                    "db_id": example["db_id"],
                    "natural_language_question": example["question"],
                    "gold_sql": example["gold_sql"],
                    "raw_model_output": raw,
                    "clean_sql_extracted": self.runner.extract_sql(raw),
                    "retrieved_context": context if retriever is not None else None,
                }
            )
            print(f"{index}/{len(selected)} completed")
        return rows
