"""CodeLlama loading, generation and optional LoRA training."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CodeLlamaRunner:
    MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load_adapter(
        cls,
        adapter_path: str | Path,
        model_name: str = MODEL_NAME,
    ) -> "CodeLlamaRunner":
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
        return cls(model, tokenizer)

    @classmethod
    def load_base(cls, model_name: str = MODEL_NAME) -> "CodeLlamaRunner":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        return cls(model, tokenizer)

    @staticmethod
    def _input_device(model: Any) -> Any:
        # The embedding layer is the safe input target when Accelerate dispatches layers.
        return model.get_input_embeddings().weight.device

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device(self.model))
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Decode only the completion by removing the known prompt prefix.
        return decoded.replace(prompt, "", 1).strip()

    @classmethod
    def build_trainable_lora(cls, model_name: str = MODEL_NAME):
        """Build a full-FP16 LoRA model; this generally needs more VRAM than a T4."""
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, padding_side="right", truncation_side="right"
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        return get_peft_model(model, config), tokenizer
