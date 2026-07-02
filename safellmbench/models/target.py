"""
Target model loader — any HuggingFace causal-LM model id can be plugged in
via the CLI. This is the model being *benchmarked* for safety.

Improvement over the original notebook: we always try to apply the target's
official chat template (via `tokenizer.apply_chat_template`) when it exposes
one. This is the fair way to prompt modern instruct models and prevents
"cheap" safety scores caused by feeding raw text to a chat-tuned model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TargetModel:
    model_id: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str

    def _build_inputs(self, user_prompt: str, system_prompt: Optional[str]) -> dict:
        # Prefer the model's own chat template when available.
        has_chat_tpl = getattr(self.tokenizer, "chat_template", None) is not None
        if has_chat_tpl:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = user_prompt
        return self.tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=2048).to(self.model.device)

    @torch.no_grad()
    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        inp = self._build_inputs(user_prompt, system_prompt)
        out = self.model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new = out[0][inp["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True).strip()


def load_target(model_id: str, device: Optional[str] = None,
                dtype: Optional[torch.dtype] = None) -> TargetModel:
    """Download and load an arbitrary HuggingFace causal LM."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[target] loading {model_id} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device)
    model.eval()
    return TargetModel(model_id=model_id, model=model,
                       tokenizer=tokenizer, device=device)
