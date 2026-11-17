"""Local Qwen-backed Summit runtime. Model weights are loaded only from disk."""
from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_SYSTEM = (
    "You are Summit AI, an assistant focused on useful, accurate help in English and Krio. "
    "Be clear about uncertainty. Do not invent Krio translations or claim local expertise you cannot support."
)


def resolve_adapter_path(adapter_path):
    """Resolve the active-adapter pointer while preventing paths outside its model directory."""
    if not adapter_path:
        return adapter_path
    root = Path(adapter_path).resolve()
    pointer = root / "active.json"
    if not pointer.is_file():
        return adapter_path
    with pointer.open(encoding="utf-8") as source:
        manifest = json.load(source)
    relative = manifest.get("adapter")
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"active adapter manifest has no adapter path: {pointer}")
    candidate = (root / relative).resolve()
    if os.path.commonpath((str(root), str(candidate))) != str(root):
        raise ValueError("active adapter path must remain inside its model directory")
    if not candidate.is_dir():
        raise FileNotFoundError(f"active adapter directory does not exist: {candidate}")
    return str(candidate)


def normalize_messages(messages, system=DEFAULT_SYSTEM):
    """Validate chat turns and ensure one leading system message."""
    if not isinstance(system, str) or not system.strip():
        system = DEFAULT_SYSTEM
    normalized = [{"role": "system", "content": system}]
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            raise ValueError("conversation turns must have role 'user' or 'assistant'")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("conversation turn content must be non-empty text")
        normalized.append({"role": message["role"], "content": content.strip()})
    if len(normalized) == 1 or normalized[-1]["role"] != "user":
        raise ValueError("a conversation must end with a user message")
    return normalized


def trim_history(messages, token_count, max_input_tokens):
    """Drop oldest complete user/assistant turns; never truncate message text silently."""
    kept = list(messages)
    while token_count(kept) > max_input_tokens:
        if len(kept) < 4:
            raise ValueError("system prompt and latest user message exceed the model context window")
        del kept[1:3]
    return kept


class QwenRuntime:
    """Transformers runtime for a local Qwen3 checkpoint and optional local LoRA adapter."""

    def __init__(self, model_path, adapter_path=None, device="auto", local_files_only=True):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install requirements-qwen.txt in the selected Python environment") from exc

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA-enabled PyTorch device is available")
        self.device = torch.device(device)
        dtype = "auto" if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
            attn_implementation="eager",
        )
        if adapter_path:
            adapter_path = resolve_adapter_path(adapter_path)
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("install peft from requirements-qwen.txt to load this adapter") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path, local_files_only=local_files_only)

        self.model.to(self.device)
        self.model.eval()
        self.context_length = int(getattr(self.model.config, "max_position_embeddings", 32768))
        if not getattr(self.tokenizer, "chat_template", None):
            raise RuntimeError("the selected tokenizer has no chat template; use the official Qwen3 tokenizer files")

    def _encode(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

    def respond(self, messages, system=DEFAULT_SYSTEM, max_new_tokens=256):
        if not 1 <= max_new_tokens < self.context_length:
            raise ValueError("max_new_tokens must be positive and smaller than the model context")
        chat = normalize_messages(messages, system)

        def count(items):
            return self._encode(items).shape[-1]

        chat = trim_history(chat, count, self.context_length - max_new_tokens)
        input_ids = self._encode(chat)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                              else self.tokenizer.eos_token_id),
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        answer_ids = output[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
