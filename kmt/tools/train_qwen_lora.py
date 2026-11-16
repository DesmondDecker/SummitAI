"""Supervised LoRA fine-tuning for Summit's local Qwen3 checkpoint."""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmt.qwen_runtime import DEFAULT_SYSTEM


def load_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def load_approved_teacher_rows(path, limit):
    if not path or not os.path.exists(path) or limit == 0:
        return []
    approved = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            if row.get("review_status") != "approved":
                continue
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            if any(
                not isinstance(message, dict)
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
                for message in messages
            ):
                continue
            if messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
                continue
            if any(left["role"] == right["role"] for left, right in zip(messages, messages[1:])):
                continue
            approved.append(row)
            if len(approved) >= limit:
                break
    return approved


def estimate_parameter_count(config):
    """Approximate dense Qwen parameters from its public architecture config."""
    hidden = getattr(config, "hidden_size", None)
    layers = getattr(config, "num_hidden_layers", None)
    vocab = getattr(config, "vocab_size", None)
    intermediate = getattr(config, "intermediate_size", None)
    heads = getattr(config, "num_attention_heads", None)
    if not all(isinstance(value, int) and value > 0
               for value in (hidden, layers, vocab, intermediate, heads)):
        return None
    head_dim = getattr(config, "head_dim", None) or hidden // heads
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    q_width, kv_width = heads * head_dim, kv_heads * head_dim
    tied = bool(getattr(config, "tie_word_embeddings", False))
    embeddings = vocab * hidden * (1 if tied else 2)
    attention = hidden * q_width + 2 * hidden * kv_width + q_width * hidden
    mlp = 3 * hidden * intermediate
    norms = 8 * hidden
    return embeddings + layers * (attention + mlp + norms)


def encode_example(tokenizer, row, max_length):
    messages = [{"role": "system", "content": DEFAULT_SYSTEM}, *row["messages"]]
    if messages[-1]["role"] != "assistant":
        return None
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError("Qwen chat template prefix mismatch; refusing to mis-mask SFT labels")
    if len(full) > max_length or len(full) <= len(prompt):
        return None
    labels = [-100] * len(prompt) + full[len(prompt):]
    return {"input_ids": full, "labels": labels}


def collate(examples, pad_id, multiple=8):
    import torch
    width = max(len(row["input_ids"]) for row in examples)
    width = min(((width + multiple - 1) // multiple) * multiple, 1 << 30)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for index, row in enumerate(examples):
        size = len(row["input_ids"])
        ids[index, :size] = torch.tensor(row["input_ids"], dtype=torch.long)
        labels[index, :size] = torch.tensor(row["labels"], dtype=torch.long)
        mask[index, :size] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def encode_rows(rows, tokenizer, max_length):
    encoded, skipped = [], 0
    for row in rows:
        value = encode_example(tokenizer, row, max_length)
        if value is None:
            skipped += 1
        else:
            encoded.append(value)
    return encoded, skipped


def evaluate(model, rows, pad_id, device, max_batches=32):
    import torch
    model.eval()
    losses = []
    with torch.inference_mode():
        for start in range(0, min(len(rows), max_batches * 2), 2):
            batch = collate(rows[start:start + 2], pad_id)
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="local Qwen base checkpoint directory")
    parser.add_argument("--data", default="data_qwen")
    parser.add_argument("--output", default="outputs/summit-qwen3-0.6b-lora")
    parser.add_argument("--base-model-id", default="")
    parser.add_argument("--base-revision", default="")
    parser.add_argument("--base-license", default="unknown")
    parser.add_argument("--resume-adapter", default="", help="continue training from an existing compatible LoRA adapter")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--max-validation", type=int, default=512)
    parser.add_argument("--approved-teacher-data", default="", help="optional reviewed teacher JSONL; only review_status=approved rows load")
    parser.add_argument("--max-teacher-examples", type=int, default=0, help="teacher rows to add; default 0 disables distillation")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if min(args.max_steps, args.max_examples, args.max_validation, args.max_length,
           args.batch_size, args.gradient_accumulation, args.eval_every) < 1:
        parser.error("training sizes and intervals must be positive")
    if args.max_teacher_examples < 0:
        parser.error("max-teacher-examples cannot be negative")
    if not os.path.isfile(os.path.join(args.data, "manifest.json")):
        parser.error("dataset manifest missing; run tools/prepare_qwen_data.py first")
    if not os.path.isdir(args.base_model):
        parser.error("base model path must be a local directory; download the official checkpoint first")

    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install requirements-qwen.txt in the isolated .venv-qwen environment first.") from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but this Python environment has no CUDA-enabled device.")
    config = AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    estimated_parameters = estimate_parameter_count(config)
    if device == "cpu" and estimated_parameters and estimated_parameters >= 1_000_000_000:
        raise SystemExit(
            f"Refusing CPU fine-tuning for an estimated {estimated_parameters:,}-parameter model. "
            "Use a CUDA GPU for 3B/4B training; this guard avoids loading the full model into limited system RAM."
        )
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, local_files_only=True, attn_implementation="eager"
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    base_parameters = sum(parameter.numel() for parameter in base.parameters())
    if args.resume_adapter:
        if not os.path.isdir(args.resume_adapter):
            raise SystemExit(f"resume adapter directory does not exist: {args.resume_adapter}")
        model = PeftModel.from_pretrained(
            base, args.resume_adapter, is_trainable=True, local_files_only=True
        )
    else:
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base, lora)
    model = model.to(device)
    model.print_trainable_parameters()
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    raw_train = load_jsonl(os.path.join(args.data, "train.jsonl"), args.max_examples)
    teacher_rows = load_approved_teacher_rows(args.approved_teacher_data, args.max_teacher_examples)
    raw_train.extend(teacher_rows)
    raw_val = load_jsonl(os.path.join(args.data, "validation.jsonl"), args.max_validation)
    train_rows, skipped_train = encode_rows(raw_train, tokenizer, args.max_length)
    val_rows, skipped_val = encode_rows(raw_val, tokenizer, args.max_length)
    if not train_rows or not val_rows:
        raise SystemExit("non-empty train and validation splits are required")
    print(f"device={device} dtype={dtype} train={len(train_rows)} val={len(val_rows)} "
          f"skipped_long={skipped_train + skipped_val}", flush=True)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, "training.jsonl")
    best_loss = float("inf")
    step = 0
    accumulated = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    start_time = time.time()
    while step < args.max_steps:
        random.shuffle(train_rows)
        for offset in range(0, len(train_rows), args.batch_size):
            chunk = train_rows[offset:offset + args.batch_size]
            batch = collate(chunk, tokenizer.pad_token_id or tokenizer.eos_token_id)
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / args.gradient_accumulation
            loss.backward()
            accumulated += 1
            if accumulated < args.gradient_accumulation and offset + args.batch_size < len(train_rows):
                continue
            if accumulated < args.gradient_accumulation:
                scale = args.gradient_accumulation / accumulated
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            step += 1
            if step % args.eval_every == 0 or step == args.max_steps:
                val_loss = evaluate(model, val_rows, tokenizer.pad_token_id or tokenizer.eos_token_id, device)
                event = {"step": step, "train_loss": float(loss.detach().cpu()) * args.gradient_accumulation,
                         "validation_loss": val_loss, "elapsed_s": round(time.time() - start_time, 1)}
                print(json.dumps(event), flush=True)
                with open(log_path, "a", encoding="utf-8") as log:
                    log.write(json.dumps(event) + "\n")
                if val_loss < best_loss:
                    best_loss = val_loss
                    model.save_pretrained(os.path.join(args.output, "adapter"), safe_serialization=True)
                    tokenizer.save_pretrained(os.path.join(args.output, "tokenizer"))
                if step >= args.max_steps:
                    break
        if step >= args.max_steps:
            break

    manifest = {
        "base_model": args.base_model_id or getattr(config, "_name_or_path", "") or os.path.abspath(args.base_model),
        "base_model_path": os.path.abspath(args.base_model),
        "base_revision": args.base_revision or getattr(config, "_commit_hash", None),
        "base_license": args.base_license,
        "base_parameters": base_parameters,
        "estimated_parameters_from_config": estimated_parameters,
        "trainable_parameters": trainable_parameters,
        "training_sources": json.load(open(os.path.join(args.data, "manifest.json"), encoding="utf-8"))["sources"],
        "method": "LoRA supervised fine-tuning",
        "approved_teacher_examples": len(teacher_rows),
        "lora_rank": 8,
        "steps": step,
        "best_validation_loss": best_loss,
        "device": device,
        "dtype": str(dtype),
        "training_examples": len(train_rows),
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    with open(os.path.join(args.output, "model_manifest.json"), "w", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2)
    print(f"saved best adapter; validation loss={best_loss:.4f}; base weights were not modified")


if __name__ == "__main__":
    main()