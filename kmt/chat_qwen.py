"""Run Summit with a locally stored Qwen3 checkpoint; no remote inference calls."""
import argparse
import json
import os
from pathlib import Path

from kmt.qwen_runtime import DEFAULT_SYSTEM, QwenRuntime


def load_history(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    if not isinstance(value, list):
        raise ValueError(f"conversation history must be a JSON list: {path}")
    return value[-20:]


def save_history(path, messages):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(messages[-20:], ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def main(argv=None):
    local_default = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "SummitAI" / "models" / "Qwen3-0.6B-Base"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("SUMMIT_QWEN_MODEL", str(local_default)))
    parser.add_argument("--adapter", default=os.environ.get("SUMMIT_QWEN_ADAPTER"))
    parser.add_argument("--history", default="memory/qwen_history.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args(argv)

    runtime = QwenRuntime(args.model, args.adapter, args.device)
    messages = load_history(args.history)
    status = "fine-tuned Summit adapter" if args.adapter else "raw pretrained base (not instruction-tuned)"
    print(f"Summit Qwen ready on {runtime.device} using {status}. /reset clears this conversation; /quit exits.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in {"/quit", "/exit"}:
            break
        if text == "/reset":
            messages.clear()
            save_history(args.history, messages)
            print("Conversation cleared.")
            continue
        messages.append({"role": "user", "content": text})
        try:
            reply = runtime.respond(messages, max_new_tokens=args.max_new_tokens)
        except Exception as exc:
            messages.pop()
            print(f"Summit could not answer: {exc}")
            continue
        messages.append({"role": "assistant", "content": reply})
        save_history(args.history, messages)
        print(reply)


if __name__ == "__main__":
    main()