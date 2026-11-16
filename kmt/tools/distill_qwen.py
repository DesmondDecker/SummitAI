"""Generate local Qwen teacher candidates for human review; never auto-trains them."""
import argparse
import json
import os
import sys
import urllib.request
from urllib.parse import urlsplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kmt.qwen_runtime import DEFAULT_SYSTEM


def teacher_answer(messages, model, url="http://127.0.0.1:11434", timeout=180):
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("teacher endpoint must be local Ollama")
    request_messages = [{"role": "system", "content": DEFAULT_SYSTEM}, *messages]
    body = json.dumps({
        "model": model,
        "messages": request_messages,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 512},
    }).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = json.loads(response.read())["message"]["content"]
    if not isinstance(content, str) or not content.strip() or len(content) > 20000:
        raise ValueError("teacher returned empty or oversized content")
    return content.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data_qwen/train.jsonl")
    parser.add_argument("--output", default="data_qwen/teacher_review.jsonl")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 1000:
        parser.error("--limit must be between 1 and 1000")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    count = 0
    with open(args.input, encoding="utf-8") as source, open(args.output, "a", encoding="utf-8") as output:
        for line in source:
            if not line.strip():
                continue
            sample = json.loads(line)
            messages = sample.get("messages", [])
            if len(messages) < 2 or messages[-1].get("role") != "assistant":
                continue
            prompt = messages[:-1]
            if not prompt or prompt[-1].get("role") != "user":
                continue
            try:
                answer = teacher_answer(prompt, args.model, args.ollama_url)
            except Exception as exc:
                print(f"skipped sample after teacher error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                continue
            record = {
                "messages": [*prompt, {"role": "assistant", "content": answer}],
                "source": "local Ollama teacher candidate",
                "teacher_model": args.model,
                "upstream_source": sample.get("source"),
                "review_status": "pending",
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            count += 1
            if count % 10 == 0:
                print(f"wrote {count} pending review examples", flush=True)
            if count >= args.limit:
                break
    print(f"wrote {count} pending examples to {args.output}; they are not trainable until reviewed and approved")


if __name__ == "__main__":
    main()