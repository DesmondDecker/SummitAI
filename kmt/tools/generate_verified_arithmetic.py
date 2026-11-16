"""Generate bounded, exactly verified arithmetic SFT rows from the local Ollama teacher."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmt.distillation import OllamaTeacher, canonical_solution, make_expressions


EVAL_SEED = 739391
EVAL_COUNT = 48
MAX_QUEUE_SIZE = 20000


def existing_expressions(path):
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                continue
            verification = record.get("verification")
            expression = verification.get("expression") if isinstance(verification, dict) else None
            if isinstance(expression, str):
                seen.add(expression)
    return seen


def _record(solution, teacher_model):
    canonical = canonical_solution(solution["expression"])
    if canonical != solution:
        raise ValueError("teacher result does not match the independently generated arithmetic trace")
    answer = "\n".join(
        f"Step {index}: {step}" for index, step in enumerate(canonical["steps"], 1)
    ) + f"\nAnswer: {canonical['answer']}"
    digest = hashlib.sha256(canonical["expression"].encode("utf-8")).hexdigest()
    return {
        "group_id": f"verified-arithmetic:{digest}",
        "source": "synthetic arithmetic curriculum",
        "teacher_model": teacher_model,
        "language": "en",
        "difficulty": "two-operation integer arithmetic",
        "review_status": "approved",
        "verification": {
            "method": "independent exact integer arithmetic and canonical AST trace",
            "expression": canonical["expression"],
            "answer": canonical["answer"],
        },
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Evaluate exactly: {canonical['expression']}\n"
                    "Show the arithmetic steps and final answer."
                ),
            },
            {"role": "assistant", "content": answer},
        ],
    }


def generate_rows(teacher, count, seed, reserved, existing):
    rows = []
    seen = set(existing) | set(reserved)
    candidate_seed = seed
    attempts = 0
    while len(rows) < count and attempts < 100:
        expressions = [
            expression
            for expression in make_expressions(max(16, (count - len(rows)) * 2), candidate_seed)
            if expression not in seen
        ]
        candidate_seed += 1
        attempts += 1
        if not expressions:
            continue
        seen.update(expressions)
        for start in range(0, len(expressions), 8):
            solutions = teacher.solve(expressions[start:start + 8])
            rows.extend(_record(solution, teacher.model) for solution in solutions)
            if len(rows) >= count:
                return rows[:count]
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", default="data_qwen/teacher_verified.jsonl")
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--max-queue-size", type=int, default=MAX_QUEUE_SIZE)
    args = parser.parse_args(argv)
    if not 1 <= args.examples <= 256:
        parser.error("--examples must be between 1 and 256")
    if args.max_queue_size < args.examples or args.max_queue_size > MAX_QUEUE_SIZE:
        parser.error(f"--max-queue-size must be between --examples and {MAX_QUEUE_SIZE}")

    reserved = set(make_expressions(EVAL_COUNT, EVAL_SEED))
    path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = existing_expressions(path)
    if len(existing) + args.examples > args.max_queue_size:
        raise SystemExit("teacher queue cap reached; inspect the queue before increasing its configured limit")

    teacher = OllamaTeacher(args.model, args.ollama_url)
    rows = generate_rows(teacher, args.examples, args.seed, reserved, existing)
    if not rows:
        raise SystemExit("the teacher produced no independently verified examples")
    with open(path, "a", encoding="utf-8") as target:
        target.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        target.flush()
        os.fsync(target.fileno())
    print(f"appended {len(rows)} exact-verified arithmetic examples to {path}")
    if len(rows) != args.examples:
        print(f"teacher verification accepted {len(rows)}/{args.examples} requested examples", file=sys.stderr)


if __name__ == "__main__":
    main()
