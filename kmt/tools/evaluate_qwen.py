"""Evaluate a local Summit Qwen checkpoint/adapter on held-out chat and translations."""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmt.evaluate import chrf
from kmt.distillation import canonical_solution, evaluate_expression, make_expressions
from kmt.qwen_runtime import QwenRuntime
from kmt.tokenizer import fold_text


ARITHMETIC_EVAL_SEED = 739391
ARITHMETIC_REGRESSION = "67897644457 * 1235656768979"
ARITHMETIC_REGRESSION_ANSWER = 83898183971021528899403


def token_f1(prediction, reference):
    pred = Counter(re.findall(r"\w+", prediction.lower()))
    ref = Counter(re.findall(r"\w+", reference.lower()))
    overlap = sum((pred & ref).values())
    if not pred or not ref:
        return 0.0
    precision, recall = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_chat(runtime, path, limit):
    exact, f1s, count = 0, [], 0
    with open(path, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            if len(messages) < 2 or messages[-1].get("role") != "assistant":
                continue
            target = messages[-1]["content"]
            prediction = runtime.respond(messages[:-1])
            exact += prediction.strip() == target.strip()
            f1s.append(token_f1(prediction, target))
            count += 1
            if count >= limit:
                break
    return {"examples": count, "exact_match": exact / max(1, count),
            "token_f1": sum(f1s) / max(1, len(f1s))}


def evaluate_translation(runtime, data_dir, limit):
    scores = {}
    for language in ("krio", "mende", "temne"):
        path = os.path.join(data_dir, "heldout", f"{language}-en.tsv")
        if not os.path.exists(path):
            continue
        values = []
        with open(path, encoding="utf-8-sig") as source:
            for line in source:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) < 2:
                    continue
                local, english = fields[:2]
                for src, text, tgt, reference in ((language, local, "English", english),
                                                   ("English", english, language.title(), local)):
                    prompt = [{"role": "user", "content": f"Translate from {src} to {tgt}:\n{text}"}]
                    prediction = runtime.respond(prompt, max_new_tokens=96)
                    values.append(chrf(fold_text(prediction).lower(), fold_text(reference).lower()))
                if len(values) >= limit * 2:
                    break
        scores[language] = {"examples": len(values), "chrF": sum(values) / max(1, len(values))}
    return scores


def evaluate_arithmetic(runtime, limit):
    expressions = make_expressions(limit, ARITHMETIC_EVAL_SEED) + [ARITHMETIC_REGRESSION]
    answer_correct = steps_correct = 0
    regression_result = None
    for expression in expressions:
        expected = evaluate_expression(expression)
        if expression == ARITHMETIC_REGRESSION and expected != ARITHMETIC_REGRESSION_ANSWER:
            raise AssertionError("independent arithmetic regression oracle disagrees with its fixed expected answer")
        prediction = runtime.respond(
            [{"role": "user", "content": (
                f"Evaluate exactly: {expression}\n"
                "Show the arithmetic steps and final answer. Put the final integer on an Answer: line."
            )}],
            max_new_tokens=192,
        )
        answers = re.findall(r"(?im)^\s*answer\s*:\s*([+-]?\d[\d,]*)\s*$", prediction)
        answer_ok = bool(answers) and int(answers[-1].replace(",", "")) == expected
        answer_correct += answer_ok

        canonical = canonical_solution(expression) if expression != ARITHMETIC_REGRESSION else None
        if canonical is not None:
            equation_pattern = r"(?m)^\s*(?:step\s+\d+\s*:\s*)?(-?\d+)\s*([+*x×-])\s*(-?\d+)\s*=\s*(-?\d+)\s*$"
            actual = [
                f"{left}{operator.replace('x', '*').replace('×', '*')}{right}={result}"
                for left, operator, right, result in re.findall(equation_pattern, prediction, re.I)
            ]
            expected_steps = [
                re.sub(r"\s+", "", step).replace("×", "*").replace("x", "*")
                for step in canonical["steps"]
            ]
            steps_correct += actual == expected_steps
        if expression == ARITHMETIC_REGRESSION:
            regression_result = {
                "expected": ARITHMETIC_REGRESSION_ANSWER,
                "answer_line": answers[-1] if answers else None,
                "correct": answer_ok,
            }

    return {
        "examples": len(expressions),
        "exact_answer_accuracy": answer_correct / len(expressions),
        "valid_step_accuracy": steps_correct / max(1, len(expressions) - 1),
        "large_number_regression": regression_result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local base checkpoint")
    parser.add_argument("--adapter")
    parser.add_argument("--data", default="data")
    parser.add_argument("--sft-data", default="data_qwen")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--math-limit", type=int, default=48)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    if args.limit < 1 or args.math_limit < 1:
        parser.error("--limit and --math-limit must be positive")
    runtime = QwenRuntime(args.model, args.adapter, args.device)
    report = {
        "model": os.path.abspath(args.model),
        "adapter": os.path.abspath(args.adapter) if args.adapter else None,
        "device": str(runtime.device),
        "chat_metrics": "exact match and token F1 against held-out references; these are overlap metrics, not human quality scores",
        "benchmarks": {
            "heldout_conversation": evaluate_chat(runtime, os.path.join(args.sft_data, "test.jsonl"), args.limit),
            "translation_chrF": evaluate_translation(runtime, args.data, args.limit),
            "arithmetic_reasoning": evaluate_arithmetic(runtime, args.math_limit),
            "coding": {"status": "not measured; no licensed coding benchmark configured"},
            "sierra_leone_culture": {"status": "not measured; requires locally reviewed benchmark"},
        },
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()