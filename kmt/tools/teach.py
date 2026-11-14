"""Distil a verified arithmetic curriculum from local Ollama into Summit.

One bounded round by default. Use --continuous for a stoppable background loop.
Only local Ollama is accepted; generated answers are independently verified.
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from kmt.data import Dataset, qa_example, translate_prompt
from kmt.distillation import (OllamaTeacher, canonical_solution, evaluate_expression, make_expressions,
                              validate_solution)
from kmt.evaluate import chrf
from kmt.tokenizer import BPE, fold_text
from kmt.transformer import Transformer


SAFE_SOURCES = ("migeod1908:", "peacecorps_", "mbsse_syllabus2020")
EVAL_SEED = 739391
MAX_STORED_EXAMPLES = 20000


class CurriculumLimitReached(RuntimeError):
    pass


def _read_pairs(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 3 or not any(fields[2].startswith(prefix) for prefix in SAFE_SOURCES):
                continue
            try:
                quality = float(fields[3]) if len(fields) > 3 else 1.0
            except ValueError:
                continue
            if quality >= 0.5 and fields[0].strip() and fields[1].strip():
                rows.append((fields[0].strip(), fields[1].strip()))
    return rows


def prepare_replay_data(source_dir, target_dir):
    """Copy only explicitly rights-cleared, quality-screened pairs into a temporary dataset."""
    os.makedirs(target_dir, exist_ok=True)
    for lang in ("en", "krio", "mende", "temne"):
        with open(os.path.join(target_dir, lang + ".txt"), "w", encoding="utf-8") as out:
            out.write("# No monolingual data used by the distillation replay set.\n")
    counts = {}
    for split in ("parallel", "heldout"):
        split_dir = os.path.join(target_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for lang in ("krio", "mende", "temne"):
            rows = _read_pairs(os.path.join(source_dir, split, f"{lang}-en.tsv"))
            counts[(split, lang)] = len(rows)
            with open(os.path.join(split_dir, f"{lang}-en.tsv"), "w", encoding="utf-8") as out:
                out.write(f"# filtered provenance-cleared {lang}-English pairs\n")
                out.writelines(f"{local}\t{english}\n" for local, english in rows)
    return counts


def _expected_record(expression):
    return canonical_solution(expression)


def _qa_record(tok, solution):
    answer = "\n".join(
        f"Step {index}: {step}" for index, step in enumerate(solution["steps"], 1)
    ) + f"\nAnswer: {solution['answer']}"
    return qa_example(
        tok,
        "en",
        f"Evaluate exactly: {solution['expression']}",
        "Show the arithmetic steps and final answer.",
        answer,
    )


def arithmetic_scores(model, tok, expressions):
    answer_correct = chain_correct = 0
    ban = [tok.pad, tok.unk, tok.bos, tok.sep] + [
        tok.id(token) for token in ("<tr>", "<intent>", "<qa>", "<krio>", "<mende>", "<temne>", "<en>")
    ]
    for expression in expressions:
        prompt = [tok.bos, tok.id("<qa>"), tok.id("<en>")] + tok.encode(
            f"Evaluate exactly: {expression}"
        ) + [tok.sep] + tok.encode("Show the arithmetic steps and final answer.") + [tok.sep]
        if len(prompt) >= model.cfg["ctx"] - 2:
            continue
        output, _ = model.generate(prompt, 48, tok.eos, ban)
        text = tok.decode(output)
        match = re.search(r"(?i)answer\s*:\s*(-?\d+)\b", text)
        if match and int(match.group(1)) == evaluate_expression(expression):
            answer_correct += 1
        equation_pattern = r"(-?\d+)\s*([+*x×-])\s*(-?\d+)\s*=\s*(-?\d+)"
        normalize = lambda value: value.replace("x", "*").replace("×", "*").replace("−", "-")
        expected = [normalize(row) for row in _expected_record(expression)["steps"]]
        actual = [f"{a}{operator}{b}={result}" for a, operator, b, result in re.findall(equation_pattern, text)]
        if actual == expected:
            chain_correct += 1
    denominator = max(1, len(expressions))
    return {"answer_accuracy": answer_correct / denominator,
            "valid_step_accuracy": chain_correct / denominator}


def translation_chrf(model, tok, data_dir, cap=12):
    scores = []
    ban = [tok.pad, tok.unk, tok.bos, tok.sep] + [
        tok.id(token) for token in ("<tr>", "<intent>", "<qa>", "<krio>", "<mende>", "<temne>", "<en>")
    ]
    for lang in ("krio", "mende", "temne"):
        pairs = _read_pairs(os.path.join(data_dir, "heldout", f"{lang}-en.tsv"))[:cap]
        for local, english in pairs:
            for source_lang, source, target_lang, reference in (
                (lang, local, "en", english), ("en", english, lang, local)
            ):
                prefix = translate_prompt(tok, source_lang, source, target_lang)
                if len(prefix) >= model.cfg["ctx"] - 2:
                    continue
                output, _ = model.generate(prefix, min(64, max(12, 2 * len(tok.encode(source)) + 12)), tok.eos, ban)
                hypothesis = fold_text(tok.decode(output)).lower()
                scores.append(chrf(hypothesis, fold_text(reference).lower()))
    return float(np.mean(scores)) if scores else None


def _save_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as out:
        json.dump(value, out, indent=2)
    os.replace(temporary, path)


def _load_verified(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                try:
                    record = json.loads(line)
                    records.append(validate_solution(record["expression"], record))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
    return records


def _train_round(model, tok, new_solutions, old_solutions, replay, args, round_number):
    teacher_examples = [
        example for example in (_qa_record(tok, row) for row in old_solutions + new_solutions)
        if len(example["ids"]) <= model.cfg["ctx"]
    ]
    if not teacher_examples:
        raise RuntimeError("all verified teacher examples exceed the model context window")
    rng = np.random.default_rng(args.seed + round_number)
    optimizer = __import__("kmt.optim", fromlist=["Adam"]).Adam(model.parameters(), lr=args.lr, wd=0.0)
    qa_count = args.batch // 2
    replay_count = args.batch - qa_count
    for step in range(1, args.steps + 1):
        examples = [teacher_examples[i] for i in rng.integers(0, len(teacher_examples), qa_count)]
        if replay_count:
            examples.extend(replay[i] for i in rng.integers(0, len(replay), replay_count))
        ids, mask = Dataset.collate(examples, tok.pad)
        loss = model.loss(ids, mask, train=True)
        optimizer.zero_grad()
        loss.backward()
        norm = optimizer.step()
        if not np.isfinite(float(loss.data)) or not np.isfinite(norm):
            raise RuntimeError(f"non-finite training value at step {step}")
        if step % 25 == 0 or step == args.steps:
            print(f"  train step {step}/{args.steps}, loss {float(loss.data):.3f}", flush=True)


def _run_round(args, teacher, data_dir, workdir, round_number):
    token_path = os.path.join(args.ckpt, "tokenizer.json")
    active_path = os.path.join(args.ckpt, "kmt.npz")
    candidate_path = os.path.join(workdir, "candidate.npz")
    stored_path = os.path.join(workdir, "verified_arithmetic.jsonl")
    metrics_path = os.path.join(workdir, "best_metrics.json")
    tok = BPE.load(token_path)
    active_model, active_extra = Transformer.load(active_path)
    heldout = make_expressions(args.eval_examples, EVAL_SEED)
    if os.path.exists(candidate_path):
        model, extra = Transformer.load(candidate_path)
        if model.cfg != active_model.cfg:
            raise ValueError("saved candidate architecture differs from the active checkpoint")
    else:
        model, extra = active_model, active_extra

    records = _load_verified(stored_path)
    if len(records) + args.examples > MAX_STORED_EXAMPLES:
        raise CurriculumLimitReached(
            f"verified-example cap ({MAX_STORED_EXAMPLES}) reached; inspect/archive {stored_path} before continuing"
        )
    seen = {record["expression"] for record in records} | set(heldout)
    solutions = []
    seed = args.seed + round_number * 1009
    attempt_limit = args.examples * 8
    attempted = 0
    while len(solutions) < args.examples and attempted < attempt_limit:
        request_count = min(args.teacher_batch, args.examples - len(solutions), attempt_limit - attempted)
        candidates = [expression for expression in make_expressions(request_count * 2, seed)
                      if expression not in seen][:request_count]
        seed += 1
        if not candidates:
            continue
        seen.update(candidates)
        attempted += len(candidates)
        for index in range(0, len(candidates), args.teacher_batch):
            for solution in teacher.solve(candidates[index:index + args.teacher_batch]):
                if solution["expression"] in candidates:
                    solutions.append(solution)
    if len(solutions) < args.examples:
        raise RuntimeError(
            f"teacher verified only {len(solutions)}/{args.examples} examples after {attempted} attempts; "
            "lower --examples or inspect Ollama/model quality"
        )
    solutions = solutions[:args.examples]

    with open(stored_path, "a", encoding="utf-8") as out:
        for solution in solutions:
            out.write(json.dumps(solution) + "\n")
    prior_metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as source:
            prior_metrics = json.load(source)
    if prior_metrics is None:
        prior_metrics = {
            **arithmetic_scores(active_model, tok, heldout),
            "translation_chrf": translation_chrf(active_model, tok, data_dir),
        }
        _save_json(metrics_path, prior_metrics)
    before = {
        **arithmetic_scores(model, tok, heldout),
        "translation_chrf": translation_chrf(model, tok, data_dir),
    }

    with tempfile.TemporaryDirectory(prefix="summit-replay-") as temp_data:
        counts = prepare_replay_data(data_dir, temp_data)
        replay_ds = Dataset(temp_data, tok, model.cfg["ctx"], mono_frac=0.0)
        replay = replay_ds.train
        if not replay:
            raise RuntimeError("no provenance-cleared replay examples available; refusing to train without retention data")
        print(f"  replay examples: {len(replay)}; filtered sources only: {counts}", flush=True)
        _train_round(model, tok, solutions, records, replay, args, round_number)

    after = {
        **arithmetic_scores(model, tok, heldout),
        "translation_chrf": translation_chrf(model, tok, data_dir),
    }
    model.save(candidate_path, {**extra, "distillation_round": round_number})
    accepted = (
        after["answer_accuracy"] >= prior_metrics["answer_accuracy"] + 1 / max(1, args.eval_examples)
        and after["valid_step_accuracy"] >= prior_metrics["valid_step_accuracy"] + 1 / max(1, args.eval_examples)
        and after["translation_chrf"] is not None
        and prior_metrics["translation_chrf"] is not None
        and after["translation_chrf"] >= prior_metrics["translation_chrf"] - args.max_translation_drop
    )
    report = {
        "round": round_number,
        "verified_new_examples": len(solutions),
        "baseline": prior_metrics,
        "candidate_before": before,
        "candidate_after": after,
        "promoted": accepted,
    }
    _save_json(os.path.join(workdir, f"round-{round_number:04d}.json"), report)
    print("  metrics:", json.dumps(report, sort_keys=True), flush=True)
    if accepted:
        backup = active_path + ".pre_distillation.bak"
        if not os.path.exists(backup):
            shutil.copy2(active_path, backup)
        model.save(active_path, {**extra, "distillation_round": round_number})
        _save_json(metrics_path, after)
        os.remove(candidate_path)
        print("  promoted candidate; original checkpoint backup preserved", flush=True)
    else:
        print("  candidate retained for the next round; active checkpoint unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="checkpoints")
    parser.add_argument("--data", default="data")
    parser.add_argument("--workdir", default="runs/distillation")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--continuous", action="store_true", help="repeat until Ctrl+C or a STOP file appears in --workdir")
    parser.add_argument("--interval", type=int, default=600, help="seconds between continuous rounds")
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--teacher-batch", type=int, default=8)
    parser.add_argument("--eval-examples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-translation-drop", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.rounds < 1 or args.interval < 1 or args.examples < 1 or args.teacher_batch < 1 or args.eval_examples < 1 or args.steps < 1 or args.batch < 2:
        parser.error("rounds, interval, examples, teacher batch, evaluation examples, and steps must be positive; batch must be at least 2")
    if args.examples > 512 or args.eval_examples > 512 or args.teacher_batch > 8:
        parser.error("examples and evaluation examples are capped at 512; teacher batches are capped at 8")
    if args.max_translation_drop < 0:
        parser.error("max-translation-drop must be non-negative")
    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    lock_path = os.path.join(workdir, "teacher.lock")
    try:
        lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"distillation already running (or stale lock exists): {lock_path}")
    os.close(lock)

    try:
        teacher = OllamaTeacher(args.model, args.ollama_url)
        with tempfile.TemporaryDirectory(prefix="summit-release-") as replay_dir:
            counts = prepare_replay_data(args.data, replay_dir)
            if not any(counts[("parallel", lang)] for lang in ("krio", "mende", "temne")):
                raise RuntimeError("no provenance-cleared parallel replay data found")
        print(f"local teacher={args.model}; verified arithmetic only; output={workdir}", flush=True)
        round_number = 1
        stop_path = os.path.join(workdir, "STOP")
        while True:
            if args.continuous and os.path.exists(stop_path):
                print("STOP file found; ending distillation loop", flush=True)
                break
            try:
                _run_round(args, teacher, args.data, workdir, round_number)
            except CurriculumLimitReached as exc:
                print(f"stopping: {exc}", flush=True)
                break
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"round {round_number} failed; active checkpoint unchanged: {exc}", file=sys.stderr, flush=True)
                if not args.continuous:
                    raise
            if not args.continuous:
                if round_number >= args.rounds:
                    break
                round_number += 1
                continue
            round_number += 1
            for _ in range(args.interval):
                if os.path.exists(stop_path):
                    print("STOP file found; ending distillation loop", flush=True)
                    return
                time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted; active checkpoint remains at its last promoted version.", flush=True)
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()