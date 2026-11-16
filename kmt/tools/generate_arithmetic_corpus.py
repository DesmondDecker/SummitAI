"""Generate deterministic, exact-verified arithmetic JSONL datasets."""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmt.distillation import evaluate_expression, make_expressions


OPERATORS = ("+", "-", "*")
OPERANDS = tuple(range(2, 101))
EXPRESSIONS_PER_PATTERN = len(OPERANDS) ** 3
ROWS_PER_DATASET = 250_000
DATASET_COUNT = 18
TOTAL_ROWS = ROWS_PER_DATASET * DATASET_COUNT
EVAL_SEED = 739391
EVAL_COUNT = 48
PROMPTS = (
    "Evaluate exactly: {expression}\nShow the arithmetic steps and final answer.",
    "Work out {expression}. Write both intermediate arithmetic steps and the final answer.",
    "Calculate {expression}, showing how you get each intermediate result.",
    "Solve {expression} exactly. Include the two calculation steps and the answer.",
)


def load_excluded_expressions(training_path, teacher_path):
    excluded = set()
    for path in (Path(training_path), Path(teacher_path)):
        if not path.is_file():
            raise FileNotFoundError(f"required existing training data is missing: {path}")
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                verification = row.get("verification")
                expression = verification.get("expression") if isinstance(verification, dict) else None
                if isinstance(expression, str):
                    excluded.add(expression)
    excluded.update(make_expressions(EVAL_COUNT, EVAL_SEED))
    return excluded


def _permutation(rng):
    offset = rng.randrange(EXPRESSIONS_PER_PATTERN)
    stride = rng.randrange(1, EXPRESSIONS_PER_PATTERN)
    while math.gcd(stride, EXPRESSIONS_PER_PATTERN) != 1:
        stride = rng.randrange(1, EXPRESSIONS_PER_PATTERN)
    return offset, stride


def _decode_expression(index, inner_operator, outer_operator):
    a_index, remainder = divmod(index, len(OPERANDS) ** 2)
    b_index, c_index = divmod(remainder, len(OPERANDS))
    a, b, c = OPERANDS[a_index], OPERANDS[b_index], OPERANDS[c_index]
    return f"({a} {inner_operator} {b}) {outer_operator} {c}"


def _answer_record(expression, dataset_id, sequence_number, prompt_seed):
    left_open = expression.index("(") + 1
    left_close = expression.index(")")
    left = expression[left_open:left_close].strip().split()
    outer = expression[left_close + 1:].strip().split()
    a, inner_operator, b = int(left[0]), left[1], int(left[2])
    outer_operator, c = outer[0], int(outer[1])
    intermediate = evaluate_expression(f"{a} {inner_operator} {b}")
    answer = evaluate_expression(expression)
    if evaluate_expression(f"{intermediate} {outer_operator} {c}") != answer:
        raise ArithmeticError(f"step trace disagrees with exact expression evaluation: {expression}")
    steps = (
        f"Step 1: {a} {inner_operator} {b} = {intermediate}\n"
        f"Step 2: {intermediate} {outer_operator} {c} = {answer}\n"
        f"Answer: {answer}"
    )
    prompt_index = (sequence_number + prompt_seed) % len(PROMPTS)
    digest = hashlib.sha256(expression.encode("utf-8")).hexdigest()
    return {
        "group_id": f"verified-arithmetic:{digest}",
        "source": "synthetic arithmetic corpus",
        "dataset_id": dataset_id,
        "teacher_model": "local deterministic generator (no LLM)",
        "language": "en",
        "difficulty": "two-operation integer arithmetic; operands 2..100",
        "review_status": "approved",
        "verification": {
            "method": "exact local integer arithmetic and canonical two-step trace",
            "expression": expression,
            "answer": answer,
        },
        "messages": [
            {"role": "user", "content": PROMPTS[prompt_index].format(expression=expression)},
            {"role": "assistant", "content": steps},
        ],
    }


def generate(output_dir, training_path, teacher_path, rows_per_dataset, seed):
    if rows_per_dataset < 1:
        raise ValueError("rows per dataset must be positive")
    if rows_per_dataset * 2 > EXPRESSIONS_PER_PATTERN:
        raise ValueError(
            "two datasets per operator pattern cannot exceed the unique expression space"
        )
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing corpus directory: {output}")

    excluded = load_excluded_expressions(training_path, teacher_path)
    output.mkdir(parents=True)
    rng = random.Random(seed)
    dataset_manifests = []
    dataset_index = 0

    for inner_operator in OPERATORS:
        for outer_operator in OPERATORS:
            offset, stride = _permutation(rng)
            cursor = 0
            for tranche in (1, 2):
                dataset_index += 1
                dataset_id = (
                    f"arith_{inner_operator.replace('*', 'mul').replace('+', 'add').replace('-', 'sub')}_"
                    f"{outer_operator.replace('*', 'mul').replace('+', 'add').replace('-', 'sub')}_"
                    f"{tranche:02d}"
                )
                path = output / f"{dataset_id}.jsonl"
                temporary = path.with_suffix(".jsonl.tmp")
                accepted = 0
                digest = hashlib.sha256()
                with temporary.open("wb") as target:
                    while accepted < rows_per_dataset:
                        if cursor >= EXPRESSIONS_PER_PATTERN:
                            raise RuntimeError(
                                f"unique expression space exhausted for {inner_operator}, {outer_operator}"
                            )
                        index = (offset + cursor * stride) % EXPRESSIONS_PER_PATTERN
                        cursor += 1
                        expression = _decode_expression(index, inner_operator, outer_operator)
                        if expression in excluded:
                            continue
                        sequence_number = (tranche - 1) * rows_per_dataset + accepted
                        row = _answer_record(expression, dataset_id, sequence_number, seed + dataset_index)
                        encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                        target.write(encoded)
                        digest.update(encoded)
                        accepted += 1
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, path)
                dataset_manifests.append({
                    "dataset_id": dataset_id,
                    "file": path.name,
                    "rows": accepted,
                    "inner_operator": inner_operator,
                    "outer_operator": outer_operator,
                    "sha256": digest.hexdigest(),
                    "bytes": path.stat().st_size,
                })
                print(f"completed {dataset_id}: {accepted:,} rows", flush=True)

    manifest = {
        "format": "UTF-8 JSONL, one supervised chat record per line",
        "generation": "deterministic local arithmetic; no model-generated answers",
        "seed": seed,
        "rows_per_dataset": rows_per_dataset,
        "dataset_count": DATASET_COUNT,
        "total_rows": sum(item["rows"] for item in dataset_manifests),
        "operand_values": [min(OPERANDS), max(OPERANDS)],
        "expression_shape": "(a op b) op c",
        "operators": list(OPERATORS),
        "answer_verification": (
            "Each answer is evaluated by the repository's exact integer AST evaluator; "
            "the displayed intermediate result is independently evaluated and checked."
        ),
        "excluded_existing_and_evaluation_expressions": len(excluded),
        "heldout_arithmetic_eval": {"seed": EVAL_SEED, "count": EVAL_COUNT},
        "human_reviewed": False,
        "limitations": [
            "All 4.5 million records are narrow two-operation arithmetic, not general reasoning.",
            "Large repeated synthetic proportion can harm conversational, coding, translation, and factual capability.",
            "Use only as a separate arithmetic experiment; do not merge blindly into the existing general SFT training set.",
        ],
        "datasets": dataset_manifests,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = (
        "ARITHMETIC CORPUS - 18 DATASETS\n\n"
        f"Total: {manifest['total_rows']:,} records in {DATASET_COUNT} separate JSONL datasets "
        f"({rows_per_dataset:,} rows each).\n"
        "All records are exact-verified two-operation integer arithmetic with operands 2..100 "
        "and operators +, -, *. Each JSONL row contains a user prompt, a locally constructed "
        "two-step answer, a unique group ID, and verification metadata. No answer was generated "
        "by an LLM. See manifest.json for per-file counts and SHA-256 digests.\n\n"
        "IMPORTANT: These datasets are arithmetic-only and not human-reviewed. They are not "
        "18 different subject areas and do not provide general reasoning, coding, science, history, "
        "writing, customer support, or Krio/Mende/Temne coverage. The repetition may damage broader "
        "capability if mixed into general SFT. Keep separate and evaluate before use. This only "
        "creates data; it does not train Summit or alter any model weights.\n"
    )
    (output / "README.txt").write_text(readme, encoding="utf-8")

    archive_path = output.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(manifest_path, arcname=manifest_path.name)
        archive.write(output / "README.txt", arcname="README.txt")
        for item in dataset_manifests:
            archive.write(output / item["file"], arcname=item["file"])
    print(json.dumps({
        "output_dir": str(output),
        "archive": str(archive_path),
        "datasets": len(dataset_manifests),
        "total_rows": manifest["total_rows"],
        "archive_bytes": archive_path.stat().st_size,
    }, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data_qwen/arithmetic_18x250k")
    parser.add_argument("--training-data", default="data_qwen/train.jsonl")
    parser.add_argument("--teacher-data", default="data_qwen/teacher_verified.jsonl")
    parser.add_argument("--rows-per-dataset", type=int, default=ROWS_PER_DATASET)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args(argv)
    if args.rows_per_dataset != ROWS_PER_DATASET:
        parser.error(f"--rows-per-dataset must be {ROWS_PER_DATASET} for the requested 18 x 250,000 corpus")
    try:
        generate(args.output_dir, args.training_data, args.teacher_data, args.rows_per_dataset, args.seed)
    except (OSError, ValueError, RuntimeError, ArithmeticError) as exc:
        raise SystemExit(f"corpus generation failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
