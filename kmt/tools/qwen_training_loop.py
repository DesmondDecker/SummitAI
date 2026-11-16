"""Run bounded, resumable Qwen-teacher to Qwen-student SFT cycles."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.train_qwen_lora import load_approved_teacher_rows


MAX_TOTAL_EXAMPLES = 20000
MIN_FREE_VRAM_GB = 16


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def promotion_decision(before, candidate, chat_drop=0.03, translation_drop=0.02):
    """Require measurable math improvement and bounded retention regressions."""
    reasons = []
    before_math = before.get("arithmetic_reasoning", {})
    candidate_math = candidate.get("arithmetic_reasoning", {})
    math_metrics = ("exact_answer_accuracy", "valid_step_accuracy")
    if not all(key in before_math and key in candidate_math for key in math_metrics):
        return False, ["arithmetic benchmark metrics are missing"]
    if any(candidate_math[key] < before_math[key] for key in math_metrics):
        reasons.append("arithmetic accuracy regressed")
    if all(candidate_math[key] == before_math[key] for key in math_metrics):
        reasons.append("no arithmetic metric improved")
    before_regression = before_math.get("large_number_regression", {}).get("correct")
    candidate_regression = candidate_math.get("large_number_regression", {}).get("correct")
    if before_regression and not candidate_regression:
        reasons.append("large-number regression test failed")

    before_chat = before.get("heldout_conversation", {}).get("token_f1")
    candidate_chat = candidate.get("heldout_conversation", {}).get("token_f1")
    if before_chat is None or candidate_chat is None:
        reasons.append("conversation retention metrics are missing")
    elif candidate_chat < before_chat - chat_drop:
        reasons.append("held-out conversation score regressed beyond tolerance")

    before_translation = before.get("translation_chrF", {})
    candidate_translation = candidate.get("translation_chrF", {})
    if not before_translation or set(before_translation) != set(candidate_translation):
        reasons.append("translation retention metrics are missing or incomplete")
    else:
        for language, metrics in before_translation.items():
            old_score = metrics.get("chrF")
            new_score = candidate_translation[language].get("chrF")
            if old_score is None or new_score is None or new_score < old_score - translation_drop:
                reasons.append(f"{language} translation score regressed beyond tolerance")
    return not reasons, reasons


def _queue_size(path):
    return len(load_approved_teacher_rows(path, MAX_TOTAL_EXAMPLES + 1))


def _check_training_device():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "3B/4B student training requires a CUDA GPU. Use --generate-only on this PC "
            "or run the trainer on a CUDA machine; no CPU model load was attempted."
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    if free_gb < MIN_FREE_VRAM_GB:
        raise RuntimeError(
            f"at least {MIN_FREE_VRAM_GB} GiB free GPU memory is required for this configuration; "
            f"device has {free_gb:.1f} GiB free of {total_gb:.1f} GiB"
        )
    return {"device": torch.cuda.get_device_name(), "free_vram_gb": round(free_gb, 2),
            "total_vram_gb": round(total_gb, 2)}


def _run_logged(command, log_path):
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] $ {' '.join(map(str, command))}\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def _read_metrics(path):
    with open(path, encoding="utf-8") as source:
        report = json.load(source)
    return report["benchmarks"]


def _evaluate(args, adapter, destination, log_path):
    script = Path(__file__).with_name("evaluate_qwen.py")
    command = [
        sys.executable, str(script), "--model", args.base_model,
        "--data", args.data, "--sft-data", args.data_qwen,
        "--limit", str(args.eval_limit), "--math-limit", str(args.math_limit),
        "--device", "cuda", "--out", str(destination),
    ]
    if adapter:
        command.extend(["--adapter", str(adapter)])
    _run_logged(command, log_path)
    return _read_metrics(destination)


def _active_adapter(output_dir):
    pointer = Path(output_dir) / "active.json"
    if not pointer.is_file():
        return None, None
    with pointer.open(encoding="utf-8") as source:
        manifest = json.load(source)
    relative = manifest.get("adapter")
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"active adapter manifest has no adapter path: {pointer}")
    root = Path(output_dir).resolve()
    candidate = (root / relative).resolve()
    if os.path.commonpath((str(root), str(candidate))) != str(root):
        raise ValueError("active adapter path must remain inside the output directory")
    if not candidate.is_dir():
        raise FileNotFoundError(f"active adapter directory does not exist: {candidate}")
    return candidate, manifest


def _publish_adapter(output_dir, adapter_dir, args, metrics, cycle):
    root = Path(output_dir).resolve()
    relative = Path(adapter_dir).resolve().relative_to(root).as_posix()
    pointer = {
        "adapter": relative,
        "cycle": cycle,
        "base_model": args.base_model_id,
        "base_revision": args.base_revision,
        "base_license": args.base_license,
        "parameter_count": args.parameter_count,
        "metrics": metrics,
        "promoted_utc": _utc_now(),
    }
    active_path = root / "active.json"
    if active_path.is_file():
        previous = json.loads(active_path.read_text(encoding="utf-8"))
        _write_json(root / "previous.json", previous)
    _write_json(active_path, pointer)
    return pointer


def _wait(run_dir, seconds, status, cycle):
    stop_path = Path(run_dir) / "STOP"
    pause_path = Path(run_dir) / "PAUSE"
    for _ in range(seconds):
        if stop_path.exists():
            return False
        if pause_path.exists():
            status.update(state="paused", updated_utc=_utc_now())
            _write_json(Path(run_dir) / "status.json", status)
            while pause_path.exists():
                if stop_path.exists():
                    return False
                time.sleep(1)
            status.update(state="waiting", updated_utc=_utc_now())
            _write_json(Path(run_dir) / "status.json", status)
        time.sleep(1)
    return not stop_path.exists()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="local Qwen3-4B model directory")
    parser.add_argument("--base-model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--base-revision", default="1cfa9a7208912126459214e8b04321603b3df60c")
    parser.add_argument("--base-license", default="Apache-2.0")
    parser.add_argument("--parameter-count", type=int, default=4022468096)
    parser.add_argument("--data", default="data")
    parser.add_argument("--data-qwen", default="data_qwen")
    parser.add_argument("--output", default="outputs/summit-qwen3-4b")
    parser.add_argument("--run-dir", default="runs/qwen3-4b")
    parser.add_argument("--teacher-model", default="qwen2.5-coder:7b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--examples-per-cycle", type=int, default=64)
    parser.add_argument("--max-total-examples", type=int, default=MAX_TOTAL_EXAMPLES)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--generate-only", action="store_true",
                        help="generate verified teacher data without loading/training the student")
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--max-validation", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--math-limit", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.examples_per_cycle <= 256:
        parser.error("--examples-per-cycle must be between 1 and 256")
    if not args.examples_per_cycle <= args.max_total_examples <= MAX_TOTAL_EXAMPLES:
        parser.error(f"--max-total-examples must be between --examples-per-cycle and {MAX_TOTAL_EXAMPLES}")
    if not 1 <= args.max_cycles <= 1000 or args.interval < 60:
        parser.error("--max-cycles must be 1..1000 and --interval at least 60 seconds")
    if min(args.max_steps, args.max_examples, args.max_validation, args.max_length, args.eval_every,
           args.eval_limit, args.math_limit, args.batch_size, args.gradient_accumulation) < 1:
        parser.error("training and evaluation sizes must be positive")
    if args.parameter_count < 1:
        parser.error("--parameter-count must be positive")
    if args.continuous and args.max_cycles < 2:
        parser.error("--continuous requires --max-cycles of at least 2")
    return args


def main(argv=None):
    args = _parse_args(argv)
    run_dir, output_dir = Path(args.run_dir).resolve(), Path(args.output).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stop_path = run_dir / "STOP"
    if stop_path.exists():
        raise SystemExit(f"STOP marker exists; remove it only when intentionally starting a new run: {stop_path}")
    lock_path = run_dir / "worker.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit(f"worker already running or stale lock exists: {lock_path}") from exc
    os.close(lock_fd)

    status_path = run_dir / "status.json"
    log_path = run_dir / "worker.log"
    queue_path = Path(args.data_qwen).resolve() / "teacher_verified.jsonl"
    generation_script = Path(__file__).with_name("generate_verified_arithmetic.py")
    training_script = Path(__file__).with_name("train_qwen_lora.py")
    candidate_root = output_dir / "candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    status = {"state": "starting", "cycle": 0, "queue_path": str(queue_path),
              "base_model": args.base_model_id, "parameter_count": args.parameter_count,
              "generate_only": args.generate_only, "updated_utc": _utc_now()}

    try:
        if not (Path(args.data_qwen) / "manifest.json").is_file():
            raise FileNotFoundError(f"Qwen dataset manifest missing: {Path(args.data_qwen) / 'manifest.json'}")
        if not args.generate_only:
            if not (Path(args.base_model) / "config.json").is_file():
                raise FileNotFoundError(f"base model config is missing: {Path(args.base_model) / 'config.json'}")
            if not any((Path(args.base_model) / name).is_file()
                       for name in ("model.safetensors", "model.safetensors.index.json")):
                raise FileNotFoundError("base model weights are missing; refusing to train an incomplete checkpoint")
            status["accelerator"] = _check_training_device()

        teacher_count = _queue_size(queue_path)
        state_path = run_dir / "state.json"
        saved_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        first_cycle = int(saved_state.get("last_cycle", 0)) + 1
        for cycle in range(first_cycle, args.max_cycles + 1):
            if stop_path.exists():
                break
            while (run_dir / "PAUSE").exists():
                status.update(state="paused", cycle=cycle, updated_utc=_utc_now())
                _write_json(status_path, status)
                if stop_path.exists():
                    break
                time.sleep(1)
            if stop_path.exists():
                break
            if teacher_count + args.examples_per_cycle > args.max_total_examples:
                status.update(state="queue_cap_reached", cycle=cycle, updated_utc=_utc_now())
                _write_json(status_path, status)
                break

            cycle_dir = candidate_root / f"cycle-{cycle:04d}"
            if cycle_dir.exists() and not args.generate_only:
                raise FileExistsError(f"refusing to overwrite existing candidate directory: {cycle_dir}")
            if not args.generate_only:
                cycle_dir.mkdir(parents=True)
            status.update(state="generating", cycle=cycle, updated_utc=_utc_now(),
                          verified_examples=teacher_count)
            _write_json(status_path, status)
            _run_logged([
                sys.executable, str(generation_script), "--model", args.teacher_model,
                "--ollama-url", args.ollama_url, "--output", str(queue_path),
                "--examples", str(args.examples_per_cycle), "--seed", str(args.seed + cycle),
                "--max-queue-size", str(args.max_total_examples),
            ], log_path)
            new_teacher_count = _queue_size(queue_path)
            if new_teacher_count <= teacher_count:
                raise RuntimeError("teacher cycle produced no newly approved examples")
            teacher_count = new_teacher_count
            status.update(verified_examples=teacher_count, updated_utc=_utc_now())
            _write_json(status_path, status)

            if not args.generate_only:
                active_adapter, active_manifest = _active_adapter(output_dir)
                if active_manifest:
                    before_metrics = active_manifest["metrics"]
                else:
                    baseline_path = output_dir / "baseline.json"
                    if baseline_path.exists():
                        before_metrics = _read_metrics(baseline_path)
                    else:
                        status.update(state="evaluating_base", updated_utc=_utc_now())
                        _write_json(status_path, status)
                        before_metrics = _evaluate(args, None, baseline_path, log_path)

                status.update(state="training", updated_utc=_utc_now(),
                              parent_adapter=str(active_adapter) if active_adapter else None)
                _write_json(status_path, status)
                train_command = [
                    sys.executable, str(training_script),
                    "--base-model", args.base_model,
                    "--base-model-id", args.base_model_id,
                    "--base-revision", args.base_revision,
                    "--base-license", args.base_license,
                    "--data", args.data_qwen,
                    "--output", str(cycle_dir),
                    "--max-steps", str(args.max_steps),
                    "--max-examples", str(args.max_examples),
                    "--max-validation", str(args.max_validation),
                    "--max-length", str(args.max_length),
                    "--eval-every", str(args.eval_every),
                    "--learning-rate", str(args.learning_rate),
                    "--batch-size", str(args.batch_size),
                    "--gradient-accumulation", str(args.gradient_accumulation),
                    "--approved-teacher-data", str(queue_path),
                    "--max-teacher-examples", str(min(teacher_count, args.max_total_examples)),
                    "--device", "cuda",
                ]
                if active_adapter:
                    train_command.extend(["--resume-adapter", str(active_adapter)])
                _run_logged(train_command, log_path)

                candidate_adapter = cycle_dir / "adapter"
                if not candidate_adapter.is_dir():
                    raise FileNotFoundError(f"training completed without an adapter: {candidate_adapter}")
                status.update(state="evaluating_candidate", updated_utc=_utc_now())
                _write_json(status_path, status)
                candidate_metrics = _evaluate(
                    args, candidate_adapter, cycle_dir / "evaluation.json", log_path
                )
                promoted, reasons = promotion_decision(before_metrics, candidate_metrics)
                report = {
                    "cycle": cycle,
                    "base_model": args.base_model_id,
                    "base_revision": args.base_revision,
                    "parameter_count": args.parameter_count,
                    "verified_teacher_examples": teacher_count,
                    "before": before_metrics,
                    "candidate": candidate_metrics,
                    "promoted": promoted,
                    "reasons": reasons,
                    "evaluated_utc": _utc_now(),
                }
                _write_json(cycle_dir / "cycle_report.json", report)
                if promoted:
                    _publish_adapter(output_dir, candidate_adapter, args, candidate_metrics, cycle)
                status.update(state="promoted" if promoted else "candidate_rejected",
                              last_result=report, updated_utc=_utc_now())
                _write_json(status_path, status)
            else:
                status.update(state="data_generated", updated_utc=_utc_now())
                _write_json(status_path, status)

            _write_json(state_path, {"last_cycle": cycle, "verified_examples": teacher_count,
                                     "updated_utc": _utc_now()})
            if not args.continuous or cycle >= args.max_cycles:
                break
            now = datetime.now(timezone.utc)
            status.update(state="waiting",
                          next_cycle_utc=(now + timedelta(seconds=args.interval)).isoformat(),
                          updated_utc=now.isoformat())
            _write_json(status_path, status)
            if not _wait(run_dir, args.interval, status, cycle):
                break

        final_state = "stopped" if stop_path.exists() else status.get("state", "complete")
        status.update(state=final_state, updated_utc=_utc_now())
        _write_json(status_path, status)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}", updated_utc=_utc_now())
        _write_json(status_path, status)
        raise
    except KeyboardInterrupt:
        status.update(state="stopped", updated_utc=_utc_now())
        _write_json(status_path, status)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
