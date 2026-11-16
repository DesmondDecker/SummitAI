"""Prepare Apache-2.0 English instruction/chat data for Summit's Qwen SFT path."""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone


DATA_SOURCES = {
    "OpenAssistant/oasst2": {
        "license": "Apache-2.0",
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst2",
        "use": "English conversation trees; accepted, non-deleted messages only",
    },
    "CohereLabs/aya_dataset": {
        "license": "Apache-2.0",
        "url": "https://huggingface.co/datasets/CohereLabs/aya_dataset",
        "use": "English instruction examples only",
    },
    "Summit local manuals": {
        "license": "US government-produced / public-domain sources; confirm jurisdiction and third-party material",
        "url": "data/SOURCES.md",
        "use": "quality >= 0.5 pairs from Peace Corps manuals, Migeod 1908, and MBSSE syllabus; excludes restricted sources",
    },
}
SAFE_LOCAL_SOURCES = ("peacecorps_", "migeod1908:", "mbsse_syllabus2020")


def _bucket(group_id):
    value = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if value < 90 else "validation" if value < 95 else "test"


def _valid_message(row):
    if row.get("deleted") is True or row.get("review_result") is False:
        return False
    if row.get("tree_state") is not None and row.get("tree_state") != "ready_for_export":
        return False
    labels = row.get("labels") or {}
    for label in ("spam", "not_appropriate", "lang_mismatch", "pii"):
        value = labels.get(label)
        if isinstance(value, dict) and value.get("value", 0) > 0:
            return False
    if str(row.get("lang", "")).lower() not in {"en", "eng"}:
        return False
    if row.get("role") not in {"prompter", "assistant"}:
        return False
    return isinstance(row.get("text"), str) and bool(row["text"].strip())


def conversation_examples(rows):
    """Turn flat OASST2 message rows into prefix/answer examples grouped by tree."""
    by_id, groups = {}, defaultdict(list)
    heldout_trees = {
        row.get("message_tree_id") for row in rows
        if row.get("_hf_split") in {"validation", "test"} and row.get("message_tree_id")
    }
    for row in rows:
        if not _valid_message(row):
            continue
        message_id = row.get("message_id")
        tree_id = row.get("message_tree_id")
        if not message_id or not tree_id:
            continue
        by_id[message_id] = row
        groups[tree_id].append(row)

    examples = []
    for tree_id, members in groups.items():
        member_ids = {row["message_id"] for row in members}
        children = defaultdict(list)
        for row in members:
            parent_id = row.get("parent_id")
            if parent_id in member_ids:
                children[parent_id].append(row["message_id"])
        for row in members:
            if row["role"] != "assistant" or children.get(row["message_id"]):
                continue
            path, current, seen = [], row, set()
            while current:
                mid = current["message_id"]
                if mid in seen:
                    path = []
                    break
                seen.add(mid)
                path.append(current)
                parent = current.get("parent_id")
                current = by_id.get(parent) if parent else None
            path.reverse()
            turns = []
            for item in path:
                role = "user" if item["role"] == "prompter" else "assistant"
                if turns and turns[-1]["role"] == role:
                    break
                turns.append({"role": role, "content": item["text"].strip()})
            if len(turns) >= 2 and turns[0]["role"] == "user" and turns[-1]["role"] == "assistant":
                for index in range(1, len(turns), 2):
                    if turns[index]["role"] == "assistant":
                        example = {
                            "group_id": f"oasst2:{tree_id}",
                            "source": "OpenAssistant/oasst2",
                            "language": "en",
                            "messages": turns[:index + 1],
                        }
                        if tree_id in heldout_trees:
                            example["forced_split"] = "test"
                        examples.append(example)
    return examples


def aya_examples(rows):
    examples = []
    for row in rows:
        language = str(row.get("language", "")).strip().lower()
        code = str(row.get("language_code", "")).strip().lower()
        if language != "english" and code not in {"en", "eng"}:
            continue
        user, assistant = row.get("inputs"), row.get("targets")
        if not isinstance(user, str) or not user.strip() or not isinstance(assistant, str) or not assistant.strip():
            continue
        signature = hashlib.sha256((user.strip() + "\0" + assistant.strip()).encode("utf-8")).hexdigest()
        examples.append({
            "group_id": f"aya:{signature}",
            "source": "CohereLabs/aya_dataset",
            "language": "en",
            "messages": [
                {"role": "user", "content": user.strip()},
                {"role": "assistant", "content": assistant.strip()},
            ],
        })
    return examples


def split_examples(examples):
    """Deduplicate and split by group ID, never by individual message."""
    selected, group_splits = {}, {}
    splits = {name: [] for name in ("train", "validation", "test")}
    priority = {"train": 0, "validation": 1, "test": 2}
    for example in examples:
        key = json.dumps(example["messages"], ensure_ascii=False, sort_keys=True)
        split = example.get("forced_split") or _bucket(example["group_id"])
        previous = selected.get(key)
        if previous is None or priority[split] > priority[previous[0]]:
            selected[key] = (split, example)
    for split, example in selected.values():
        group_id = example["group_id"]
        previous = group_splits.setdefault(group_id, split)
        if previous != split:
            raise ValueError(f"group {group_id} appears in multiple requested splits")
        splits[split].append({key: value for key, value in example.items() if key != "forced_split"})
    return splits


def local_translation_examples(data_dir):
    """Add quality-screened, provenance-allowlisted bilingual pairs."""
    examples, counts = [], {"train": 0, "test": 0}
    csv.field_size_limit(sys.maxsize)
    for source_split, target_split in (("parallel", "train"), ("heldout", "test")):
        for language in ("krio", "mende", "temne"):
            path = os.path.join(data_dir, source_split, f"{language}-en.tsv")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8-sig", newline="") as source:
                rows = csv.reader((line for line in source if not line.startswith("#")), delimiter="\t")
                for row in rows:
                    if len(row) < 3 or not any(row[2].startswith(prefix) for prefix in SAFE_LOCAL_SOURCES):
                        continue
                    try:
                        quality = float(row[3]) if len(row) > 3 else 1.0
                    except ValueError:
                        continue
                    local, english = row[0].strip(), row[1].strip()
                    if quality < 0.5 or not local or not english:
                        continue
                    digest = hashlib.sha256(f"{language}\0{local}\0{english}".encode("utf-8")).hexdigest()
                    for direction, text, target in (
                        (f"English to {language.title()}", english, local),
                        (f"{language.title()} to English", local, english),
                    ):
                        examples.append({
                            "group_id": f"local-pair:{digest}",
                            "source": "Summit local manuals",
                            "language": "en" if direction.startswith("English") else language,
                            "forced_split": target_split,
                            "messages": [
                                {"role": "user", "content": f"Translate from {direction}:\n{text}"},
                                {"role": "assistant", "content": target},
                            ],
                        })
                        counts[target_split] += 1
    return examples, counts


def _resolve_dataset_revision(repo_id, revision):
    from huggingface_hub import HfApi
    return HfApi().dataset_info(repo_id, revision=revision).sha


def _local_data_hashes(data_dir):
    hashes = {}
    for split in ("parallel", "heldout"):
        for language in ("krio", "mende", "temne"):
            path = os.path.join(data_dir, split, f"{language}-en.tsv")
            if not os.path.isfile(path):
                continue
            digest = hashlib.sha256()
            with open(path, "rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[os.path.relpath(path, data_dir).replace(os.sep, "/")] = digest.hexdigest()
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data_qwen")
    parser.add_argument("--oasst-revision", default="main")
    parser.add_argument("--aya-revision", default="main")
    parser.add_argument("--local-data", default="data")
    parser.add_argument("--skip-aya", action="store_true")
    args = parser.parse_args(argv)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install requirements-qwen.txt in the isolated Qwen environment first.") from exc

    oasst_revision = _resolve_dataset_revision("OpenAssistant/oasst2", args.oasst_revision)
    oasst = load_dataset("OpenAssistant/oasst2", revision=oasst_revision)
    oasst_rows = [dict(row, _hf_split=split) for split in oasst for row in oasst[split]]
    examples = conversation_examples(oasst_rows)
    aya_count = 0
    aya_revision = None
    if not args.skip_aya:
        aya_revision = _resolve_dataset_revision("CohereLabs/aya_dataset", args.aya_revision)
        aya = load_dataset("CohereLabs/aya_dataset", revision=aya_revision, split="train")
        aya_rows = aya_examples(dict(row) for row in aya)
        aya_count = len(aya_rows)
        examples.extend(aya_rows)
    local_rows, local_counts = local_translation_examples(args.local_data)
    examples.extend(local_rows)
    splits = split_examples(examples)

    os.makedirs(args.out, exist_ok=True)
    for name, rows in splits.items():
        with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "OpenAssistant/oasst2": {**DATA_SOURCES["OpenAssistant/oasst2"], "revision": oasst_revision},
        },
        "filtering": ["English only", "exclude deleted or explicitly rejected messages", "conversation-tree grouped 90/5/5 split", "deduplicate exact message paths"],
        "aya_included": not args.skip_aya,
        "aya_rows_before_dedup": aya_count,
        "local_translation_counts": local_counts,
        "local_data_sha256": _local_data_hashes(args.local_data),
        "counts": {name: len(rows) for name, rows in splits.items()},
    }
    if not args.skip_aya:
        manifest["sources"]["CohereLabs/aya_dataset"] = {**DATA_SOURCES["CohereLabs/aya_dataset"], "revision": aya_revision}
    if local_rows:
        manifest["sources"]["Summit local manuals"] = DATA_SOURCES["Summit local manuals"]
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as out:
        json.dump(manifest, out, ensure_ascii=False, indent=2)
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()