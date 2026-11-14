"""Inspect (and optionally repair) a chat-format JSONL training file before it goes anywhere near a model.
   python tools/inspect_chat_jsonl.py data.jsonl [--repair-out cleaned.jsonl]
Reports: rows, task types, language pairs, reasoning-trace share, tool-call rows, text-encoding damage (mojibake) and whether it can be repaired,
and whether any row mentions the languages this project targets."""
import argparse, collections, json, re, sys

_HI = "".join(c for c in (bytes([b]).decode("cp1252", "ignore") for b in range(0x80, 0xC0)))
# classic mojibake = a UTF-8 lead byte shown as a letter followed by a continuation byte shown via cp1252 (so legitimate 'Ì', 'Ê', 'Ã' in Yoruba/Igbo text no longer match)
MOJI = re.compile("[\u00C2\u00C3\u00C4\u00C5\u00E1\u00E2\u00E0\u00EA\u00EB\u00C6][" + re.escape(_HI) + "]")
TARGETS = ("krio", "mende", "temne", "sierra leone")
_CP = {}
for b in range(256):
    try: _CP[bytes([b]).decode("cp1252")] = b
    except UnicodeDecodeError: _CP[chr(b)] = b                       # bytes cp1252 leaves undefined: keep as latin-1 control chars


def repair(s):
    """UTF-8 text that was decoded as cp1252/latin-1 and re-saved: undo it. Returns (text, changed_ok)."""
    if not MOJI.search(s):
        return s, False
    try:
        fixed = bytes(_CP[c] for c in s).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return s, False
    return (fixed, True) if len(MOJI.findall(fixed)) < len(MOJI.findall(s)) else (s, False)


def walk(obj, fn):
    if isinstance(obj, str): return fn(obj)
    if isinstance(obj, list): return [walk(x, fn) for x in obj]
    if isinstance(obj, dict): return {k: walk(v, fn) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--repair-out"); a = ap.parse_args()
    rows, bad = [], 0
    for line in open(a.path, encoding="utf-8"):
        if line.strip():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: bad += 1
    tasks, pairs, langs_in = collections.Counter(), collections.Counter(), collections.Counter()
    think = tools = moji_rows = fixable = 0; target_hits = []
    for r in rows:
        msgs = r.get("messages", []); sysm = next((m["content"] for m in msgs if m["role"] == "system"), "")
        m = re.search(r"Translate from (\w+) to (\w+)", sysm) or re.search(r"Correct the below translation to (\w+)", sysm)
        kind = ("correct-translation" if "Correct the below" in sysm else "dictionary" if "Dictionary" in sysm
                else "language-detect" if "detect" in sysm.lower() else "translate" if "Translate" in sysm else "tool-use/other")
        tasks[kind] += 1
        mm = re.search(r"from (\w+) to (\w+)", sysm)
        if mm: pairs[f"{mm.group(1)}>{mm.group(2)}"] += 1
        blob = json.dumps(r, ensure_ascii=False)
        think += "<think>" in blob; tools += "tools" in r or any("tool_calls" in x for x in msgs)
        if MOJI.search(blob):
            moji_rows += 1; fixable += all(repair(x)[1] or not MOJI.search(x) for x in [c["content"] for c in msgs if isinstance(c.get("content"), str)])
        if any(t in blob.lower() for t in TARGETS): target_hits.append(blob[:90])
    print(f"rows: {len(rows)} (unparseable lines: {bad})")
    print("task types:", dict(tasks.most_common()))
    print("most common language pairs:", dict(pairs.most_common(8)))
    print(f"rows with visible reasoning traces (<think>): {think} | rows with function-calling: {tools}")
    print(f"rows with text-encoding damage: {moji_rows} | of which fully repairable: {fixable}")
    print(f"rows mentioning {TARGETS}: {len(target_hits)}")
    if a.repair_out:
        n = 0
        with open(a.repair_out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(walk(r, lambda s: repair(s)[0]), ensure_ascii=False) + "\n"); n += 1
        print(f"wrote {n} repaired rows -> {a.repair_out}")


if __name__ == "__main__":
    main()
