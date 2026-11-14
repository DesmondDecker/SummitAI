"""Measure the translation-memory knowledge base at the level users experience it.
   python tools/eval_kb.py --data data
Policies compared on the same queries:
  old          top-1 character-n-gram match accepted at >= 0.92 similarity (the earlier behaviour)
  lookup       canonical keys + quality ranking + edit-distance check; short fuzzy matches become 'did you mean' suggestions
  lookup+auto  as above but also auto-answers an unambiguous 1-2 edit match for short queries
Query sets: typo'd copies of stored entries (should be answered correctly or suggested) and HELD-OUT text the memory has never
seen (must NOT be answered with a confident wrong translation)."""
import argparse, os, random, sys, time, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kmt.memory import MemoryStore, _norm
from kmt.data import read_tsv

LANGS = ("krio", "mende", "temne")


def load_rows(data):
    rows = []
    for lang in LANGS:
        p = os.path.join(data, "parallel", f"{lang}-en.tsv")
        for line in open(p, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            rows.append((lang, f[0], f[1], float(f[3]) if len(f) > 3 else 1.0))
    return rows


def build(rows, qmin):
    m = MemoryStore(None)
    for lang, l, e, q in rows:
        if q >= qmin:
            m.add_pair(lang, l, "en", e, q=q)
    return m


def perturb(s, rng):
    s = list(s)
    if len(s) < 5:
        return "".join(s).upper()
    for _ in range(max(1, len(s) // 12)):
        i = rng.randrange(len(s)); op = rng.choice("dsc")
        if op == "d": s.pop(i)
        elif op == "s": s[i] = rng.choice("aeiounrtls")
        else: s[i] = s[i].swapcase()
        if len(s) < 3: break
    out = "".join(s)
    return out if rng.random() < 0.5 else out.rstrip(".!?") + rng.choice(["", " ", "."])


def answer(m, policy, q, sl, tl):
    """-> (answers_set_or_None, suggestions_set)"""
    if policy == "old":
        h = m.search(q, 1, kind="translation", lang=sl, tgt_lang=tl)
        return ({_norm(h[0][1]["tgt"])}, set()) if h and h[0][0] >= 0.92 else (None, set())
    lk = m.lookup(q, sl, tl, short_auto=(policy == "lookup+auto"))
    if lk["status"] in ("exact", "near"):
        return ({_norm(lk["answer"])} | {_norm(a) for a in lk["alternatives"]}, set())
    return None, {_norm(s["translation"]) for s in lk["suggestions"]}


def evaluate(m, rows, data, policy, seed=1, n=1500):
    rng = random.Random(seed); gold = collections.defaultdict(set)
    for lang, l, e, q in rows:
        if q >= 0.5:
            gold[(lang, "en", _norm(l))].add(_norm(e)); gold[("en", lang, _norm(e))].add(_norm(l))
    keys = sorted(gold); rng.shuffle(keys); keys = keys[:n]
    out = {}
    by = collections.defaultdict(lambda: [0, 0, 0, 0])                # answered, wrong, suggest-hit, total
    for sl, tl, src in keys:
        q = perturb(src, rng); ans, sug = answer(m, policy, q, sl, tl)
        b = by["1 word" if len(src.split()) == 1 else ("2 words" if len(src.split()) == 2 else "3+ words")]
        b[3] += 1
        if ans is not None:
            b[0] += 1; b[1] += not (ans & gold[(sl, tl, src)])
        else:
            b[2] += bool(sug & gold[(sl, tl, src)])
    out["noisy"] = by
    held = {"sentences": [], "words": []}
    for lang in LANGS:
        for r in read_tsv(os.path.join(data, "heldout", f"{lang}-en.tsv"), 2):
            held["sentences" if len(r[0].split()) >= 3 else "words"] += [(lang, "en", r[0], r[1]), ("en", lang, r[1], r[0])]
    for kind, items in held.items():
        acc = wrong = 0
        for sl, tl, q, ref in items:
            ans, _ = answer(m, policy, q, sl, tl)
            if ans is not None:
                acc += 1; wrong += _norm(ref) not in ans
        out["unseen " + kind] = (acc, wrong, len(items))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); a = ap.parse_args()
    rows = load_rows(a.data); allm, clean = build(rows, 0.0), build(rows, 0.5)
    print(f"stored entries: all {len(allm):,} | quality >= 0.5: {len(clean):,} (removed {len(allm) - len(clean):,}, listed in data/KB_REVIEW.tsv)")
    for policy in ("old", "lookup", "lookup+auto"):
        store = allm if policy == "old" else clean
        r = evaluate(store, rows, a.data, policy)
        print(f"\n== policy: {policy} ({'all entries' if policy == 'old' else 'quality-filtered'})")
        for k, (ans, wr, sh, tot) in sorted(r["noisy"].items()):
            print(f"  typo'd {k:<8} n={tot:<4} answered {ans/tot:>4.0%} (wrong {wr/max(1,ans):>3.0%} of answered) | not answered but right answer offered as suggestion: {sh/max(1, tot-ans):>4.0%}")
        for k in ("unseen sentences", "unseen words"):
            acc, wr, n = r[k]; print(f"  {k:<17} n={n:<4} answered {acc}  WRONG answers {wr}")


if __name__ == "__main__":
    main()
