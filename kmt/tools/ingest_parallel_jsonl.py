"""Clean a large English-<local language> parallel file in JSONL ({"eng": ..., "kri": ...}) and score its alignment.
   python tools/ingest_parallel_jsonl.py FILE.jsonl --src-key kri --out data/raw_extracts/krio_jw.tsv [--drop-worst 0.08]
Steps: cheap filters (length ratio, mismatched numbers, citation fragments, duplicates, over-long) -> IBM Model 1 alignment score
(learned from the data itself, no outside resources) -> drop the worst-scoring share. Output TSV: local, english, align_score."""
import argparse, collections, json, math, os, re, sys, time


def toks(s):
    return re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", s.lower())


def ibm1(pairs, iters=4):
    """t[e][f] ~ p(f | e) for token lists (e = English, f = local), with a NULL source word."""
    cnt = collections.Counter(w for e, f in pairs for w in e + f)
    keep = lambda w: w if cnt[w] >= 2 else "<unk>"
    P = [(["<null>"] + [keep(w) for w in e], [keep(w) for w in f]) for e, f in pairs]
    fv = {w for _, f in P for w in f}
    t = collections.defaultdict(lambda: collections.defaultdict(lambda: 1.0 / len(fv)))
    for it in range(iters):
        t0 = time.time(); count = collections.defaultdict(lambda: collections.defaultdict(float)); tot = collections.defaultdict(float)
        for e, f in P:
            for fw in f:
                z = sum(t[ew][fw] for ew in e)
                for ew in e:
                    c = t[ew][fw] / z; count[ew][fw] += c; tot[ew] += c
        t = collections.defaultdict(lambda: collections.defaultdict(lambda: 1e-9), {ew: collections.defaultdict(lambda: 1e-9, {fw: c / tot[ew] for fw, c in d.items()}) for ew, d in count.items()})
        print(f"  IBM1 iteration {it + 1}/{iters} ({time.time() - t0:.0f}s)", flush=True)
    return t, P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--src-key", default="kri"); ap.add_argument("--en-key", default="eng")
    ap.add_argument("--out", required=True); ap.add_argument("--drop-worst", type=float, default=0.08); ap.add_argument("--max-words", type=int, default=40)
    ap.add_argument("--max-formula", type=int, default=400, help="cap on near-identical '(Read ...)' scripture-reference lines")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path, encoding="utf-8") if l.strip()]
    n0 = len(rows); drops = collections.Counter(); seen = set(); keep = []; formula = 0
    for r in rows:
        e, k = " ".join(r[a.en_key].split()), " ".join(r[a.src_key].split())
        ew, kw = e.split(), k.split()
        if not e or not k or e.lower() == k.lower(): drops["empty/identical"] += 1; continue
        if not (2 <= len(ew) <= a.max_words and 2 <= len(kw) <= int(a.max_words * 1.4)): drops["length"] += 1; continue
        ratio = len(kw) / len(ew)
        if not 0.5 <= ratio <= 2.4: drops["length ratio"] += 1; continue
        de, dk = re.findall(r"\d+", e), re.findall(r"\d+", k)
        if de and set(de) != set(dk): drops["numbers differ"] += 1; continue
        if re.match(r"^\s*\d+:\d+", k) or re.match(r"^\s*\d+:\d+", e): drops["citation fragment"] += 1; continue
        if re.match(r"^\W*\(?Read\b", e):
            formula += 1
            if formula > a.max_formula: drops["formulaic (Read ...)"] += 1; continue
        key = re.sub(r"\W+", " ", e.lower()).strip()
        if key in seen: drops["duplicate English"] += 1; continue
        seen.add(key); keep.append((e, k))
    print(f"read {n0:,} | after cheap filters {len(keep):,} | dropped: {dict(drops)}")
    t, P = ibm1([(toks(e), toks(k)) for e, k in keep])
    scores = []
    for (e, f) in P:
        s = sum(math.log(sum(t[ew][fw] for ew in e) / len(e) + 1e-12) for fw in f) / max(1, len(f))
        scores.append(s)
    order = sorted(range(len(keep)), key=lambda i: scores[i]); cut = int(len(keep) * a.drop_worst)
    bad = set(order[:cut])
    print(f"\nlowest-scoring pairs (what the alignment model distrusts most):")
    for i in order[:6]: print(f"  {scores[i]:.2f} | EN: {keep[i][0][:70]}\n        | KR: {keep[i][1][:70]}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# local\tenglish\talign_score   (human-translated text from jw.org-style publications; alignment-filtered; RIGHTS UNCONFIRMED, see data/SOURCES.md)\n")
        for i, (e, k) in enumerate(keep):
            if i not in bad: f.write(f"{k}\t{e}\t{scores[i]:.2f}\n")
    print(f"\nkept {len(keep) - cut:,} pairs (dropped worst {a.drop_worst:.0%} by alignment) -> {a.out}")


if __name__ == "__main__":
    main()
