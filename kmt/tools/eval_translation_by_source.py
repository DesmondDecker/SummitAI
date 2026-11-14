"""Held-out translation and language-ID quality, split by data source (in-domain vs out-of-domain).
   python tools/eval_translation_by_source.py --ckpt checkpoints --data data --lang krio --n 60
chrF is computed on spelling-folded text (so e-open/o-open letters vs plain e/o do not count as errors)."""
import argparse, collections, os, random, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kmt.assistant import Assistant
from kmt.evaluate import chrf
from kmt.tokenizer import fold_text


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="checkpoints"); ap.add_argument("--data", default="data")
    ap.add_argument("--lang", default="krio"); ap.add_argument("--n", type=int, default=60); ap.add_argument("--memdir", default="/tmp/_evalmem")
    ap.add_argument("--show", type=int, default=3); a = ap.parse_args()
    os.system(f"rm -rf {a.memdir}")
    bot = Assistant(a.ckpt, a.memdir, a.data); rng = random.Random(0)
    by = collections.defaultdict(list)
    for l in open(os.path.join(a.data, "heldout", f"{a.lang}-en.tsv"), encoding="utf-8"):
        if l.startswith("#") or not l.strip(): continue
        p = l.rstrip("\n").split("\t")
        if len(p) >= 3: by[p[2].split(":")[0]].append((p[0], p[1]))
    print(f"{'source':<24}{'dir':<9}{'n':>4}  {'mean chrF':>9}  {'chrF>=0.5':>9}   {'LID acc':>8}")
    shown = []
    for src, rows in sorted(by.items(), key=lambda kv: -len(kv[1])):
        rows = rows if len(rows) <= a.n else rng.sample(rows, a.n)
        for sl, tl, pairs in ((a.lang, "en", rows), ("en", a.lang, [(e, l) for l, e in rows])):
            sc, ok = [], 0
            for s, ref in pairs:
                r = bot.translate(s, sl, tl, use_memory=False, force=True)
                sc.append(chrf(fold_text(r["text"]).lower(), fold_text(ref).lower()))
                ok += bot.identify(s)[0] == sl
                if len(shown) < 40: shown.append((src, f"{sl}>{tl}", s, ref, r["text"]))
            print(f"{src:<24}{sl + '>' + tl:<9}{len(pairs):>4}  {np.mean(sc):>9.3f}  {np.mean(np.array(sc) >= 0.5):>9.0%}   {ok / len(pairs):>8.0%}")
    print("\nexamples (held out, never seen in training):")
    seen = collections.Counter()
    for src, d, s, ref, out in shown:
        if seen[(src, d)] >= a.show // 2 + 1 or d.endswith("en") is False and seen[(src, d)] >= 1: continue
        seen[(src, d)] += 1
        print(f"  [{src} {d}]\n    in : {s[:100]}\n    ref: {ref[:100]}\n    out: {out[:100]}")


if __name__ == "__main__":
    main()
