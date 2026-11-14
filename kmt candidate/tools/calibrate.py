"""Measure when the model's translations can be trusted and write checkpoints/calibration.json.
   python tools/calibrate.py --ckpt checkpoints --data data --target-chrf 0.5
For each direction (e.g. mende>en) it finds the lowest model-confidence threshold at which held-out translations average at least
--target-chrf. If no threshold reaches the target, the direction is set to 'never trust' and the assistant will answer only from
its translation memory (phrasebook/dictionary entries) and say so otherwise. Re-run after every training run."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kmt.assistant import Assistant
from kmt.data import read_tsv, SRC_LANGS
from kmt.evaluate import chrf
from kmt.tokenizer import fold_text


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="checkpoints"); ap.add_argument("--data", default="data")
    ap.add_argument("--target-chrf", type=float, default=0.5); ap.add_argument("--max-per-direction", type=int, default=0, help="evaluate a random sample of this many held-out sentences (large held-out sets)"); ap.add_argument("--min-examples", type=int, default=20)
    a = ap.parse_args()
    bot = Assistant(a.ckpt, os.path.join(a.ckpt, "_calib_mem"), a.data)
    out = {"default": 1.01, "target_chrf": a.target_chrf, "directions": {}}
    for lang in SRC_LANGS:
        rows = [r for r in read_tsv(os.path.join(a.data, "heldout", f"{lang}-en.tsv"), 2) if len(r[0].split()) >= 3]
        for src, tgt, pairs in ((lang, "en", rows), ("en", lang, [(e, l) for l, e in rows])):
            if len(pairs) < a.min_examples:
                out["directions"][f"{src}>{tgt}"] = dict(min_conf=1.01, n=len(pairs), note="too few held-out examples to calibrate"); continue
            if a.max_per_direction and len(pairs) > a.max_per_direction:
                import random; pairs = random.Random(0).sample(pairs, a.max_per_direction)
            res = []
            for s, ref in pairs:
                r = bot.translate(s, src, tgt, use_memory=False, force=True)
                res.append((r.get("confidence", 0.0), chrf(fold_text(r["text"]).lower(), fold_text(ref).lower())))
            res.sort(reverse=True); conf = np.array([c for c, _ in res]); f = np.array([x for _, x in res])
            thr, best = 1.01, None
            for i in range(a.min_examples, len(res) + 1):          # smallest prefix (most confident i examples) that still meets the target
                if f[:i].mean() >= a.target_chrf:
                    thr, best = float(conf[i - 1]), float(f[:i].mean())
            out["directions"][f"{src}>{tgt}"] = dict(min_conf=round(thr, 3), n=len(res), mean_chrf_all=round(float(f.mean()), 3),
                                                     mean_chrf_trusted=best, note="trusted" if thr <= 1 else "never trusted: target not reached at any confidence")
    json.dump(out, open(os.path.join(a.ckpt, "calibration.json"), "w"), indent=2)
    for k, v in out["directions"].items():
        print(f"{k:<10} held-out n={v['n']:<4} mean chrF {v.get('mean_chrf_all', '-')!s:<6} -> {v['note']}")


if __name__ == "__main__":
    main()
