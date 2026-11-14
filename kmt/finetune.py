"""Continual learning: fine-tune on corrections collected in memory/feedback.jsonl,
mixed with replay from the original data so old knowledge is not forgotten.
   python finetune.py --steps 150
"""
import argparse, os, shutil
import numpy as np
from kmt.tokenizer import BPE
from kmt.data import Dataset, feedback_load, feedback_save, feedback_to_examples
from kmt.transformer import Transformer
from kmt.optim import Adam

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--ckpt", default="checkpoints")
ap.add_argument("--memory", default="memory"); ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=4e-4)
ap.add_argument("--replay", type=float, default=0.5, help="fraction of each batch drawn from old data")
a = ap.parse_args()

fb_path = os.path.join(a.memory, "feedback.jsonl")
recs = feedback_load(fb_path)
new = [r for r in recs if not r.get("used")]
if not new:
    raise SystemExit("No new feedback to learn from.")
tok = BPE.load(os.path.join(a.ckpt, "tokenizer.json"))
model, extra = Transformer.load(os.path.join(a.ckpt, "kmt.npz"))
ctx = model.cfg["ctx"]
fb = [e for r in new for e in feedback_to_examples(tok, r) if len(e["ids"]) <= ctx]
unk = sum(int((e["ids"] == tok.unk).sum()) for e in fb)
if unk:
    print(f"note: {unk} unknown tokens in feedback (new characters); retrain the tokenizer occasionally.")
replay = Dataset(a.data, tok, ctx).train
print(f"{len(fb)} feedback examples, {len(replay)} replay examples")

opt = Adam(model.parameters(), lr=a.lr, wd=0.0)
rng = np.random.default_rng(1)
n_old = int(a.batch * a.replay) if replay else 0
for step in range(1, a.steps + 1):
    ex = [fb[i] for i in rng.integers(0, len(fb), a.batch - n_old)] + \
         [replay[i] for i in rng.integers(0, len(replay), n_old)]
    ids, mask = Dataset.collate(ex, tok.pad)
    loss = model.loss(ids, mask, train=True)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 25 == 0:
        print(f"step {step:4d} loss {float(loss.data):.3f}", flush=True)

path = os.path.join(a.ckpt, "kmt.npz")
shutil.copy(path, path + ".bak")
langs = set(extra.get("langs", [])) | {r.get("lang") or r.get("src_lang") for r in new if r["task"] != "translate"}
intents = set(extra.get("intents", [])) | {r["label"] for r in new if r["task"] == "intent"}
extra.update(langs=sorted(l for l in langs if l), intents=sorted(intents), steps=extra.get("steps", 0) + a.steps)
model.save(path, extra)
for r in recs:
    if not r.get("used"):
        r["used"] = True
feedback_save(fb_path, recs)
print("fine-tuned and saved; previous checkpoint kept as kmt.npz.bak")
