"""Train the tokenizer (if needed) and the transformer.
   python train.py --data data --steps 6000
Robust for long runs: crash-safe checkpoints, exact resume (weights + optimizer + averaging state),
Ctrl-C saves and exits cleanly, NaN protection, gradient accumulation for larger effective batches.
--steps is the TOTAL number of steps; --resume continues where the last run stopped."""
import argparse, json, math, os, time
import numpy as np
from kmt.autograd import set_dtype
from kmt.tokenizer import BPE
from kmt.data import Dataset, collect_texts
from kmt.transformer import Transformer, atomic_savez
from kmt.optim import Adam, EMA
from kmt.evaluate import group_perplexity

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--out", default="checkpoints")
ap.add_argument("--steps", type=int, default=4000, help="total steps (resume continues up to this number)")
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--accum", type=int, default=1,
                help="gradient accumulation: effective batch = batch x accum, memory stays that of one batch")
ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--heads", type=int, default=4); ap.add_argument("--ctx", type=int, default=128)
ap.add_argument("--dropout", type=float, default=0.0, help="try 0.1 once val loss rises while train loss falls (overfitting)")
ap.add_argument("--smoothing", type=float, default=0.0, help="label smoothing; measured to SLOW context-lookup skills on small data")
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--ema", type=float, default=0.995)
ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
ap.add_argument("--resume", action="store_true", help="continue from the last saved state in --out")
ap.add_argument("--stop_after", type=int, default=0, help="stop (saving state) after this step; continue later with --resume")
ap.add_argument("--no-fold", action="store_true", help="keep special letters/marks (default: ASCII-fold, matching the OCR-derived data)")
ap.add_argument("--langs", default="", help="comma list, e.g. krio: train a single-language specialist (default: all)")
ap.add_argument("--max-per-group", type=int, default=0); ap.add_argument("--mono-frac", type=float, default=1.0)
ap.add_argument("--max-rep", type=int, default=10, help="oversampling cap for small groups")
ap.add_argument("--merges", type=int, default=800); ap.add_argument("--retok", action="store_true")
ap.add_argument("--eval_every", type=int, default=250); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

set_dtype(a.dtype)
os.makedirs(a.out, exist_ok=True)
tok_path, best_path, state_path = (os.path.join(a.out, n) for n in ("tokenizer.json", "kmt.npz", "last_state.npz"))
if os.path.exists(tok_path) and not a.retok:
    tok = BPE.load(tok_path); print("loaded tokenizer", len(tok))
else:
    texts = collect_texts(a.data)
    if not texts:
        raise SystemExit("No text found in the data folder. See data/README.md.")
    t0 = time.time(); tok = BPE.train(texts, a.merges, fold=not a.no_fold); tok.save(tok_path)
    print(f"trained tokenizer, vocab {len(tok)} ({time.time()-t0:.0f}s)")

t0 = time.time()
ds = Dataset(a.data, tok, a.ctx, seed=a.seed, langs=[x for x in a.langs.split(',') if x] or None, max_per_group=a.max_per_group or None,
             mono_frac=a.mono_frac, max_rep=a.max_rep)
print(f"dataset built in {time.time()-t0:.0f}s | groups:", ds.group_sizes)
print("train examples (after balancing):", len(ds.train), "| skipped (too long for --ctx):", ds.skipped,
      "| validation is a copy of train (too little data):", ds.val_is_train)
if not ds.train:
    raise SystemExit("No training examples.")
if ds.skipped > 0.05 * (len(ds.train) + ds.skipped):
    print(f"WARNING: {ds.skipped} examples were longer than --ctx {a.ctx} and dropped; consider a larger --ctx.")

model = Transformer(len(tok), a.d, a.layers, a.heads, a.ctx, a.dropout, a.seed)
opt = Adam(model.parameters(), lr=a.lr)
ema = EMA(model.params, a.ema)
same_arch = lambda c1, c2: {k: v for k, v in c1.items() if k != "dropout"} == {k: v for k, v in c2.items() if k != "dropout"}
start, best = 0, float("inf")


def save_state(step):
    arrs = {f"w.{k}": v.data for k, v in model.params.items()}
    arrs.update({f"e.{k}": v for k, v in ema.shadow.items()})
    arrs.update({f"m.{i}": m for i, m in enumerate(opt.m)}); arrs.update({f"v.{i}": v for i, v in enumerate(opt.v)})
    atomic_savez(state_path, meta=json.dumps(dict(step=step, t=opt.t, n=ema.n, best=best, cfg=model.cfg)), **arrs)


if a.resume and os.path.exists(state_path):                      # exact resume: weights, Adam moments, EMA, step
    z = np.load(state_path, allow_pickle=False); meta = json.loads(str(z["meta"]))
    assert same_arch(meta["cfg"], model.cfg), "architecture flags differ from the saved state"
    for k, v in model.params.items():
        v.data[...] = z[f"w.{k}"]; ema.shadow[k][...] = z[f"e.{k}"]
    for i in range(len(opt.m)):
        opt.m[i][...] = z[f"m.{i}"]; opt.v[i][...] = z[f"v.{i}"]
    opt.t, ema.n, start, best = meta["t"], meta["n"], meta["step"], meta["best"]
    print(f"resumed exactly at step {start}")
elif a.resume and os.path.exists(best_path):                      # weights only (older checkpoints)
    old, _ = Transformer.load(best_path); assert same_arch(old.cfg, model.cfg), "architecture flags differ"
    for k, v in model.params.items():
        v.data[...] = old.params[k].data
    ema = EMA(model.params, a.ema); print("resumed weights only (no optimizer state found)")
print(f"model: {model.n_params():,} parameters ({a.dtype}) | effective batch {a.batch * a.accum}")

rng = np.random.default_rng(a.seed + start)
batches = ds.batches(a.batch, rng, tok.pad)
extra = lambda s: dict(langs=ds.lm_langs, intents=ds.intent_labels, steps=s)
t0, toks, skipped_in_row, step = time.time(), 0, 0, start
try:
    for step in range(start + 1, a.steps + 1):
        warm = max(1, int(0.05 * a.steps))
        opt.lr = a.lr * (step / warm if step < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, a.steps - warm))))
        opt.zero_grad(); total = 0.0
        for _ in range(a.accum):
            ids, mask = next(batches); toks += ids.size
            loss = model.loss(ids, mask, train=True, smoothing=a.smoothing) / a.accum
            loss.backward(); total += float(loss.data)
        if not np.isfinite(total):
            raise SystemExit(f"Loss became {total} at step {step}. Lower --lr (try {a.lr / 3:.4g}) or check the data.")
        gnorm = opt.step(); ema.update(model.params)
        if not np.isfinite(gnorm):
            skipped_in_row += 1
            if skipped_in_row > 20:
                raise SystemExit("Gradients stayed non-finite for 20 steps. Lower --lr.")
        else:
            skipped_in_row = 0
        if step % 25 == 0:
            print(f"step {step:6d}  loss {total:.3f}  lr {opt.lr:.5f}  {toks/(time.time()-t0):.0f} tok/s  {time.time()-t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ema.swap(model.params)                               # evaluate the averaged weights
            ppl = group_perplexity(model, ds, tok.pad)
            score = float(np.mean([math.log(v) for v in ppl.values()]))
            print("  val perplexity:", {g: round(v, 2) for g, v in ppl.items()}, "| mean log-ppl", round(score, 4))
            if score <= best:
                best = score; model.save(best_path, extra(step)); print("  saved best")
            ema.swap(model.params)                               # back to live weights
            save_state(step)
        if a.stop_after and step >= a.stop_after and step < a.steps:
            save_state(step); print(f"stopped at step {step} as requested; continue with --resume"); raise SystemExit(0)
except KeyboardInterrupt:
    print("\nInterrupted: saving state so you can continue with --resume ...")
    save_state(step); raise SystemExit(0)
print("done; best checkpoint:", best_path)
