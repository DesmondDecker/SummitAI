"""Back-translation: turn plain monolingual text into extra TRAINING PAIRS using the current model.
   python tools/backtranslate.py --lang krio --min-conf 0.8 --limit 2000
For each real line in data/<lang>.txt the model writes an English version; the pair is used to train
English->lang where the TARGET side is real text (this is what makes the technique work). The same is
done in reverse with data/en.txt. Only confident outputs are kept. Then retrain/fine-tune with:
   python train.py --resume --steps 2000 ...
Output goes to data/synthetic.tsv (training only, never used for validation).
Caution: this amplifies whatever the model already gets right AND its systematic errors, so use a high
--min-conf, spot-check the file with a native speaker, and never treat it as a substitute for real pairs."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kmt.assistant import Assistant
from kmt.data import read_lines

ap = argparse.ArgumentParser()
ap.add_argument("--lang", required=True); ap.add_argument("--data", default="data")
ap.add_argument("--ckpt", default="checkpoints"); ap.add_argument("--min-conf", type=float, default=0.8)
ap.add_argument("--limit", type=int, default=1000); ap.add_argument("--beam", type=int, default=2)
a = ap.parse_args()

bot = Assistant(a.ckpt, "memory", a.data)
out_path = os.path.join(a.data, "synthetic.tsv")
have = set(read_lines(out_path))
rows = []
for src_lang, tgt_lang, path in ((a.lang, "en", f"{a.lang}.txt"), ("en", a.lang, "en.txt")):
    n = 0
    for line in read_lines(os.path.join(a.data, path))[: a.limit]:
        r = bot.translate(line, src_lang, tgt_lang, use_memory=False, beam=a.beam, force=True)
        if r["source"] != "model" or not r["text"] or r["confidence"] < a.min_conf:
            continue
        # real text is always the TARGET of the training pair; the model's output is the (synthetic) source
        row = f"{tgt_lang}\t{r['text']}\t{src_lang}\t{line}"
        if row not in have:
            rows.append(row); have.add(row); n += 1
    print(f"{src_lang}->{tgt_lang}: kept {n} confident pairs")
with open(out_path, "a", encoding="utf-8") as f:
    f.write("".join(r + "\n" for r in rows))
print(f"appended {len(rows)} pairs to {out_path}")
