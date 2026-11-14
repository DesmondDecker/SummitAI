"""python evaluate.py  -> perplexity, language-ID accuracy and translation chrF on held-out data."""
import argparse
from kmt.assistant import Assistant
from kmt.data import Dataset
from kmt.evaluate import group_perplexity, lid_accuracy, translation_scores, qa_accuracy, next_token_accuracy

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--ckpt", default="checkpoints")
a = ap.parse_args()
bot = Assistant(a.ckpt, "memory", a.data)
ds = Dataset(a.data, bot.tok, bot.ctx)
if ds.val_is_train:
    print("WARNING: too little data for a held-out set; numbers below are on training data.")
print("perplexity      :", {g: round(v, 2) for g, v in group_perplexity(bot.model, ds, bot.tok.pad).items()})
print("language-ID acc :", {k: round(v, 2) for k, v in lid_accuracy(bot, ds).items()})
print("translation chrF:", {k: round(v, 3) for k, v in translation_scores(bot, ds).items()})
qa = qa_accuracy(bot, ds)
if qa:
    print("QA exact match  :", {k: round(v, 2) for k, v in qa.items()})
print("next-token top1/top5:", {k: (round(float(v[0]), 2), round(float(v[1]), 2)) for k, v in next_token_accuracy(bot.model, ds, bot.tok.pad).items()})
