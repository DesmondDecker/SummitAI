"""Metrics: per-group perplexity, language-ID accuracy, translation chrF."""
import math
from collections import Counter
import numpy as np
from .autograd import no_grad
from .data import Dataset
from .tokenizer import fold_text


def chrf(hyp, ref, n=6, beta=2.0):
    grams = lambda s, k: Counter(s.replace(" ", "")[i:i + k] for i in range(len(s.replace(" ", "")) - k + 1))
    ps, rs = [], []
    for k in range(1, n + 1):
        h, r = grams(hyp, k), grams(ref, k)
        if not h or not r:
            continue
        ov = sum((h & r).values())
        ps.append(ov / sum(h.values())); rs.append(ov / sum(r.values()))
    if not ps:
        return 0.0
    p, r = float(np.mean(ps)), float(np.mean(rs))
    return 0.0 if p + r == 0 else (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def group_perplexity(model, ds, pad, cap=48, B=16):
    """Perplexity on the target tokens of each task/language group."""
    out = {}
    for g, exs in ds.val_by_group.items():
        exs = exs[:cap]; tot = 0.0; n = 0
        with no_grad():
            for i in range(0, len(exs), B):
                chunk = exs[i:i + B]
                ids, mask = Dataset.collate(chunk, pad)
                l = float(model.loss(ids, mask, train=False).data); w = mask[:, 1:].sum()
                tot += l * w; n += w
        out[g] = math.exp(min(20.0, tot / max(n, 1)))
    return out


def lid_accuracy(assistant, ds, cap=40):
    res = {}
    for g, exs in ds.val_by_group.items():
        if not g.startswith("lm:"):
            continue
        lang = g.split(":")[1]
        ok = sum(assistant.identify(e["meta"]["text"])[0] == lang for e in exs[:cap])
        res[lang] = ok / max(1, len(exs[:cap]))
    return res


def translation_scores(assistant, ds, cap=30):
    res = {}
    for g, exs in ds.val_by_group.items():
        if not g.startswith("tr:"):
            continue
        sc = []
        for e in exs[:cap]:
            m = e["meta"]
            hyp = assistant.translate(m["src"], m["src_lang"], m["tgt_lang"], use_memory=False, force=True)["text"]
            sc.append(chrf(fold_text(hyp).lower(), fold_text(m["tgt"]).lower()))
        res[g] = float(np.mean(sc)) if sc else 0.0
    return res


def qa_accuracy(assistant, ds, cap=40):
    res = {}
    for g, exs in ds.val_by_group.items():
        if g.startswith("qa:") and exs:
            ok = sum(assistant.ask(e["meta"]["question"], e["meta"]["lang"], e["meta"]["context"], min_conf=0.0)
                     ["guess"] == e["meta"]["answer"] for e in exs[:cap])
            res[g] = ok / max(1, len(exs[:cap]))
    return res


def next_token_accuracy(model, ds, pad, cap=48, B=16):
    """Prediction quality: how often the true next token is the model's top-1 / top-5 guess."""
    res = {}
    for g, exs in ds.val_by_group.items():
        if not g.startswith("lm:"):
            continue
        hit1 = hit5 = tot = 0
        for i in range(0, len(exs[:cap]), B):
            ids, mask = Dataset.collate(exs[i:i + B], pad)
            logits, _ = model.infer(ids[:, :-1], all_pos=True)
            top5 = np.argsort(-logits, -1)[..., :5]; tgt = ids[:, 1:]; m = mask[:, 1:] > 0
            hit1 += ((top5[..., 0] == tgt) & m).sum(); hit5 += ((top5 == tgt[..., None]).any(-1) & m).sum(); tot += m.sum()
        res[g.split(":")[1]] = (hit1 / max(1, tot), hit5 / max(1, tot))
    return res
