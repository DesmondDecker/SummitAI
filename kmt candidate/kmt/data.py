"""Datasets and task formats.

Files (UTF-8) in the data folder:
  krio.txt mende.txt temne.txt en.txt     one sentence per line (monolingual)
  parallel/krio-en.tsv  (also mende-en, temne-en)   source<TAB>English
  heldout/krio-en.tsv (etc.)                        pairs used ONLY for validation (made by tools/bible_align.py)
  intents.tsv                                       lang<TAB>text<TAB>label
  qa.tsv                                            lang<TAB>context<TAB>question<TAB>answer
  synthetic.tsv   (made by tools/backtranslate.py)  src_lang<TAB>src<TAB>tgt_lang<TAB>tgt

Every example is a token sequence plus a mask saying which tokens the model is trained to predict:
  lm        <bos> <lang> text <eos>
  translate <bos> <tr> <src> text <sep> <tgt> text <eos>
  intent    <bos> <intent> <lang> text <sep> label <eos>
  qa        <bos> <qa> <lang> context <sep> question <sep> answer <eos>
"""
import collections, json, os, random, re, unicodedata
import numpy as np

LANGS = ["krio", "mende", "temne", "en"]
SRC_LANGS = ["krio", "mende", "temne"]
tag = lambda lang: f"<{lang}>"


def _clean(l):
    return unicodedata.normalize("NFC", l.strip())


def read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", errors="replace") as f:      # utf-8-sig: tolerate a Windows BOM
        return [l for l in map(_clean, f) if l and not l.startswith("#")]


def read_tsv(path, ncols):
    rows = []
    for l in read_lines(path):
        parts = [p.strip() for p in l.split("\t")]
        if len(parts) >= ncols and all(parts[:ncols]):
            rows.append(parts[:ncols])
    return rows


# ------------------------------------------------------------ prompts / examples
def mono_prompt(tok, lang):
    return [tok.bos, tok.id(tag(lang))]


def translate_prompt(tok, src_lang, src, tgt_lang):
    return [tok.bos, tok.id("<tr>"), tok.id(tag(src_lang))] + tok.encode(src) + [tok.sep, tok.id(tag(tgt_lang))]


def intent_prompt(tok, lang, text):
    return [tok.bos, tok.id("<intent>"), tok.id(tag(lang))] + tok.encode(text) + [tok.sep]


def qa_prompt(tok, lang, context, question):
    return ([tok.bos, tok.id("<qa>"), tok.id(tag(lang))] + tok.encode(context) + [tok.sep]
            + tok.encode(question) + [tok.sep])


def _ex(tok, prompt, target, group, meta):
    ids = prompt + target + [tok.eos]
    mask = [0.0] * len(prompt) + [1.0] * (len(target) + 1)
    return dict(ids=np.array(ids, dtype=np.int64), mask=np.array(mask), group=group, meta=meta)


def mono_example(tok, lang, text):
    return _ex(tok, mono_prompt(tok, lang), tok.encode(text), f"lm:{lang}", dict(task="lm", lang=lang, text=text))


def translate_example(tok, sl, src, tl, tgt):
    return _ex(tok, translate_prompt(tok, sl, src, tl), tok.encode(tgt), f"tr:{sl}>{tl}",
               dict(task="translate", src_lang=sl, src=src, tgt_lang=tl, tgt=tgt))


def intent_example(tok, lang, text, label):
    return _ex(tok, intent_prompt(tok, lang, text), tok.encode(label), f"intent:{lang}",
               dict(task="intent", lang=lang, text=text, label=label))


def qa_example(tok, lang, context, question, answer):
    return _ex(tok, qa_prompt(tok, lang, context, question), tok.encode(answer), f"qa:{lang}",
               dict(task="qa", lang=lang, context=context, question=question, answer=answer))


# ------------------------------------------------------------ raw text for tokenizer
def collect_texts(data_dir):
    texts = []
    for lang in LANGS:
        texts += read_lines(os.path.join(data_dir, lang + ".txt"))
    for lang in SRC_LANGS:
        for s, t in read_tsv(os.path.join(data_dir, "parallel", f"{lang}-en.tsv"), 2):
            texts += [s, t]
    for _, text, label in read_tsv(os.path.join(data_dir, "intents.tsv"), 3):
        texts += [text, label]
    for _, c, q, ans in read_tsv(os.path.join(data_dir, "qa.tsv"), 4):
        texts += [c, q, ans]
    for _, src, _, tgt in read_tsv(os.path.join(data_dir, "synthetic.tsv"), 4):
        texts += [src, tgt]
    return texts


# ------------------------------------------------------------ dataset
def _val_flags(n, frac, rng):
    """Hold out a validation subset only when there is enough data (>=20 items)."""
    flags = [False] * n
    if n >= 20 and frac > 0:
        for i in rng.sample(range(n), max(1, int(n * frac))):
            flags[i] = True
    return flags


class Dataset:
    def __init__(self, data_dir, tok, ctx, val_frac=0.05, seed=0, balance=True, max_rep=10, langs=None, max_per_group=None, mono_frac=1.0):
        rng = random.Random(seed)
        held = {l: read_tsv(os.path.join(data_dir, "heldout", f"{l}-en.tsv"), 2) for l in SRC_LANGS}
        if any(held.values()):                                   # explicit held-out books replace random splitting
            val_frac = 0.0
        train, val = collections.defaultdict(list), collections.defaultdict(list)
        self.skipped = 0

        def add(bucket, ex):
            if len(ex["ids"]) > ctx:
                self.skipped += 1
            else:
                bucket[ex["group"]].append(ex)

        for lang in LANGS:                                       # monolingual text
            rows = read_lines(os.path.join(data_dir, lang + ".txt"))
            for line, isv in zip(rows, _val_flags(len(rows), val_frac, rng)):
                add(val if isv else train, mono_example(tok, lang, line))
        for lang in SRC_LANGS:                                   # parallel text (both directions)
            rows = read_tsv(os.path.join(data_dir, "parallel", f"{lang}-en.tsv"), 2)
            for (s, t), isv in zip(rows, _val_flags(len(rows), val_frac, rng)):
                b = val if isv else train                        # split by pair: no leakage across directions
                add(b, translate_example(tok, lang, s, "en", t))
                add(b, translate_example(tok, "en", t, lang, s))
                if not isv and rng.random() < mono_frac:                 # parallel sides also give free monolingual data
                    add(b, mono_example(tok, lang, s)); add(b, mono_example(tok, "en", t))
        for lang, rows in held.items():
            for s, t in rows:
                add(val, translate_example(tok, lang, s, "en", t)); add(val, translate_example(tok, "en", t, lang, s))
                add(val, mono_example(tok, lang, s)); add(val, mono_example(tok, "en", t))
        rows = read_tsv(os.path.join(data_dir, "intents.tsv"), 3)
        for (lang, text, label), isv in zip(rows, _val_flags(len(rows), val_frac, rng)):
            add(val if isv else train, intent_example(tok, lang, text, label))

        rows = read_tsv(os.path.join(data_dir, "qa.tsv"), 4)
        for (lang, c, q, ans), isv in zip(rows, _val_flags(len(rows), val_frac, rng)):
            add(val if isv else train, qa_example(tok, lang, c, q, ans))
        for sl, src, tl, tgt in read_tsv(os.path.join(data_dir, "synthetic.tsv"), 4):   # back-translated pairs
            ex = translate_example(tok, sl, src, tl, tgt); ex["group"] = f"syn:{sl}>{tl}"
            add(train, ex)                                       # train only, one direction only

        if langs:                                                    # e.g. langs=["krio"]: a single-language specialist
            ok = lambda g: g.split(":")[0] in ("lm", "tr") and set(re.findall(r"krio|mende|temne", g)) <= set(langs)
            train = {g: v for g, v in train.items() if ok(g)}; val = {g: v for g, v in val.items() if ok(g)}
        if max_per_group:
            train = {g: (rng.sample(v, max_per_group) if len(v) > max_per_group else v) for g, v in train.items()}
        self.lm_langs = sorted(g.split(":")[1] for g in train if g.startswith("lm:"))
        self.intent_labels = sorted({e["meta"]["label"] for g in train if g.startswith("intent:") for e in train[g]})
        self.group_sizes = {g: len(v) for g, v in train.items()}
        self.val_by_group = dict(val)

        top = max(self.group_sizes.values(), default=1)
        self.train = []
        for g, exs in train.items():
            rep = min(max_rep, max(1, round(top / len(exs)))) if balance else 1   # oversample small groups
            self.train += exs * rep
        self.val = [e for v in val.values() for e in v]
        self.val_is_train = not self.val
        if self.val_is_train:
            self.val_by_group = {g: v[:16] for g, v in train.items()}
            self.val = self.train[:64]

    @staticmethod
    def collate(examples, pad):
        L = max(len(e["ids"]) for e in examples)
        ids = np.full((len(examples), L), pad, dtype=np.int64)
        mask = np.zeros((len(examples), L))
        for i, e in enumerate(examples):
            ids[i, :len(e["ids"])] = e["ids"]; mask[i, :len(e["mask"])] = e["mask"]
        return ids, mask

    def batches(self, B, rng, pad, pool_mult=16):
        """Endless stream of length-bucketed batches: similar-length examples share a batch,
        so far less compute is wasted on padding."""
        n = len(self.train)
        while True:
            idx = sorted(rng.integers(0, n, B * pool_mult), key=lambda i: len(self.train[i]["ids"]))
            chunks = [idx[i:i + B] for i in range(0, len(idx), B)]
            for j in rng.permutation(len(chunks)):
                yield self.collate([self.train[i] for i in chunks[j]], pad)

    def batch(self, B, rng, pad, pool=None):
        pool = self.train if pool is None else pool
        return self.collate([pool[i] for i in rng.integers(0, len(pool), B)], pad)


# ------------------------------------------------------------ feedback log (continual learning)
def feedback_append(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(rec, used=False), ensure_ascii=False) + "\n")


def feedback_load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def feedback_save(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def feedback_to_examples(tok, rec):
    t = rec["task"]
    if t == "translate":
        return [translate_example(tok, rec["src_lang"], rec["src"], rec["tgt_lang"], rec["tgt"])]
    if t == "intent":
        return [intent_example(tok, rec["lang"], rec["text"], rec["label"])]
    if t == "lm":
        return [mono_example(tok, rec["lang"], rec["text"])]
    return []
