"""Byte-pair-encoding tokenizer trained on your own Krio/Mende/Temne text."""
import json, re, unicodedata
from collections import Counter

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<tr>", "<intent>", "<qa>",
            "<krio>", "<mende>", "<temne>", "<en>"]
_PRE = re.compile(r"\s?\w+|\s?[^\w\s]", re.UNICODE)
SP = "\u2581"                                     # visible marker for a leading space


_FOLD = str.maketrans({"\u025b": "e", "\u0190": "E", "\u0254": "o", "\u0186": "O", "\u014b": "ng", "\u014a": "Ng", "\u0253": "b", "\u0257": "d", "\u028c": "a", "\u0259": "a", "\u03b5": "e", "\u028c": "a", "\u0259": "e", "\u2019": "'", "\u2018": "'"})


def fold_text(s):
    """ASCII-fold: strip tone/length/dot marks and map open vowels (\u025b \u0254) to e/o. Used so text typed with special letters
    matches training data whose source books could only be OCR'd without them."""
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).translate(_FOLD)


def pretokens(text):
    return [t.replace(" ", SP) for t in _PRE.findall(" ".join(text.split()))]


def _merge(word, pair):
    out, i = [], 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1]); i += 2
        else:
            out.append(word[i]); i += 1
    return tuple(out)


class BPE:
    def __init__(self, vocab, merges, fold=False):
        self.fold = fold
        self.itos = list(vocab)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.merges = [tuple(m) for m in merges]
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.cache = {}
        for name in ("pad", "unk", "bos", "eos", "sep"):
            setattr(self, name, self.stoi[f"<{name}>"])

    def __len__(self):
        return len(self.itos)

    def id(self, token):
        return self.stoi[token]

    @classmethod
    def train(cls, texts, n_merges=800, min_freq=2, fold=False):
        words = Counter()
        for t in texts:
            t = fold_text(t) if fold else t
            for w in pretokens(t):
                words[tuple(w)] += 1
        vocab = list(SPECIALS) + sorted({c for w in words for c in w})
        seen, merges = set(vocab), []
        for _ in range(n_merges):
            pairs = Counter()
            for w, f in words.items():
                for a, b in zip(w, w[1:]):
                    pairs[(a, b)] += f
            if not pairs:
                break
            best, f = pairs.most_common(1)[0]
            if f < min_freq:
                break
            merges.append(best)
            if best[0] + best[1] not in seen:
                vocab.append(best[0] + best[1]); seen.add(best[0] + best[1])
            new = Counter()
            for w, fr in words.items():
                new[_merge(w, best)] += fr
            words = new
        return cls(vocab, merges, fold)

    def norm(self, text):
        return fold_text(text) if self.fold else text

    def _bpe(self, w):
        if w in self.cache:
            return self.cache[w]
        syms = list(w)
        while len(syms) > 1:
            r, i = min((self.rank.get((a, b), 1 << 60), i) for i, (a, b) in enumerate(zip(syms, syms[1:])))
            if r == 1 << 60:
                break
            syms[i:i + 2] = [syms[i] + syms[i + 1]]
        self.cache[w] = syms
        return syms

    def encode(self, text):
        ids = []
        for w in pretokens(self.norm(text)):
            ids += [self.stoi.get(s, self.unk) for s in self._bpe(w)]
        return ids

    def decode(self, ids):
        s = "".join(self.itos[i] for i in ids if self.itos[i] not in SPECIALS)
        return s.replace(SP, " ").strip()

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.itos, "merges": self.merges, "fold": self.fold}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["vocab"], d["merges"], d.get("fold", False))
