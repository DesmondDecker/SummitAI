"""Retrieval memory: translation memory + remembered facts, using hashed character n-gram vectors.
Needs no training, so it learns instantly from corrections.

Built to scale to ~100k items: float32 vectors in a growing buffer (no per-search rebuild),
vectorised n-gram hashing, an exact-match index (O(1)), and top-k selection without a full sort."""
import json, os, re, unicodedata
import numpy as np
from .tokenizer import fold_text

_MULT = np.uint64(1000003)


def _norm(text):
    """Canonical key: case, special letters and punctuation do not matter ('Just now.' == 'just now')."""
    t = fold_text(unicodedata.normalize("NFC", " ".join(text.lower().split())))
    return " ".join(re.sub(r"[^\w\s]|_", " ", t).split())


def _lev(a, b):
    """Edit distance between two short strings."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class MemoryStore:
    def __init__(self, path=None, dim=1024, capacity=256):
        self.path, self.dim = path, dim
        self.items, self._seen, self._exact = [], set(), {}
        self._buf = np.zeros((capacity, dim), dtype=np.float32)
        self._kind, self._lang, self._tgt = (np.zeros(capacity, dtype=np.int32) for _ in range(3))
        self._codes = {None: 0}                         # string -> small int for fast filtering
        if path and os.path.exists(path):
            with open(path, encoding="utf-8-sig") as f:
                for line in f:
                    if line.strip():
                        try:
                            self.add(json.loads(line), persist=False)
                        except json.JSONDecodeError:
                            pass                          # skip a corrupted/truncated line instead of failing

    # ---- vectors ----------------------------------------------------------
    def _vec(self, text):
        cp = np.frombuffer((" " + _norm(text) + " ").encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
        v = np.zeros(self.dim, dtype=np.float32)
        for n in (2, 3, 4):
            if len(cp) < n:
                continue
            h = np.zeros(len(cp) - n + 1, dtype=np.uint64)
            with np.errstate(over="ignore"):                    # uint64 wrap-around is intended
                for i in range(n):
                    h = h * _MULT + cp[i:len(cp) - n + 1 + i]
            v += np.bincount((h % np.uint64(self.dim)).astype(np.int64), minlength=self.dim).astype(np.float32)
        v = np.log1p(v); nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def _code(self, s):
        return self._codes.setdefault(s, len(self._codes))

    @staticmethod
    def _key(it):
        return it["src"] if it["kind"] == "translation" else it["text"]

    # ---- writing ----------------------------------------------------------
    def add(self, item, persist=True):
        sig = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if sig in self._seen:
            return False
        self._seen.add(sig)
        i = len(self.items)
        if i >= len(self._buf):                             # amortised O(1) growth
            n = len(self._buf) * 2
            self._buf = np.vstack([self._buf, np.zeros((n - len(self._buf), self.dim), np.float32)])
            for name in ("_kind", "_lang", "_tgt"):
                setattr(self, name, np.concatenate([getattr(self, name), np.zeros(n - len(getattr(self, name)), np.int32)]))
        self._buf[i] = self._vec(self._key(item))
        self._kind[i], self._lang[i], self._tgt[i] = self._code(item["kind"]), self._code(item.get("lang")), self._code(item.get("tgt_lang"))
        self._exact.setdefault((item["kind"], item.get("lang"), item.get("tgt_lang"), _norm(self._key(item))), []).append(i)
        self.items.append(item)
        if persist and self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(sig + "\n")
        return True

    def add_pair(self, src_lang, src, tgt_lang, tgt, q=None):
        """Store a translation in both directions. q = optional 0-1 quality (higher wins when answers conflict)."""
        extra = {} if q is None else {"q": round(float(q), 2)}
        a = self.add(dict(kind="translation", lang=src_lang, src=src, tgt_lang=tgt_lang, tgt=tgt, **extra))
        b = self.add(dict(kind="translation", lang=tgt_lang, src=tgt, tgt_lang=src_lang, tgt=src, **extra))
        return a or b

    def add_fact(self, text, lang=None):
        return self.add(dict(kind="fact", lang=lang, text=text))

    # ---- reading ----------------------------------------------------------
    def search(self, query, k=3, kind=None, lang=None, tgt_lang=None):
        """Returns [(similarity, item)] best first. An exact (normalised) match scores 1.0 instantly."""
        n = len(self.items)
        if n == 0:
            return []
        out, used = [], set()
        if kind and (kind, lang, tgt_lang, _norm(query)) in self._exact:
            for i in reversed(self._exact[(kind, lang, tgt_lang, _norm(query))]):   # newest (a correction) first
                out.append((1.0, self.items[i])); used.add(i)
                if len(out) >= k:
                    return out
        sims = self._buf[:n] @ self._vec(query)
        ok = np.ones(n, dtype=bool)
        for arr, val in ((self._kind, kind), (self._lang, lang), (self._tgt, tgt_lang)):
            if val is not None:
                ok &= arr[:n] == self._codes.get(val, -1)
        sims = np.where(ok, sims, -np.inf)
        m = min(k + len(used), int(ok.sum()))
        if m > 0:
            top = np.argpartition(-sims, m - 1)[:m] if m < n else np.arange(n)
            for i in top[np.argsort(-sims[top])]:
                if i not in used and np.isfinite(sims[i]):
                    out.append((float(sims[i]), self.items[i]))
                    if len(out) >= k:
                        break
        return out

    def _ranked(self, idxs):
        """Items for one canonical key: highest quality first, newest first among equals; distinct answers only."""
        its = sorted((self.items[i] for i in reversed(idxs)), key=lambda it: -(it.get("q") or 0.0))
        seen, out = set(), []
        for it in its:
            k = it["tgt"].strip().lower()
            if k not in seen:
                seen.add(k); out.append(it)
        return out

    def lookup(self, query, lang, tgt_lang, accept=0.92, max_suggest=3, short_auto=False):
        """Policy-aware lookup. Returns dict(status, answer, alternatives, suggestions, score, matched) with status
          exact   canonical key matches an entry (punctuation/case/special letters ignored)
          near    a long query (3+ words) is a close variant of one entry  [short queries only if short_auto=True and unambiguous]
          suggest nothing safe to answer, but similar entries exist ('did you mean ...?')
          none    nothing similar"""
        res = dict(status="none", answer=None, alternatives=[], suggestions=[], score=0.0, matched=None)
        qn = _norm(query)
        if not qn:
            return res

        def answered(status, its, score):
            return dict(status=status, answer=its[0]["tgt"], alternatives=[x["tgt"] for x in its[1:4]], suggestions=[],
                        score=round(float(score), 3), matched=its[0]["src"])

        ex = self._exact.get(("translation", lang, tgt_lang, qn))
        if ex:
            return answered("exact", self._ranked(ex), 1.0)
        cands = self.search(query, 25, kind="translation", lang=lang, tgt_lang=tgt_lang)
        groups = {}
        for cos, it in cands:
            k = _norm(it["src"])
            if k and k not in groups:
                groups[k] = [cos, None, None]
        if not groups:
            return res
        for k, g in list(groups.items())[:8]:                       # edit distance only for the best few (cheap, and enough)
            d = _lev(qn, k); g[1] = d; g[2] = 1 - d / max(len(qn), len(k))
        scored = sorted(((k, g) for k, g in groups.items() if g[1] is not None), key=lambda kg: -kg[1][2])
        nwords = len(qn.split())
        k0, (cos0, d0, es0) = scored[0]
        if nwords >= 3:
            if cos0 >= accept and es0 >= 0.85:
                return answered("near", self._ranked(self._exact[("translation", lang, tgt_lang, k0)]), es0)
        elif short_auto:
            allowed = 0 if len(qn) < 5 else (1 if len(qn) < 10 else 2)
            close = [kg for kg in scored if kg[1][1] <= allowed]
            if close and len(close) == 1:                             # one unambiguous neighbour only
                return answered("near", self._ranked(self._exact[("translation", lang, tgt_lang, close[0][0])]), close[0][1][2])
        sug = []
        for k, (cos, d, es) in scored:
            if es >= 0.6 or cos >= 0.6:
                it = self._ranked(self._exact[("translation", lang, tgt_lang, k)])[0]
                sug.append(dict(source=it["src"], translation=it["tgt"], similarity=round(max(es, 0.0), 2)))
            if len(sug) >= max_suggest:
                break
        if sug:
            res.update(status="suggest", suggestions=sug)
        return res

    def __len__(self):
        return len(self.items)
