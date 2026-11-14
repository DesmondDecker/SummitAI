"""Decoder-only transformer (GPT-style) built on kmt.autograd."""
import json, os
import numpy as np
from .autograd import (Tensor, embed, gather_rows, softmax, layer_norm, gelu, dropout, cross_entropy,
                       get_dtype, set_dtype)


_GK = float(np.sqrt(2 / np.pi))          # python float: np.float64 scalars upcast float32 arrays in NumPy 2


def atomic_savez(path, **arrays):
    """Write to a temp file then rename, so a crash or Ctrl-C mid-save can never corrupt the checkpoint."""
    tmp = (path[:-4] if path.endswith(".npz") else path) + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(_GK * (x + 0.044715 * x ** 3)))


def _log_softmax(z):
    z = z.astype(np.float64); z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


class Transformer:
    def __init__(self, vocab_size, d=128, n_layer=3, n_head=4, ctx=128, dropout=0.1, seed=0):
        assert d % n_head == 0
        self.cfg = dict(vocab_size=vocab_size, d=d, n_layer=n_layer, n_head=n_head, ctx=ctx, dropout=dropout)
        rng = np.random.default_rng(seed)
        N = lambda *s, sc=0.02: Tensor(rng.normal(0, sc, s), requires_grad=True)
        Z = lambda *s: Tensor(np.zeros(s), requires_grad=True)
        O = lambda *s: Tensor(np.ones(s), requires_grad=True)
        rs = 0.02 / np.sqrt(2 * n_layer)                    # scaled init on residual projections
        P = {"tok": N(vocab_size, d), "pos": N(ctx, d), "lnf.g": O(d), "lnf.b": Z(d)}
        for l in range(n_layer):
            k = f"h{l}."
            P[k + "ln1.g"], P[k + "ln1.b"] = O(d), Z(d)
            for n in "qkv":
                P[k + "W" + n], P[k + "b" + n] = N(d, d), Z(d)
            P[k + "Wo"], P[k + "bo"] = N(d, d, sc=rs), Z(d)
            P[k + "ln2.g"], P[k + "ln2.b"] = O(d), Z(d)
            P[k + "W1"], P[k + "b1"] = N(d, 4 * d), Z(4 * d)
            P[k + "W2"], P[k + "b2"] = N(4 * d, d, sc=rs), Z(d)
        self.params = P
        self.mask = np.triu(np.full((ctx, ctx), -1e9), 1)   # causal mask
        self.rng = np.random.default_rng(seed + 1)

    def parameters(self):
        return list(self.params.values())

    def n_params(self):
        return sum(p.data.size for p in self.params.values())

    def _attn(self, x, k, B, T, train):
        p, c = self.params, self.cfg
        nh, hd = c["n_head"], c["d"] // c["n_head"]
        heads = lambda n: (x @ p[k + "W" + n] + p[k + "b" + n]).reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        q, kk, v = heads("q"), heads("k"), heads("v")
        att = softmax(q @ kk.transpose(0, 1, 3, 2), self.mask[:T, :T], 1.0 / float(np.sqrt(hd)))
        if train:
            att = dropout(att, c["dropout"], self.rng)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, c["d"])
        return y @ p[k + "Wo"] + p[k + "bo"]

    def forward(self, idx, train=False):
        """idx: int array (B,T) -> logits Tensor (B,T,V)."""
        return self.hidden(idx, train) @ self.params["tok"].transpose(1, 0)     # tied input/output embeddings

    def hidden(self, idx, train=False):
        """idx: int array (B,T) -> final hidden states Tensor (B,T,d)."""
        B, T = idx.shape
        p, c = self.params, self.cfg
        assert T <= c["ctx"], "sequence longer than context window"
        x = embed(p["tok"], idx) + embed(p["pos"], np.arange(T))
        if train:
            x = dropout(x, c["dropout"], self.rng)
        for l in range(c["n_layer"]):
            k = f"h{l}."
            a = self._attn(layer_norm(x, p[k + "ln1.g"], p[k + "ln1.b"]), k, B, T, train)
            x = x + (dropout(a, c["dropout"], self.rng) if train else a)
            h = layer_norm(x, p[k + "ln2.g"], p[k + "ln2.b"])
            m = gelu(h @ p[k + "W1"] + p[k + "b1"]) @ p[k + "W2"] + p[k + "b2"]
            x = x + (dropout(m, c["dropout"], self.rng) if train else m)
        return layer_norm(x, p["lnf.g"], p["lnf.b"])

    def loss(self, ids, mask, train=True, smoothing=0.0):
        """Next-token loss over positions where mask>0 (the target part). The output projection and
        softmax are computed ONLY at those positions, which saves a large share of the work."""
        h = self.hidden(ids[:, :-1], train)
        B, T, d = h.shape
        w = mask[:, 1:].reshape(-1)
        sel = np.flatnonzero(w > 0)
        if sel.size == 0:
            return h.sum() * 0.0
        logits = gather_rows(h.reshape(B * T, d), sel) @ self.params["tok"].transpose(1, 0)
        return cross_entropy(logits, ids[:, 1:].reshape(-1)[sel], w[sel], smoothing if train else 0.0)

    # ------------------------------------------------------------ inference
    def infer(self, ids, cache=None, all_pos=False):
        """Fast NumPy forward pass with a key/value cache (no autograd graph).
        ids: (B,t) new tokens. cache: from a previous call (already holds the earlier tokens).
        Returns logits (B,1,V) for the last position (or (B,t,V) if all_pos) and the new cache."""
        c = self.cfg; d, nh = c["d"], c["n_head"]; hd = d // nh
        P = {k: v.data for k, v in self.params.items()}
        ids = np.asarray(ids); B, t = ids.shape
        start = cache["len"] if cache else 0
        assert start + t <= c["ctx"], "sequence longer than context window"
        x = P["tok"][ids] + P["pos"][start:start + t]
        new = {"len": start + t, "k": [], "v": []}
        for l in range(c["n_layer"]):
            k = f"h{l}."
            h = _ln(x, P[k + "ln1.g"], P[k + "ln1.b"])
            sp = lambda n: (h @ P[k + "W" + n] + P[k + "b" + n]).reshape(B, t, nh, hd).transpose(0, 2, 1, 3)
            q, kk, vv = sp("q"), sp("k"), sp("v")
            if cache:
                kk = np.concatenate([cache["k"][l], kk], 2); vv = np.concatenate([cache["v"][l], vv], 2)
            new["k"].append(kk); new["v"].append(vv)
            att = q @ kk.transpose(0, 1, 3, 2) * (1.0 / float(np.sqrt(hd))) + self.mask[start:start + t, :start + t].astype(x.dtype)
            att = np.exp(att - att.max(-1, keepdims=True)); att /= att.sum(-1, keepdims=True)
            x = x + (att @ vv).transpose(0, 2, 1, 3).reshape(B, t, d) @ P[k + "Wo"] + P[k + "bo"]
            h = _ln(x, P[k + "ln2.g"], P[k + "ln2.b"])
            x = x + _gelu(h @ P[k + "W1"] + P[k + "b1"]) @ P[k + "W2"] + P[k + "b2"]
        x = _ln(x, P["lnf.g"], P["lnf.b"])
        if not all_pos:
            x = x[:, -1:]
        return x @ P["tok"].T, new

    def seq_logprob(self, ids, start):
        """Sum of log p(ids[:,t] | ids[:,:t]) for t >= start. ids: (B,T) ints -> (B,)."""
        logits, _ = self.infer(ids[:, :-1], all_pos=True)
        lp = _log_softmax(logits)
        tok_lp = np.take_along_axis(lp, ids[:, 1:, None], -1)[..., 0]
        return tok_lp[:, start - 1:].sum(1)

    def token_logprobs(self, ids):
        """Per-token log-probs of a single sequence: entry i is log p(ids[i+1] | ids[:i+1])."""
        ids = np.asarray(ids)[None]
        logits, _ = self.infer(ids[:, :-1], all_pos=True)
        return np.take_along_axis(_log_softmax(logits), ids[:, 1:, None], -1)[0, :, 0]

    def next_token_dist(self, prefix, ban=()):
        logits, _ = self.infer(np.array([prefix]))
        lp = _log_softmax(logits[0, 0]); lp[list(ban)] = -1e9
        return lp

    def generate(self, prefix, max_new, eos, ban=(), temperature=0.0, top_k=0, seed=None):
        """Greedy (temperature=0) or sampled decoding with a KV cache. Returns (ids, mean log-prob)."""
        rng = np.random.default_rng(seed)
        out, lps = [], []
        max_new = min(max_new, self.cfg["ctx"] - len(prefix))
        logits, cache = self.infer(np.array([prefix]))
        for _ in range(max_new):
            z = logits[0, 0].astype(np.float64).copy(); z[list(ban)] = -1e9
            lp = z - z.max(); lp = lp - np.log(np.exp(lp).sum())
            if temperature <= 0:
                nxt = int(np.argmax(z))
            else:
                sc = z / temperature
                if top_k:
                    sc[sc < np.sort(sc)[-top_k]] = -1e9
                pr = np.exp(sc - sc.max()); pr /= pr.sum()
                nxt = int(rng.choice(len(pr), p=pr))
            out.append(nxt); lps.append(lp[nxt])
            if nxt == eos:
                break
            logits, cache = self.infer(np.array([[nxt]]), cache)
        return out, (float(np.mean(lps)) if lps else 0.0)

    def beam_search(self, prefix, max_new, eos, beam=4, ban=(), alpha=0.6):
        """Keeps the `beam` best partial answers instead of committing greedily: better translations.
        Returns (ids, mean log-prob)."""
        ban = list(ban); max_new = min(max_new, self.cfg["ctx"] - len(prefix))
        norm = lambda sc, n: sc / (((5 + n) / 6) ** alpha)
        logits, cache = self.infer(np.array([prefix]))
        lp = _log_softmax(logits[:, 0]); lp[:, ban] = -1e9
        seqs, scores, finished = [[]], np.zeros(1), []
        for _ in range(max_new):
            V = lp.shape[1]; cand = (scores[:, None] + lp).reshape(-1)
            top = np.argpartition(-cand, min(2 * beam, cand.size) - 1)[:2 * beam]
            top = top[np.argsort(-cand[top])]
            keep = []
            for j in top:
                i, tk = divmod(int(j), V); sc = float(cand[j])
                if sc < -1e8:
                    continue
                if tk == eos:
                    finished.append((seqs[i] + [tk], sc))
                else:
                    keep.append((i, tk, sc))
                if len(keep) == beam:
                    break
            if not keep:
                break
            if len(finished) >= beam:
                if max(norm(sc, len(q)) for q, sc in finished) >= max(norm(sc, len(seqs[i]) + 1) for i, _, sc in keep):
                    break
            idx = [i for i, _, _ in keep]
            seqs = [seqs[i] + [tk] for i, tk, _ in keep]; scores = np.array([sc for _, _, sc in keep])
            cache = {"len": cache["len"], "k": [a[idx] for a in cache["k"]], "v": [a[idx] for a in cache["v"]]}
            logits, cache = self.infer(np.array([[tk] for _, tk, _ in keep]), cache)
            lp = _log_softmax(logits[:, 0]); lp[:, ban] = -1e9
        if not finished:
            finished = [(seqs[i], float(scores[i])) for i in range(len(seqs))]
        best = max(finished, key=lambda x: norm(x[1], len(x[0])))
        return best[0], best[1] / max(1, len(best[0]))

    # ---------------------------------------------------------- persistence
    def save(self, path, extra=None):
        meta = json.dumps({"cfg": self.cfg, "extra": extra or {}, "dtype": str(np.dtype(get_dtype()))})
        atomic_savez(path, meta=meta, **{k: v.data for k, v in self.params.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        set_dtype(meta.get("dtype", "float64"))
        m = cls(**meta["cfg"])
        for k, v in m.params.items():
            v.data[...] = z[k]
        return m, meta["extra"]
