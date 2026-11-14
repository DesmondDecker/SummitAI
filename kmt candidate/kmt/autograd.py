"""Reverse-mode autograd on NumPy arrays: the mathematical core of KMT."""
import numpy as np

_GRAD_ENABLED = True
_DTYPE = np.float64


def set_dtype(dt):
    """float32 is ~2x faster and half the memory; float64 is used for gradient checks."""
    global _DTYPE
    _DTYPE = np.dtype(dt).type


def get_dtype():
    return _DTYPE


class no_grad:
    """Inside this context no graph is built (fast inference)."""
    def __enter__(self):
        global _GRAD_ENABLED
        self.prev, _GRAD_ENABLED = _GRAD_ENABLED, False

    def __exit__(self, *a):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self.prev


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def _accum(t, g):
    if t.requires_grad:
        g = g.astype(t.data.dtype, copy=False)
        t.grad = g.copy() if t.grad is None else t.grad + g


def _wrap(x):
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    def __init__(self, data, parents=(), requires_grad=False):
        self.data = np.asarray(data, dtype=_DTYPE)
        if not _GRAD_ENABLED:
            parents, requires_grad = (), False
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)
        self.grad = None
        self._parents = parents
        self._backward = None

    shape = property(lambda self: self.data.shape)

    # ---- arithmetic ----------------------------------------------------
    def __add__(self, o):
        o = _wrap(o); out = Tensor(self.data + o.data, (self, o))
        def bw():
            _accum(self, _unbroadcast(out.grad, self.shape))
            _accum(o, _unbroadcast(out.grad, o.shape))
        out._backward = bw; return out
    __radd__ = __add__

    def __mul__(self, o):
        o = _wrap(o); out = Tensor(self.data * o.data, (self, o))
        def bw():
            _accum(self, _unbroadcast(out.grad * o.data, self.shape))
            _accum(o, _unbroadcast(out.grad * self.data, o.shape))
        out._backward = bw; return out
    __rmul__ = __mul__

    def __neg__(self): return self * -1.0
    def __sub__(self, o): return self + (-_wrap(o))
    def __rsub__(self, o): return (-self) + o
    def __truediv__(self, k): return self * (1.0 / k)      # scalar divisor

    def matmul(self, o):
        """Batched matrix product (leading dims broadcast, e.g. (B,T,C)@(C,D))."""
        o = _wrap(o); out = Tensor(self.data @ o.data, (self, o))
        def bw():
            _accum(self, _unbroadcast(out.grad @ np.swapaxes(o.data, -1, -2), self.shape))
            _accum(o, _unbroadcast(np.swapaxes(self.data, -1, -2) @ out.grad, o.shape))
        out._backward = bw; return out
    __matmul__ = matmul

    # ---- shape ops ------------------------------------------------------
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(shape), (self,))
        def bw(): _accum(self, out.grad.reshape(self.shape))
        out._backward = bw; return out

    def transpose(self, *axes):
        out = Tensor(np.transpose(self.data, axes), (self,))
        inv = np.argsort(axes)
        def bw(): _accum(self, np.transpose(out.grad, inv))
        out._backward = bw; return out

    # ---- nonlinearities / reductions -----------------------------------
    def tanh(self):
        t = np.tanh(self.data); out = Tensor(t, (self,))
        def bw(): _accum(self, out.grad * (1 - t * t))
        out._backward = bw; return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data)); out = Tensor(s, (self,))
        def bw(): _accum(self, out.grad * s * (1 - s))
        out._backward = bw; return out

    def relu(self):
        m = self.data > 0; out = Tensor(self.data * m, (self,))
        def bw(): _accum(self, out.grad * m)
        out._backward = bw; return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,))
        def bw(): _accum(self, np.ones_like(self.data) * out.grad)
        out._backward = bw; return out

    # ---- backward --------------------------------------------------------
    def backward(self):
        topo, seen, stack = [], set(), [(self, False)]
        while stack:                                   # iterative DFS
            node, done = stack.pop()
            if done:
                topo.append(node); continue
            if id(node) in seen:
                continue
            seen.add(id(node)); stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward()
        for node in topo:                  # free the graph now: each node's backward closure refers to the node itself (a reference cycle),
            node._backward = None          # which Python would otherwise clear only at an occasional full garbage collection,
            node._parents = ()             # letting memory grow ~10 MB per step on large runs


# ---------------------------------------------------------------- functions
def embed(W, idx):
    """Row lookup W[idx] (idx any int shape); gradient scattered to used rows."""
    idx = np.asarray(idx)
    out = Tensor(W.data[idx], (W,))
    def bw():
        flat = idx.reshape(-1); gr = out.grad.reshape(-1, W.data.shape[1])
        order = np.argsort(flat, kind="stable"); fs = flat[order]
        starts = np.flatnonzero(np.r_[True, fs[1:] != fs[:-1]])
        g = np.zeros_like(W.data)
        g[fs[starts]] = np.add.reduceat(gr[order], starts, axis=0)      # sums duplicates, fast on any NumPy
        _accum(W, g)
    out._backward = bw; return out


def gather_rows(x, idx):
    """x (N,d) -> x[idx] for UNIQUE row indices; backward scatters into a zero array."""
    out = Tensor(x.data[idx], (x,))
    def bw():
        g = np.zeros_like(x.data); g[idx] = out.grad
        _accum(x, g)
    out._backward = bw; return out


def softmax(x, mask=None, scale=1.0):
    """softmax(x*scale + mask) over the last axis, fused (one pass instead of scale, add, softmax)."""
    z = x.data * scale if scale != 1.0 else x.data
    if mask is not None:
        z = z + mask.astype(x.data.dtype, copy=False)
    e = np.exp(z - z.max(-1, keepdims=True)); s = e / e.sum(-1, keepdims=True)
    out = Tensor(s, (x,))
    def bw():
        g = out.grad
        _accum(x, scale * s * (g - (g * s).sum(-1, keepdims=True)))
    out._backward = bw; return out


def layer_norm(x, g, b, eps=1e-5):
    mu = x.data.mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(x.data.var(-1, keepdims=True) + eps)
    xh = (x.data - mu) * inv
    out = Tensor(xh * g.data + b.data, (x, g, b))
    def bw():
        dy = out.grad; dxh = dy * g.data; N = x.data.shape[-1]
        dx = inv / N * (N * dxh - dxh.sum(-1, keepdims=True) - xh * (dxh * xh).sum(-1, keepdims=True))
        _accum(x, dx)
        _accum(g, _unbroadcast(dy * xh, g.shape))
        _accum(b, _unbroadcast(dy, b.shape))
    out._backward = bw; return out


_GELU_K = float(np.sqrt(2 / np.pi))     # python float on purpose: an np.float64 scalar would upcast float32 arrays


def gelu(x):
    k = _GELU_K; v = x.data
    t = np.tanh(k * (v + 0.044715 * v ** 3))
    out = Tensor(0.5 * v * (1 + t), (x,))
    def bw():
        d = 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * k * (1 + 3 * 0.044715 * v ** 2)
        _accum(x, out.grad * d)
    out._backward = bw; return out


def dropout(x, p, rng):
    if p <= 0:
        return x
    return x * ((rng.random(x.shape) >= p) / (1 - p)).astype(x.data.dtype)


def cross_entropy(logits, targets, weights=None, smoothing=0.0):
    """Weighted mean softmax cross-entropy with optional label smoothing.
    logits (N,V), targets (N,), weights (N,)."""
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    N, V = p.shape; rows = np.arange(N)
    w = np.ones(N) if weights is None else np.asarray(weights, dtype=np.float64)
    denom = max(w.sum(), 1e-9)
    logp = np.log(p + 1e-12)
    per = (1 - smoothing) * -logp[rows, targets] + smoothing * -logp.mean(axis=1)
    out = Tensor((per * w).sum() / denom, (logits,))
    def bw():
        g = p.astype(np.float64) - smoothing / V
        g[rows, targets] -= (1 - smoothing)
        _accum(logits, g * (w / denom)[:, None] * out.grad)
    out._backward = bw; return out
