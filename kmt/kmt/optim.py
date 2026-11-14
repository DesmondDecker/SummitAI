import numpy as np


class Adam:
    """AdamW: Adam with decoupled weight decay (on matrices only) and global-norm clipping."""
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.98), eps=1e-8, wd=0.01, clip=1.0):
        self.params, self.lr, (self.b1, self.b2) = list(params), lr, betas
        self.eps, self.wd, self.clip, self.t = eps, wd, clip, 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        grads = [p.grad if p.grad is not None else np.zeros_like(p.data) for p in self.params]
        norm = float(np.sqrt(sum((g ** 2).sum() for g in grads)))
        if not np.isfinite(norm):                     # NaN/inf gradients: skip the update instead of destroying the weights
            return norm
        scale = min(1.0, self.clip / (norm + 1e-12))
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g * scale
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            upd = (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)
            if p.data.ndim >= 2:
                upd = upd + self.wd * p.data
            p.data -= self.lr * upd
        return norm


class EMA:
    """Exponential moving average of weights: a free accuracy boost (smoother, better-generalising weights)."""
    def __init__(self, params, decay=0.995):
        self.decay, self.n = decay, 0
        self.shadow = {k: v.data.copy() for k, v in params.items()}

    def update(self, params):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in params.items():
            self.shadow[k] *= d; self.shadow[k] += (1 - d) * v.data

    def swap(self, params):
        """Exchange live and averaged weights. Call twice to restore."""
        for k, v in params.items():
            tmp = v.data.copy(); v.data[...] = self.shadow[k]; self.shadow[k] = tmp
