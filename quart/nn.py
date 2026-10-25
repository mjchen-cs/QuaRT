"""Minimal NumPy neural-network toolkit (MLP + Adam) used by all learning agents.

Implemented from scratch so that the whole code base runs on any CPU with only
NumPy/SciPy/Matplotlib installed (no deep-learning framework required).
"""
import numpy as np


class Adam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, max_grad_norm=None):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.max_grad_norm = max_grad_norm

    def step(self, grads):
        if self.max_grad_norm is not None:
            gn = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
            if gn > self.max_grad_norm:
                grads = [g * (self.max_grad_norm / (gn + 1e-12)) for g in grads]
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class MLP:
    """Fully connected network with tanh hidden activations and linear output."""

    def __init__(self, sizes, rng, out_scale=1.0):
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            scale = np.sqrt(2.0 / fan_in) if i < len(sizes) - 2 else out_scale / np.sqrt(fan_in)
            self.W.append(rng.normal(0, scale, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    @property
    def params(self):
        return self.W + self.b

    def forward(self, x):
        self.cache = [x]
        h = x
        for i in range(len(self.W)):
            h = h @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                h = np.tanh(h)
            self.cache.append(h)
        return h

    def backward(self, dout):
        """Back-propagate dL/d(out); returns grads aligned with self.params."""
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        d = dout
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                d = d * (1 - self.cache[i + 1] ** 2)
            gW[i] = self.cache[i].T @ d
            gb[i] = d.sum(0)
            d = d @ self.W[i].T
        return gW + gb

    def copy_from(self, other):
        for a, b in zip(self.params, other.params):
            a[...] = b

    def state(self):
        return [p.copy() for p in self.params]

    def load(self, state):
        for p, s in zip(self.params, state):
            p[...] = s


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))
