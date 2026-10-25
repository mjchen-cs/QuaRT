"""Learning agents: trust-masked, Lagrangian-constrained PPO (QuaRT) and a
DQN baseline. Pure NumPy."""
import numpy as np
from .nn import MLP, Adam, softmax, log_softmax
from .env import N_TGT, N_POW, N_BW, OBS_DIM

HEADS = (N_TGT, N_POW, N_BW)
NEG = -1e9


class PPOAgent:
    def __init__(self, seed=0, hidden=(128, 128), lr_actor=3e-4, lr_critic=1e-3, gamma=0.9,
                 lam=0.95, clip=0.2, epochs=6, minibatch=256, ent_coef=0.01, vf_coef=0.5):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.actor = MLP([OBS_DIM, *hidden, sum(HEADS)], rng, out_scale=0.01)
        self.critic = MLP([OBS_DIM, *hidden, 1], rng, out_scale=1.0)
        self.opt_a = Adam(self.actor.params, lr=lr_actor, max_grad_norm=0.5)
        self.opt_c = Adam(self.critic.params, lr=lr_critic, max_grad_norm=0.5)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.mb, self.ent, self.vf = epochs, minibatch, ent_coef, vf_coef
        self.reset_buffer()

    def reset_buffer(self):
        self.buf = dict(obs=[], mask=[], act=[], logp=[], val=[], rew=[], hub=[], t=[])

    def _dist(self, obs, mask):
        z = self.actor.forward(obs)
        zs = np.split(z, np.cumsum(HEADS)[:-1], axis=1)
        zs[0] = np.where(mask, zs[0], NEG)
        return zs

    def act(self, obs, mask, greedy=False):
        zs = self._dist(obs, mask)
        acts, logp = [], np.zeros(len(obs))
        for z in zs:
            p = softmax(z)
            if greedy:
                a = p.argmax(1)
            else:
                a = (p.cumsum(1) > self.rng.random((len(p), 1))).argmax(1)
            logp += np.log(p[np.arange(len(p)), a] + 1e-12)
            acts.append(a)
        v = self.critic.forward(obs)[:, 0]
        return np.stack(acts, 1), logp, v

    def store(self, obs, mask, act, logp, val, rew, hubs, t):
        b = self.buf
        b['obs'].append(obs); b['mask'].append(mask); b['act'].append(act)
        b['logp'].append(logp); b['val'].append(val); b['rew'].append(rew)
        b['hub'].append(hubs); b['t'].append(np.full(len(hubs), t))

    def update(self):
        b = {k: np.concatenate(v) for k, v in self.buf.items() if len(v)}
        if not b:
            return {}
        n = len(b['rew'])
        # GAE along each hub's own decision sequence
        adv = np.zeros(n); ret = np.zeros(n)
        order = np.lexsort((b['t'], b['hub']))
        hubs = b['hub'][order]
        last_gae = 0.0
        for k in range(n - 1, -1, -1):
            j = order[k]
            if k == n - 1 or hubs[k + 1] != hubs[k]:
                next_v, last_gae = b['val'][j], 0.0   # truncated episode: bootstrap with V(s)
            else:
                next_v = b['val'][order[k + 1]]
            delta = b['rew'][j] + self.gamma * next_v - b['val'][j]
            last_gae = delta + self.gamma * self.lam * last_gae
            adv[j] = last_gae; ret[j] = last_gae + b['val'][j]
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        stats = dict(pl=0.0, vl=0.0, ent=0.0, cf=0.0)
        cnt = 0
        for _ in range(self.epochs):
            idx = self.rng.permutation(n)
            for s in range(0, n, self.mb):
                j = idx[s:s + self.mb]
                obs, mask, act = b['obs'][j], b['mask'][j], b['act'][j]
                zs = self._dist(obs, mask)
                logp = np.zeros(len(j)); dz = []; ent_tot = 0.0
                probs = []
                for h, z in enumerate(zs):
                    lp = log_softmax(z); p = np.exp(lp)
                    logp += lp[np.arange(len(j)), act[:, h]]
                    probs.append((p, lp))
                ratio = np.exp(logp - b['logp'][j])
                A = adv_n[j]
                unclipped = ratio * A
                clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip) * A
                active = unclipped <= clipped          # min picks unclipped
                dlogp = -np.where(active, ratio * A, 0.0) / len(j)
                for h, (p, lp) in enumerate(probs):
                    oh = np.zeros_like(p); oh[np.arange(len(j)), act[:, h]] = 1
                    g = dlogp[:, None] * (oh - p)
                    lp_safe = np.where(p > 0, lp, 0.0)
                    H = -(p * lp_safe).sum(1, keepdims=True)
                    dH = -p * (lp_safe + H)
                    g -= self.ent * dH / len(j)
                    dz.append(g); ent_tot += H.mean()
                self.opt_a.step(self.actor.backward(np.concatenate(dz, 1)))
                v = self.critic.forward(obs)[:, 0]
                dv = self.vf * 2 * (v - ret[j]) / len(j)
                self.opt_c.step(self.critic.backward(dv[:, None]))
                stats['pl'] += float(-np.minimum(unclipped, clipped).mean())
                stats['vl'] += float(((v - ret[j]) ** 2).mean())
                stats['ent'] += float(ent_tot)
                stats['cf'] += float((~active).mean()); cnt += 1
        self.reset_buffer()
        return {k: v / max(cnt, 1) for k, v in stats.items()}

    def state(self):
        return dict(actor=self.actor.state(), critic=self.critic.state())

    def load(self, s):
        self.actor.load(s['actor']); self.critic.load(s['critic'])


class DQNAgent:
    def __init__(self, seed=0, hidden=(128, 128), lr=5e-4, gamma=0.9, buffer=60000, batch=256,
                 target_every=500, eps_start=1.0, eps_end=0.05, eps_decay_steps=150000):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.nA = N_TGT * N_POW * N_BW
        self.q = MLP([OBS_DIM, *hidden, self.nA], rng, out_scale=0.1)
        self.qt = MLP([OBS_DIM, *hidden, self.nA], rng, out_scale=0.1)
        self.qt.copy_from(self.q)
        self.opt = Adam(self.q.params, lr=lr, max_grad_norm=10.0)
        self.gamma, self.batch, self.target_every = gamma, batch, target_every
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_steps
        self.steps = 0
        self.cap = buffer
        self.B = dict(o=np.zeros((buffer, OBS_DIM), np.float32), a=np.zeros(buffer, int),
                      r=np.zeros(buffer), o2=np.zeros((buffer, OBS_DIM), np.float32), d=np.zeros(buffer))
        self.ptr = 0; self.size = 0
        self.pending = {}         # hub -> (obs, action, reward) awaiting next obs

    @property
    def eps(self):
        f = min(1.0, self.steps / self.eps_decay)
        return self.eps_start + f * (self.eps_end - self.eps_start)

    def act(self, obs, mask, greedy=False):
        qv = self.q.forward(obs)
        a = qv.argmax(1)
        if not greedy:
            rnd = self.rng.random(len(obs)) < self.eps
            a[rnd] = self.rng.integers(0, self.nA, rnd.sum())
        self.steps += len(obs)
        return np.stack([a // (N_POW * N_BW), (a // N_BW) % N_POW, a % N_BW], 1), a

    def _push(self, o, a, r, o2, d):
        i = self.ptr
        self.B['o'][i], self.B['a'][i], self.B['r'][i], self.B['o2'][i], self.B['d'][i] = o, a, r, o2, d
        self.ptr = (self.ptr + 1) % self.cap; self.size = min(self.size + 1, self.cap)

    def observe(self, hubs, obs, a_flat, rew):
        for h, o, a, r in zip(hubs, obs, a_flat, rew):
            if h in self.pending:
                po, pa, pr = self.pending[h]
                self._push(po, pa, pr, o, 0.0)
            self.pending[h] = (o, a, r)

    def end_episode(self):
        for h, (po, pa, pr) in self.pending.items():
            self._push(po, pa, pr, po, 1.0)
        self.pending = {}

    def train_step(self, n_updates=1):
        if self.size < 2000:
            return
        for _ in range(n_updates):
            j = self.rng.integers(0, self.size, self.batch)
            o, a, r, o2, d = (self.B[k][j] for k in ('o', 'a', 'r', 'o2', 'd'))
            # double DQN target
            a2 = self.q.forward(o2).argmax(1)
            qt = self.qt.forward(o2)[np.arange(len(j)), a2]
            y = r + self.gamma * (1 - d) * qt
            qv = self.q.forward(o)
            err = qv[np.arange(len(j)), a] - y
            g = np.zeros_like(qv)
            g[np.arange(len(j)), a] = np.clip(err, -1, 1) / len(j)
            self.opt.step(self.q.backward(g))
            self.n_upd = getattr(self, 'n_upd', 0) + 1
            if self.n_upd % self.target_every == 0:
                self.qt.copy_from(self.q)

    def state(self):
        return dict(q=self.q.state())

    def load(self, s):
        self.q.load(s['q'])
