"""Training / evaluation loops shared by all methods."""
import copy
import numpy as np
from .config import Config
from .env import HealthEnv, METHODS
from .agents import PPOAgent, DQNAgent

KEYS = ['n', 'ok', 'okw', 'w', 'emerg', 'emerg_viol', 'crit', 'crit_miss', 'leak', 'wrong',
        'energy', 'bits_ok', 'gated_honest', 'forged', 'cmp_n', 'cmp_ok', 'all_w', 'all_okw']


def _episode(env, agent, train=True, greedy=False):
    cfg = env.cfg
    acc = {k: 0.0 for k in KEYS}
    acc['prio_n'] = np.zeros(3); acc['prio_ok'] = np.zeros(3); acc['tgt'] = np.zeros(5)
    att, att_type, rep_h, rep_m, rew_sum = [], [], [], [], 0.0
    kind = METHODS[env.method][0]
    for t in range(cfg.slots):
        reqs = env.begin_slot()
        if reqs:
            obs = np.stack([r['obs'] for r in reqs])
            mask = np.stack([r['mask'] for r in reqs])
            hubs = np.array([r['i'] for r in reqs])
            if kind == 'ppo':
                act, logp, val = agent.act(obs, mask, greedy=greedy)
            elif kind == 'dqn':
                act, aflat = agent.act(obs, mask, greedy=greedy)
            else:
                act = np.stack([env.rule_action(r) for r in reqs])
        else:
            act = np.zeros((0, 3), int)
        rew, st = env.end_slot(reqs, act)
        if reqs and train:
            if kind == 'ppo':
                agent.store(obs, mask, act, logp, val, rew, hubs, t)
            elif kind == 'dqn':
                agent.observe(hubs, obs, aflat, rew)
                agent.train_step(2)
        rew_sum += rew.sum()
        for k in KEYS:
            acc[k] += st[k]
        acc['prio_n'] += st['prio_n']; acc['prio_ok'] += st['prio_ok']; acc['tgt'] += st['tgt']
        att.append(st['attack']); att_type.append(st['att_type']); rep_h.append(st['rep_h']); rep_m.append(st['rep_m'])
    upd = {}
    if train and kind == 'ppo':
        upd = agent.update()
    if train and kind == 'dqn':
        agent.end_episode()
    # Lagrange multiplier (dual ascent on the URLLC constraint)
    viol = acc['emerg_viol'] / max(acc['emerg'], 1)
    if env.lagr and train:
        env.lam = float(np.clip(env.lam + 1.0 * (viol - cfg.delta_emerg), 0.0, 10.0))
    n = max(acc['n'], 1)
    rec = dict(
        attack=float(np.mean(att)), att_type=np.mean(att_type, 0).tolist(),
        rep_h=float(np.mean(rep_h)), rep_m=float(np.mean(rep_m)),
        success=acc['ok'] / n, wsuccess=acc['okw'] / max(acc['w'], 1),
        all_wsuccess=acc['all_okw'] / max(acc['all_w'], 1), cmp_service=acc['cmp_ok'] / max(acc['cmp_n'], 1),
        emerg_viol=viol, crit_miss=acc['crit_miss'] / max(acc['crit'], 1),
        leak_per_1k=1000 * acc['leak'] / n, wrong_per_1k=1000 * acc['wrong'] / n,
        energy_mj=1000 * acc['energy'] / n,
        goodput_mbps=acc['bits_ok'] / (cfg.slots * cfg.slot_s) / 1e6,
        prio_ontime=(acc['prio_ok'] / np.maximum(acc['prio_n'], 1)).tolist(),
        tgt_frac=(acc['tgt'] / n).tolist(), gated_honest=acc['gated_honest'] / n,
        forged=acc['forged'], reward=rew_sum / n, lam=env.lam, **{f'upd_{k}': v for k, v in upd.items()})
    return rec


def make_agent(method, seed):
    kind = METHODS[method][0]
    if kind == 'ppo':
        return PPOAgent(seed=seed)
    if kind == 'dqn':
        return DQNAgent(seed=seed)
    return None


def train(method, cfg: Config, pool, seed=0, episodes=300, log_every=0):
    env = HealthEnv(cfg, method, pool, seed=seed)
    agent = make_agent(method, seed + 1000)
    hist = []
    for ep in range(episodes):
        rec = _episode(env, agent, train=True)
        hist.append(rec)
        if log_every and (ep + 1) % log_every == 0:
            print(f'[{method} s{seed}] ep {ep+1}: attack={rec["attack"]:.3f} wsucc={rec["wsuccess"]:.3f} '
                  f'crit_miss={rec["crit_miss"]:.3f} emerg_viol={rec["emerg_viol"]:.3f} '
                  f'repH={rec["rep_h"]:.2f} repM={rec["rep_m"]:.2f} lam={env.lam:.2f}', flush=True)
    return env, agent, hist


def evaluate(env, agent, cfg_eval: Config, episodes=20, adapt_attackers=True):
    """Freeze the defender, change the operating point, let attackers keep
    adapting, and report the average over the evaluation episodes."""
    env = copy.deepcopy(env)
    env.cfg = cfg_eval
    recs = [_episode(env, agent, train=False, greedy=False) for _ in range(episodes)]
    return recs
