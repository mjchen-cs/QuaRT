"""Run all network-level experiments of the paper.

Usage examples
  python run_experiments.py main   --n_hubs 40 --seeds 0 1 2
  python run_experiments.py main   --n_hubs 80 --seeds 0 1 2
  python run_experiments.py sweep  --n_hubs 40          (needs the 'main' runs)
  python run_experiments.py qclass                     (classical-forgery sweep)
  python run_experiments.py malfrac                    (compromised-node fraction)
All results are written to results/*.json (and trained models to results/models).
"""
import argparse, json, os, pickle, time, copy
from multiprocessing import Pool
import numpy as np
from quart.config import Config
from quart import runner as R

MAIN_METHODS = ['quart', 'quart_c', 'quart_u', 'quart_nomask', 'quart_nolag',
                'dqn_r', 'ppo_nr', 'brs', 'drp', 'greedy']
CMP_METHODS = ['quart', 'dqn_r', 'ppo_nr', 'brs', 'drp', 'greedy']
OUT = 'results'


def load_pool(path=f'{OUT}/ecg_pool.npz'):
    d = np.load(path)
    return {k: d[k] for k in ['y', 'pred', 'pid', 'test', 'flag']}


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def eval_greedy(env, agent, episodes=20):
    return [R._episode(env, agent, train=False, greedy=True) for _ in range(episodes)]


def job_main(args):
    method, n_hubs, seed, episodes, cfg_over, tag = args
    t0 = time.time()
    cfg = Config(n_hubs=n_hubs, **cfg_over)
    pool = load_pool()
    env, agent, hist = R.train(method, cfg, pool, seed=seed, episodes=episodes)
    ev = eval_greedy(copy.deepcopy(env), agent, 20)
    os.makedirs(f'{OUT}/models', exist_ok=True)
    with open(f'{OUT}/models/{tag}_{method}_N{n_hubs}_s{seed}.pkl', 'wb') as f:
        pickle.dump(dict(env=env, agent=agent), f)
    res = dict(method=method, n_hubs=n_hubs, seed=seed, cfg=cfg_over, hist=hist, eval=ev,
               time=time.time() - t0)
    with open(f'{OUT}/{tag}_{method}_N{n_hubs}_s{seed}.json', 'w') as f:
        json.dump(_jsonable(res), f)
    print(f'done {tag} {method} N={n_hubs} s={seed} ({res["time"]:.0f}s) '
          f'att={np.mean([r["attack"] for r in ev]):.3f} wsucc={np.mean([r["wsuccess"] for r in ev]):.3f}', flush=True)
    return tag


def job_sweep(args):
    method, n_hubs, seed, param, value = args
    with open(f'{OUT}/models/main_{method}_N{n_hubs}_s{seed}.pkl', 'rb') as f:
        d = pickle.load(f)
    env, agent = d['env'], d['agent']
    cfg = copy.deepcopy(env.cfg)
    setattr(cfg, param, value)
    env = copy.deepcopy(env); env.cfg = cfg
    # attackers keep adapting for 10 warm-up episodes, then 20 measured episodes
    for _ in range(10):
        R._episode(env, agent, train=False, greedy=True)
    ev = eval_greedy(env, agent, 20)
    return dict(method=method, n_hubs=n_hubs, seed=seed, param=param, value=value,
                **{k: float(np.mean([r[k] for r in ev])) for k in
                   ['wsuccess', 'all_wsuccess', 'goodput_mbps', 'emerg_viol', 'crit_miss', 'attack', 'energy_mj']})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['main', 'sweep', 'qclass', 'malfrac'])
    ap.add_argument('--n_hubs', type=int, default=40)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--methods', nargs='+', default=None)
    ap.add_argument('--procs', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(f'{OUT}/ecg_pool.npz'):
        from quart.ecg import build_task_pool
        m = build_task_pool('auto', out=f'{OUT}/ecg_pool.npz')
        json.dump(_jsonable(m), open(f'{OUT}/ecg_metrics.json', 'w'), indent=1)
    jobs = []
    if a.what == 'main':
        meths = a.methods or (MAIN_METHODS if a.n_hubs == 40 else CMP_METHODS)
        jobs = [(m, a.n_hubs, s, a.episodes, {}, 'main') for s in a.seeds for m in meths]
        with Pool(a.procs) as p:
            list(p.imap_unordered(job_main, jobs))
    elif a.what == 'qclass':
        for q in [0.0, 0.1, 0.3, 0.6, 0.9]:
            for s in a.seeds:
                jobs.append(('quart_c', a.n_hubs, s, a.episodes, {'q_classic': q}, f'qc{q}'))
        with Pool(a.procs) as p:
            list(p.imap_unordered(job_main, jobs))
    elif a.what == 'malfrac':
        meths = a.methods or ['quart', 'dqn_r', 'brs', 'drp', 'greedy']
        for mf in [0.1, 0.2, 0.3, 0.4]:
            for s in a.seeds:
                for m in meths:
                    jobs.append((m, a.n_hubs, s, a.episodes, {'mal_frac': mf}, f'mf{mf}'))
        with Pool(a.procs) as p:
            list(p.imap_unordered(job_main, jobs))
    elif a.what == 'sweep':
        meths = a.methods or CMP_METHODS
        grid = [('bw_mhz', v) for v in (10.0, 20.0, 30.0)] + \
               [('task_scale', v) for v in (0.5, 1.0, 1.5)] + \
               [('p_max_dbm', v) for v in (10.0, 15.0, 20.0)]
        jobs = [(m, a.n_hubs, s, p, v) for m in meths for s in a.seeds for (p, v) in grid]
        with Pool(a.procs) as p:
            out = list(p.imap_unordered(job_sweep, jobs))
        json.dump(out, open(f'{OUT}/sweep_N{a.n_hubs}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
