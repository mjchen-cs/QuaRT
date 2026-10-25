"""CV-QKD trust-backbone experiments (key rate vs. distance, registry key budget).

Writes results/qkd.json."""
import json, os
import numpy as np
from quart.cvqkd import optimise_rate, self_test

OUT = 'results'


def key_budget(K_bits_per_pulse, symbol_rate=100e6, f_upd=10.0, entry_bits=16, tag_key_bits=128):
    """Max number of reputation entries that can be one-time-padded and
    Wegman-Carter authenticated at refresh frequency f_upd (Hz)."""
    if K_bits_per_pulse <= 0:
        return 0.0
    bps = K_bits_per_pulse * symbol_rate
    return max(0.0, (bps / f_upd - tag_key_bits) / entry_bits)


def main():
    os.makedirs(OUT, exist_ok=True)
    self_test()
    dists = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    res = dict(dist=dists)
    for eps in (0.01, 0.02):
        r0, r1, a0, a1, e1 = [], [], [], [], []
        for L in dists:
            b0 = optimise_rate(L, eps, alphas=np.linspace(0.1, 1.0, 19))
            b1 = optimise_rate(L, eps, qs=True, alphas=np.linspace(0.05, 0.6, 12),
                               etas=np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]))
            r0.append(float(b0['K'])); r1.append(float(b1['K']))
            a0.append(float(b0['alpha'])); a1.append(float(b1['alpha'])); e1.append(float(b1['eta']))
            print(f'eps={eps} L={L:4d} km  K_noQS={b0["K"]:.3e}  K_QS={b1["K"]:.3e} (eta={b1["eta"]}, P={b1["P"]:.3f})', flush=True)
        res[f'K_noqs_{eps}'] = r0; res[f'K_qs_{eps}'] = r1
        res[f'alpha_noqs_{eps}'] = a0; res[f'alpha_qs_{eps}'] = a1; res[f'eta_qs_{eps}'] = e1
        best = [max(x, y, 0.0) for x, y in zip(r0, r1)]
        res[f'K_best_{eps}'] = best
        res[f'Nmax_{eps}'] = [key_budget(k) for k in best]
    # PLOB bound for reference
    T = 10 ** (-0.02 * np.array(dists))
    res['plob'] = (-np.log2(1 - T)).tolist()
    json.dump(res, open(f'{OUT}/qkd.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
