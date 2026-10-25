"""Generate every result figure (PDF) and LaTeX table of the paper from results/*.json."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = 'results'
FIG = 'paper_figures'
TAB = 'paper_tables'
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.5,
    'lines.linewidth': 1.4, 'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight',
})
# validated categorical palette, fixed order; secondary encoding by marker/linestyle
STYLE = {
    'quart':        dict(label='QuaRT (ours)', c='#2a78d6', m='o', ls='-'),
    'drp':          dict(label='DRP', c='#eb6834', m='s', ls='--'),
    'dqn_r':        dict(label='DQN-R', c='#1baf7a', m='^', ls='-.'),
    'brs':          dict(label='BRS', c='#eda100', m='D', ls=':'),
    'ppo_nr':       dict(label='PPO-NR', c='#e87ba4', m='v', ls='--'),
    'greedy':       dict(label='Greedy', c='#008300', m='x', ls=':'),
    'quart_c':      dict(label='QuaRT-C (classical)', c='#6250d6', m='P', ls='-.'),
    'quart_u':      dict(label='QuaRT-U', c='#e34948', m='*', ls='--'),
    'quart_nomask': dict(label='QuaRT w/o mask', c='#52514e', m='h', ls=':'),
    'quart_nolag':  dict(label='QuaRT w/o Lagrangian', c='#9a9890', m='<', ls='-.'),
}
CMP = ['quart', 'drp', 'dqn_r', 'brs', 'ppo_nr', 'greedy']
ABL = ['quart', 'quart_c', 'quart_u', 'quart_nomask', 'quart_nolag']
W1, W2 = 3.5, 7.16


def load(tag, method, N):
    out = []
    for f in sorted(glob.glob(f'{RES}/{tag}_{method}_N{N}_s*.json')):
        out.append(json.load(open(f)))
    return out


def hist_arr(runs, key):
    return np.array([[h[key] for h in r['hist']] for r in runs])


def smooth(x, k=5):
    if k <= 1:
        return x
    pad = np.pad(x, (k - 1, 0), mode='edge')
    return np.convolve(pad, np.ones(k) / k, mode='valid')


def ev_mean(runs, key):
    vals = [np.mean([e[key] for e in r['eval']]) for r in runs]
    return float(np.mean(vals)), float(np.std(vals))


def save(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf'); fig.savefig(f'{FIG}/{name}.png', dpi=200); plt.close(fig)


# ------------------------------------------------------------------ ECG ---
def fig_ecg():
    d = np.load(f'{RES}/ecg_pool.npz')
    met = json.load(open(f'{RES}/ecg_metrics.json'))['metrics']
    ex = d['example']
    fig, ax = plt.subplots(1, 2, figsize=(W1, 1.75), gridspec_kw=dict(width_ratios=[1.6, 1]))
    names = ['NSR', 'AF', 'PVC', 'VT']
    t = np.arange(2500) / 250
    for k in range(4):
        ax[0].plot(t, ex[k] / 4 - 1.6 * k, color='#2a78d6' if k == 0 else '#e34948', lw=0.6)
        ax[0].text(-0.2, -1.6 * k + 0.3, names[k], ha='right', fontsize=7, color='#0b0b0b')
    ax[0].set_yticks([]); ax[0].set_xlabel('Time (s)'); ax[0].set_xlim(-1.2, 10); ax[0].grid(False)
    ax[0].set_title('(a) Example 10-s MIT-BIH windows')
    cm = np.array(met['cm'], float); cmn = cm / cm.sum(1, keepdims=True)
    im = ax[1].imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax[1].text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center', fontsize=6,
                       color='white' if cmn[i, j] > 0.6 else '#0b0b0b')
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(names, fontsize=6)
    ax[1].set_yticks(range(4)); ax[1].set_yticklabels(names, fontsize=6)
    ax[1].set_xlabel('Predicted'); ax[1].set_ylabel('True'); ax[1].grid(False)
    ax[1].set_title('(b) Analytics model')
    fig.tight_layout(w_pad=0.6)
    save(fig, 'fig_ecg')
    rec = met['per_class_recall']
    src = json.load(open(f'{RES}/ecg_metrics.json')).get('source', 'synthetic')
    with open(f'{TAB}/tab_ecg.tex', 'w') as f:
        f.write('\\begin{tabular}{lc}\n\\toprule\nMetric & Value\\\\\n\\midrule\n')
        cc = met.get('class_counts', [0, 0, 0, 0])
        if src == 'mitdb':
            f.write(f"Records / 10-s windows & 44 / {met['n_windows']:,}\\\\\n")
            f.write(f"NSR / AF / PVC / VT-run windows & {cc[0]:,} / {cc[1]} / {cc[2]:,} / {cc[3]}\\\\\n")
            f.write("Evaluation & 2-fold inter-patient (DS1$\\leftrightarrow$DS2)\\\\\n\\midrule\n")
        else:
            f.write(f"Windows (train / test) & {met['n_windows']-met['n_test']} / {met['n_test']}\\\\\n")
        b = met.get('beat')
        if b:
            f.write(f"Beat level: V sensitivity / PPV & {100*b['V_sens']:.1f} / {100*b['V_ppv']:.1f}\\%\\\\\n")
        f.write(f"Window 4-class accuracy & {100*met['acc']:.1f}\\%\\\\\n")
        f.write(f"Recall NSR / AF / PVC / VT & {100*rec[0]:.1f} / {100*rec[1]:.1f} / {100*rec[2]:.1f} / {100*rec[3]:.1f}\\%\\\\\n")
        f.write(f"Abnormal-alarm sensitivity / specificity & {100*met['sens_abnormal']:.1f} / {100*met['spec_abnormal']:.1f}\\%\\\\\n")
        if 'alarm_any_vt' in met:
            f.write(f"VT-run windows raising an alarm & {100*met['alarm_any_vt']:.1f}\\%\\\\\n")
        f.write(f"On-hub triage: sensitivity / E-flag rate & {100*met['triage_sens']:.1f} / {100*met['triage_flag_rate']:.1f}\\%\\\\\n")
        f.write('\\bottomrule\n\\end{tabular}\n')


# ------------------------------------------------------------------ QKD ---
def fig_qkd():
    q = json.load(open(f'{RES}/qkd.json'))
    L = np.array(q['dist'])
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.6), sharex=True)
    def pos(v):
        v = np.array(v, float); v[v <= 0] = np.nan; return v
    ax[0].semilogy(L, q['plob'], color='#9a9890', ls=(0, (1, 1)), lw=1.0, label='PLOB bound')
    ax[0].semilogy(L, pos(q['K_noqs_0.01']), color='#2a78d6', marker='o', ms=3, label=r'no QS, $\varepsilon=0.01$')
    ax[0].semilogy(L, pos(q['K_qs_0.01']), color='#2a78d6', marker='o', ms=3, ls='--', mfc='white', label=r'QS, $\varepsilon=0.01$')
    ax[0].semilogy(L, pos(q['K_noqs_0.02']), color='#eb6834', marker='s', ms=3, label=r'no QS, $\varepsilon=0.02$')
    ax[0].semilogy(L, pos(q['K_qs_0.02']), color='#eb6834', marker='s', ms=3, ls='--', mfc='white', label=r'QS, $\varepsilon=0.02$ (no positive rate)')
    ax[0].set_ylabel('Key rate (bits/pulse)'); ax[0].set_ylim(1e-6, 1)
    ax[0].legend(ncol=1, loc='lower left'); ax[0].set_title('(a) Asymptotic key rate (QPSK, $\\beta=0.95$)')
    for eps, c, m in [(0.01, '#2a78d6', 'o'), (0.02, '#eb6834', 's')]:
        n = np.array(q[f'Nmax_{eps}'], float); n[n <= 0] = np.nan
        ax[1].semilogy(L, n, color=c, marker=m, ms=3, label=rf'$\varepsilon={eps}$ (best of QS / no QS)')
    for N, ls in [(80, '-'), (1000, '--')]:
        ax[1].axhline(N, color='#52514e', lw=0.7, ls=ls)
        ax[1].text(L[-1], N * 1.25, f'N = {N}', ha='right', fontsize=6.5, color='#52514e')
    ax[1].set_xlabel('Fibre distance $L$ (km)'); ax[1].set_ylabel('Max. OTP-protected entries')
    ax[1].legend(loc='upper right'); ax[1].set_title('(b) Registry key budget ($R_s$=100 MBd, 10 Hz refresh)')
    fig.tight_layout()
    save(fig, 'fig_keyrate')


# ------------------------------------------------------------- dynamics ---
def band(ax, x, Y, st, k=5, lw=1.3):
    mu = smooth(Y.mean(0), k)
    lo, hi = smooth(Y.min(0), k), smooth(Y.max(0), k)
    ax.plot(x, mu, color=st['c'], ls=st['ls'], lw=lw, label=st['label'])
    ax.fill_between(x, lo, hi, color=st['c'], alpha=0.12, lw=0)


def fig_attack(N=40):
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    for m in ['quart', 'quart_c', 'drp', 'dqn_r', 'brs', 'ppo_nr', 'greedy']:
        runs = load('main', m, N)
        if not runs: continue
        Y = hist_arr(runs, 'attack'); x = np.arange(1, Y.shape[1] + 1)
        band(ax, x, 100 * Y, STYLE[m])
    ax.set_xlabel('Training episode'); ax.set_ylabel('Attack rate (%)'); ax.set_ylim(0, 100)
    ax.legend(ncol=2, loc='center right')
    fig.tight_layout(); save(fig, f'fig_attack_N{N}')


def fig_rep(N=40):
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.0), sharey=True)
    for m in ['quart', 'quart_c', 'dqn_r', 'brs', 'drp']:
        runs = load('main', m, N)
        if not runs: continue
        x = np.arange(1, len(runs[0]['hist']) + 1)
        band(ax[0], x, hist_arr(runs, 'rep_h'), STYLE[m])
        band(ax[1], x, hist_arr(runs, 'rep_m'), STYLE[m])
    ax[0].set_title('(a) Honest devices'); ax[1].set_title('(b) Compromised devices')
    for a in ax:
        a.set_xlabel('Training episode'); a.axhline(0.45, color='#52514e', lw=0.6, ls=':')
    ax[0].set_ylabel('Mean trust score'); ax[0].set_ylim(0, 1.02)
    ax[1].text(5, 0.47, r'$\tau_q$', fontsize=7, color='#52514e')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); save(fig, f'fig_rep_N{N}')


def fig_types(N=40):
    names = [(3, '(a) Reputation forgery'), (4, '(b) Uplink jamming'), (5, '(c) Deception / spoofing'), (2, '(d) Eavesdropping')]
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.9), sharey=False)
    for ax, (b, title) in zip(axs, names):
        for m in ['quart', 'quart_c', 'drp', 'dqn_r', 'brs', 'greedy']:
            runs = load('main', m, N)
            if not runs: continue
            Y = np.array([[h['att_type'][b] for h in r['hist']] for r in runs]) * 100
            band(ax, np.arange(1, Y.shape[1] + 1), Y, STYLE[m], k=7, lw=1.1)
        ax.set_title(title, fontsize=7); ax.set_xlabel('Training episode'); ax.set_ylim(0, 100)
    axs[0].set_ylabel('Behaviour rate (%)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=0.8); save(fig, f'fig_types_N{N}')


def fig_clinical(N=40):
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.0), gridspec_kw=dict(width_ratios=[1.5, 1]))
    meths = [m for m in CMP if load('main', m, N)]
    x = np.arange(3); w = 0.8 / len(meths)
    for j, m in enumerate(meths):
        runs = load('main', m, N)
        v = np.mean([[np.mean([e['prio_ontime'][p] for e in r['eval']]) for p in range(3)] for r in runs], 0)
        ax[0].bar(x + (j - (len(meths) - 1) / 2) * w, 100 * v, w * 0.9, color=STYLE[m]['c'], label=STYLE[m]['label'])
    ax[0].set_xticks(x); ax[0].set_xticklabels(['Emergency', 'Urgent', 'Routine'], fontsize=6.5)
    ax[0].set_ylabel('On-time delivery (%)'); ax[0].set_ylim(0, 105); ax[0].set_title('(a) By clinical priority')
    vals = [100 * ev_mean(load('main', m, N), 'crit_miss')[0] for m in meths]
    ax[1].barh(np.arange(len(meths)), vals, 0.7, color=[STYLE[m]['c'] for m in meths])
    for i, v in enumerate(vals):
        ax[1].text(v + 1, i, f'{v:.1f}', va='center', fontsize=6.5, color='#0b0b0b')
    ax[1].set_yticks(np.arange(len(meths))); ax[1].set_yticklabels([STYLE[m]['label'].replace(' (ours)', '') for m in meths], fontsize=6.5)
    ax[1].invert_yaxis(); ax[1].set_xlabel('Missed critical alarms (%)'); ax[1].set_title('(b) Abnormal windows')
    ax[1].set_xlim(0, max(vals) * 1.3); ax[1].grid(axis='y', visible=False)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=6)
    fig.tight_layout(); save(fig, f'fig_clinical_N{N}')


def fig_training(N=40):
    fig, ax = plt.subplots(1, 2, figsize=(W1, 1.9))
    for m in ['quart', 'quart_nolag']:
        runs = load('main', m, N)
        if not runs: continue
        x = np.arange(1, len(runs[0]['hist']) + 1)
        band(ax[0], x, 100 * hist_arr(runs, 'emerg_viol'), STYLE[m], k=7)
        band(ax[1], x, 100 * hist_arr(runs, 'all_wsuccess'), STYLE[m], k=7)
    ax[0].axhline(5, color='#52514e', lw=0.6, ls=':'); ax[0].text(150, 6.5, r'$\delta$=5%', fontsize=6.5)
    ax[0].set_ylabel('Emergency violation (%)'); ax[0].set_ylim(0, 60)
    ax[1].set_ylabel('Weighted success, all patients (%)')
    for a in ax: a.set_xlabel('Training episode')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(); save(fig, f'fig_training_N{N}')


def fig_qclass(N=40):
    qs = [0.0, 0.1, 0.3, 0.6, 0.9]
    att, suc, satt, ssuc = [], [], [], []
    for q in qs:
        runs = load(f'qc{q}', 'quart_c', N)
        if not runs: return
        a = ev_mean(runs, 'attack'); s = ev_mean(runs, 'all_wsuccess')
        att.append(100 * a[0]); satt.append(100 * a[1]); suc.append(100 * s[0]); ssuc.append(100 * s[1])
    ref = load('main', 'quart', N)
    ra, rs = 100 * ev_mean(ref, 'attack')[0], 100 * ev_mean(ref, 'all_wsuccess')[0]
    fig, ax = plt.subplots(1, 2, figsize=(W1, 1.85))
    st = STYLE['quart_c']
    ax[0].errorbar(qs, att, satt, color=st['c'], marker=st['m'], ms=4, capsize=2, label='Classically signed registry')
    ax[0].axhline(ra, color=STYLE['quart']['c'], ls='-', lw=1.0, label='QKD-protected (QuaRT)')
    ax[1].errorbar(qs, suc, ssuc, color=st['c'], marker=st['m'], ms=4, capsize=2)
    ax[1].axhline(rs, color=STYLE['quart']['c'], ls='-', lw=1.0)
    ax[0].set_ylabel('Attack rate (%)'); ax[1].set_ylabel('Weighted success (%)')
    ax[0].set_title('(a) Adversary behaviour'); ax[1].set_title('(b) Clinical QoS, all patients')
    for a in ax: a.set_xlabel(r'Forgery success prob. $q_{\mathrm{c}}$')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); save(fig, 'fig_qclass')
    return dict(qs=qs, att=att, suc=suc, ref_att=ra, ref_suc=rs)


def fig_malfrac(N=40):
    mfs = [0.1, 0.2, 0.3, 0.4]
    meths = ['quart', 'drp', 'dqn_r', 'brs', 'greedy']
    fig, ax = plt.subplots(1, 3, figsize=(W1, 1.75))
    out = {}
    for m in meths:
        a, s, l = [], [], []
        for mf in mfs:
            runs = load(f'mf{mf}', m, N)
            if not runs: break
            a.append(100 * ev_mean(runs, 'attack')[0]); s.append(100 * ev_mean(runs, 'all_wsuccess')[0])
            l.append(100 * ev_mean(runs, 'crit_miss')[0])
        if len(a) < len(mfs): continue
        st = STYLE[m]; out[m] = (a, s, l)
        x = 100 * np.array(mfs)
        ax[0].plot(x, a, color=st['c'], marker=st['m'], ls=st['ls'], ms=3.5, label=st['label'])
        ax[1].plot(x, s, color=st['c'], marker=st['m'], ls=st['ls'], ms=3.5)
        ax[2].plot(x, l, color=st['c'], marker=st['m'], ls=st['ls'], ms=3.5)
    for a, t in zip(ax, ['(a) Attack rate (%)', '(b) W. success, all (%)', '(c) Missed alarms (%)']):
        a.set_xlabel(r'$\rho$ (%)'); a.set_title(t, fontsize=7); a.set_xticks(100 * np.array(mfs)); a.set_ylim(0, 100)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=5.8, columnspacing=0.8, handlelength=1.8)
    fig.tight_layout(); save(fig, 'fig_malfrac')
    return out


# --------------------------------------------------------------- tables ---
def fmt(m, s, pct=True, d=1):
    k = 100 if pct else 1
    return f'{k*m:.{d}f}$\\pm${k*s:.{d}f}'


def tab_main():
    keys = [('attack', True, 1), ('wsuccess', True, 1), ('all_wsuccess', True, 1), ('cmp_service', True, 1),
            ('emerg_viol', True, 1), ('crit_miss', True, 1), ('leak_per_1k', False, 1), ('energy_mj', False, 2), ('goodput_mbps', False, 1)]
    summary = {}
    with open(f'{TAB}/tab_main.tex', 'w') as f:
        for N in (40, 80):
            if not load('main', 'quart', N): continue
            f.write(f'\\midrule\n\\multicolumn{{10}}{{l}}{{\\textit{{N = {N} WBANs, $\\rho$ = 25\\%}}}}\\\\\n\\midrule\n')
            best = {}
            rows = {}
            for m in CMP:
                runs = load('main', m, N)
                if not runs: continue
                rows[m] = [ev_mean(runs, k) for k, _, _ in keys]
            summary[N] = rows
            if not rows: continue
            for j, (k, _, _) in enumerate(keys):
                vals = {m: r[j][0] for m, r in rows.items()}
                hib = k in ('wsuccess', 'all_wsuccess', 'cmp_service', 'goodput_mbps')
                best[k] = max(vals, key=vals.get) if hib else min(vals, key=vals.get)
            for m, r in rows.items():
                cells = []
                for j, (k, pct, d) in enumerate(keys):
                    c = fmt(r[j][0], r[j][1], pct, d)
                    if best[k] == m: c = '\\textbf{' + c + '}'
                    cells.append(c)
                f.write(STYLE[m]['label'].replace(' (ours)', '') + ' & ' + ' & '.join(cells) + '\\\\\n')
    with open(f'{TAB}/tab_ablation.tex', 'w') as f:
        rows = {}
        for m in ABL:
            runs = load('main', m, 40)
            if not runs: continue
            rows[m] = [ev_mean(runs, k) for k in ['attack', 'all_wsuccess', 'emerg_viol', 'crit_miss', 'leak_per_1k', 'energy_mj']]
            r = rows[m]
            f.write(STYLE[m]['label'].replace(' (ours)', '') + ' & ' + ' & '.join([
                fmt(*r[0]), fmt(*r[1]), fmt(*r[2]), fmt(*r[3]), fmt(*r[4], pct=False), fmt(*r[5], pct=False, d=2)]) + '\\\\\n')
        summary['abl'] = rows
    return summary


def tab_sweep():
    out = {}
    with open(f'{TAB}/tab_sweep.tex', 'w') as f:
        for N in (40, 80):
            p = f'{RES}/sweep_N{N}.json'
            if not os.path.exists(p): continue
            S = json.load(open(p))
            f.write(f'\\midrule\n\\multicolumn{{10}}{{l}}{{\\textit{{N = {N} WBANs}}}}\\\\\n\\midrule\n')
            cols = [('bw_mhz', 10.0), ('bw_mhz', 20.0), ('bw_mhz', 30.0), ('task_scale', 0.5), ('task_scale', 1.0),
                    ('task_scale', 1.5), ('p_max_dbm', 10.0), ('p_max_dbm', 15.0), ('p_max_dbm', 20.0)]
            tab = {}
            for m in CMP:
                row = []
                for (pname, v) in cols:
                    vals = [s['all_wsuccess'] for s in S if s['method'] == m and s['param'] == pname and s['value'] == v]
                    row.append(100 * np.mean(vals) if vals else np.nan)
                tab[m] = row
            best = [max(tab, key=lambda mm: tab[mm][j]) for j in range(len(cols))]
            for m in CMP:
                cells = [('\\textbf{%.1f}' if best[j] == m else '%.1f') % tab[m][j] for j in range(len(cols))]
                f.write(STYLE[m]['label'].replace(' (ours)', '') + ' & ' + ' & '.join(cells) + '\\\\\n')
            out[N] = tab
    return out


if __name__ == '__main__':
    summ = {}
    for fn in [fig_ecg, fig_qkd]:
        try: fn()
        except Exception as e: print('skip', fn.__name__, e)
    for N in (40, 80):
        for fn in [fig_attack, fig_rep, fig_types, fig_clinical]:
            try: fn(N)
            except Exception as e: print('skip', fn.__name__, N, e)
    try: fig_training(40)
    except Exception as e: print('skip training', e)
    summ['qclass'] = fig_qclass()
    summ['malfrac'] = fig_malfrac()
    summ['main'] = tab_main()
    summ['sweep'] = tab_sweep()
    json.dump(summ, open(f'{RES}/summary.json', 'w'), indent=1, default=lambda o: o if not isinstance(o, np.ndarray) else o.tolist())
    print('figures ->', FIG, ' tables ->', TAB)
