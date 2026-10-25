"""ECG workload for the WBAN analytics tasks.

Two sources are supported and produce exactly the same window format
(10-s, single-lead, fs = 250 Hz, label in {0: normal sinus rhythm,
1: atrial fibrillation, 2: ventricular ectopy (PVC), 3: ventricular
tachycardia-like}):

* ``mitdb``     - the MIT-BIH Arrhythmia Database (PhysioNet), read with
                  ``wfdb`` if it is installed and PhysioNet is reachable;
* ``synthetic`` - a sum-of-Gaussians dynamical ECG model (in the spirit of
                  ECGSYN, McSharry et al., IEEE TBME 2003) with rhythm-specific
                  RR processes, morphology changes and realistic artefacts
                  (baseline wander, power-line, EMG noise, motion bursts).

The module also implements the lightweight on-hub triage (QRS detection +
RR-irregularity / rate rules) that assigns the clinical priority of each
task, and the analytics model (an MLP on RR + morphology features) whose
inference is what gets offloaded.
"""
import os
import numpy as np
from .nn import MLP, Adam, softmax, log_softmax

FS = 250
WIN_S = 10
WIN = FS * WIN_S
CLASSES = ['NSR', 'AF', 'PVC', 'VT']


# ------------------------------------------------------------ synthesis ---
def _beat(t, hr, rng, kind, amp):
    """Sum-of-Gaussians single beat centred at R (t in seconds rel. to R)."""
    s = np.sqrt(60.0 / hr)              # QT scaling with heart rate (Bazett-like)
    if kind == 'normal':
        waves = [(-0.20 * s, 0.025, 0.12 * amp[0]),   # P
                 (-0.035, 0.010, -0.12 * amp[1]),     # Q
                 (0.0, 0.011, 1.0 * amp[1]),          # R
                 (0.035, 0.010, -0.22 * amp[1]),      # S
                 (0.30 * s, 0.060 * s, 0.30 * amp[2])]  # T
    elif kind == 'af':
        waves = [(-0.035, 0.010, -0.12 * amp[1]), (0.0, 0.011, 1.0 * amp[1]),
                 (0.035, 0.010, -0.22 * amp[1]), (0.30 * s, 0.060 * s, 0.27 * amp[2])]
    else:  # ventricular (wide QRS, discordant T, no P)
        w = rng.uniform(0.030, 0.045)
        waves = [(0.0, w, 1.25 * amp[1] * rng.choice([1, -1], p=[0.7, 0.3])),
                 (0.06, w, -0.35 * amp[1]),
                 (0.32 * s, 0.075 * s, -0.40 * amp[2])]
    y = np.zeros_like(t)
    for mu, sd, a in waves:
        y += a * np.exp(-0.5 * ((t - mu) / sd) ** 2)
    return y


def synth_window(label, rng, patient=None):
    """One 10-s ECG window of the given rhythm class."""
    if patient is None:
        patient = dict(amp=rng.uniform(0.7, 1.3, 3), hr=rng.uniform(58, 95))
    amp = patient['amp']
    T = WIN_S + 2.0
    beats = []       # (time of R, kind, local hr)
    t = rng.uniform(0, 0.8)
    if label == 0:
        hr = patient['hr'] * rng.uniform(0.95, 1.05)
        while t < T:
            rr = 60 / hr * (1 + 0.03 * rng.standard_normal() + 0.02 * np.sin(2 * np.pi * 0.25 * t))
            beats.append((t, 'normal', 60 / rr)); t += rr
    elif label == 1:
        hr = rng.uniform(85, 150)
        while t < T:
            rr = 60 / hr * rng.uniform(0.6, 1.45)
            beats.append((t, 'af', 60 / rr)); t += rr
    elif label == 2:
        hr = patient['hr']
        p_pvc = rng.uniform(0.12, 0.35)
        prev_pvc = False
        while t < T:
            rr = 60 / hr * (1 + 0.03 * rng.standard_normal())
            if (not prev_pvc) and rng.random() < p_pvc:
                rr_c = rr * rng.uniform(0.55, 0.75)              # premature
                beats.append((t + rr_c, 'ventricular', 60 / rr_c))
                t = t + rr_c + rr * rng.uniform(1.25, 1.45)      # compensatory pause
                prev_pvc = True
                continue
            beats.append((t, 'normal', 60 / rr)); t += rr; prev_pvc = False
    else:
        hr = rng.uniform(140, 200)
        while t < T:
            rr = 60 / hr * (1 + 0.02 * rng.standard_normal())
            beats.append((t, 'ventricular', 60 / rr)); t += rr
    tt = np.arange(int(T * FS)) / FS
    y = np.zeros_like(tt)
    for (tr, kind, lhr) in beats:
        idx = (tt > tr - 0.6) & (tt < tr + 0.8)
        y[idx] += _beat(tt[idx] - tr, max(lhr, 30), rng, kind, amp)
    if label == 1:   # fibrillatory baseline
        f = rng.uniform(4, 8)
        y += 0.04 * np.sin(2 * np.pi * f * tt + rng.uniform(0, 6)) * (1 + 0.5 * rng.standard_normal(tt.size).cumsum() / np.sqrt(tt.size))
    # artefacts
    y += rng.uniform(0.05, 0.25) * np.sin(2 * np.pi * rng.uniform(0.15, 0.5) * tt + rng.uniform(0, 6))
    y += rng.uniform(0.0, 0.03) * np.sin(2 * np.pi * 50 * tt)
    y += rng.uniform(0.01, 0.05) * rng.standard_normal(tt.size)
    if rng.random() < 0.25:   # motion burst
        c = rng.uniform(1, T - 1)
        y += rng.uniform(0.2, 0.6) * np.exp(-0.5 * ((tt - c) / 0.3) ** 2) * rng.standard_normal(tt.size)
    s0 = int(1.0 * FS)
    w = y[s0:s0 + WIN]
    return ((w - w.mean()) / (w.std() + 1e-6)).astype(np.float32)


def synth_dataset(n_patients=48, windows_per_patient=60, rng=None,
                  class_mix=(0.62, 0.18, 0.14, 0.06)):
    """Patient-structured synthetic corpus (48 'records', like MIT-BIH)."""
    rng = rng or np.random.default_rng(0)
    X, y, pid = [], [], []
    for p in range(n_patients):
        patient = dict(amp=rng.uniform(0.7, 1.3, 3), hr=rng.uniform(58, 95))
        # patient-specific rhythm propensity
        mix = rng.dirichlet(np.array(class_mix) * 12)
        for _ in range(windows_per_patient):
            lab = rng.choice(4, p=mix)
            X.append(synth_window(lab, rng, patient)); y.append(lab); pid.append(p)
    return np.stack(X), np.array(y), np.array(pid)


# --------------------------------------------------------------- MIT-BIH ---
MITDB_RECORDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114,
                 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205,
                 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223,
                 228, 230, 231, 232, 233, 234]


def mitdb_missing(cache_dir='data/mitdb', records=MITDB_RECORDS):
    """Records whose .hea/.atr/.dat files are absent or truncated (.dat must be 1,950,000 bytes)."""
    bad = []
    for r in records:
        p = os.path.join(cache_dir, str(r))
        ok = all(os.path.exists(p + e) and os.path.getsize(p + e) > 0 for e in ('.hea', '.atr', '.dat'))
        if ok and os.path.getsize(p + '.dat') != 1950000:
            ok = False
        if not ok:
            bad.append(r)
    return bad


def load_mitdb(cache_dir='data/mitdb', records=MITDB_RECORDS):
    """Window MIT-BIH into the common format. Labels: any '(VT'/'(VFL' rhythm or a run
    of >=3 consecutive ventricular beats -> VT/ventricular run; '(AFIB'/'(AFL' only -> AF;
    sinus rhythm with isolated V/E beats -> PVC; sinus rhythm with normal/bundle-branch
    beats only -> NSR; other windows (paced, SVTA, bigeminy-only, noise ...) dropped."""
    import wfdb
    from scipy.signal import resample_poly
    os.makedirs(cache_dir, exist_ok=True)
    missing = mitdb_missing(cache_dir, records)
    if missing:
        raise RuntimeError(f'MIT-BIH incomplete in {cache_dir}: {len(missing)} records missing '
                           f'({missing[:6]}...). Run `python download_mitdb.py` first.')
    X, y, pid = [], [], []
    for pi, rec in enumerate(records):
        r = wfdb.rdrecord(os.path.join(cache_dir, str(rec)))
        a = wfdb.rdann(os.path.join(cache_dir, str(rec)), 'atr')
        sig = resample_poly(r.p_signal[:, 0], 25, 36)          # 360 -> 250 Hz
        samp = (np.asarray(a.sample) * 250 / 360).astype(int)
        rhythm = np.empty(len(sig), dtype=object); cur = '(N'
        last = 0
        for s, sym, aux in zip(samp, a.symbol, a.aux_note):
            if sym == '+' and aux:
                rhythm[last:s] = cur; cur = aux.strip('\x00'); last = s
        rhythm[last:] = cur
        for st in range(0, len(sig) - WIN, WIN):
            rh = set(rhythm[st:st + WIN])
            m = (samp >= st) & (samp < st + WIN)
            syms = [a.symbol[i] for i in np.where(m)[0] if a.symbol[i] not in ('+', '~', '|', '"', 'x', '!')]
            # longest run of consecutive ventricular beats in the window
            run = best = 0
            for sy in syms:
                run = run + 1 if sy in ('V', 'E') else 0
                best = max(best, run)
            if '(VT' in rh or '(VFL' in rh or best >= 3:
                lab = 3          # ventricular tachycardia / ventricular run (>=3 consecutive V beats)
            elif rh <= {'(AFIB', '(AFL'}:
                lab = 1          # atrial fibrillation / flutter
            elif rh == {'(N'} and ('V' in syms or 'E' in syms):
                lab = 2          # sinus rhythm with isolated ventricular ectopy
            elif rh == {'(N'} and set(syms) <= {'N', 'L', 'R', 'e', 'j'}:
                lab = 0          # normal sinus rhythm
            else:
                continue
            w = sig[st:st + WIN]
            X.append(((w - w.mean()) / (w.std() + 1e-6)).astype(np.float32)); y.append(lab); pid.append(pi)
    return np.stack(X), np.array(y), np.array(pid)


def load_corpus(source='auto', seed=0):
    """Returns (X, y, pid, source_used)."""
    if source in ('auto', 'mitdb'):
        try:
            X, y, pid = load_mitdb()
            return X, y, pid, 'mitdb'
        except Exception as e:  # no wfdb / incomplete download
            if source == 'mitdb' or os.path.isdir('data/mitdb'):
                raise
            print(f'[ecg] MIT-BIH unavailable ({type(e).__name__}); using synthetic ECG corpus')
    X, y, pid = synth_dataset(rng=np.random.default_rng(seed))
    return X, y, pid, 'synthetic'


# ------------------------------------------------- QRS detection & features ---
def detect_r(x, fs=FS):
    """Pan-Tompkins-style detector (band-pass, derivative, squaring, MWI)."""
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(2, [5 / (fs / 2), 18 / (fs / 2)], 'band')
    f = filtfilt(b, a, x)
    d = np.gradient(f) ** 2
    w = int(0.12 * fs)
    mwi = np.convolve(d, np.ones(w) / w, mode='same')
    pk, _ = find_peaks(mwi, height=0.25 * np.percentile(mwi, 99), distance=int(0.27 * fs))
    # refine to |f| maximum
    out = []
    for p in pk:
        lo, hi = max(0, p - w), min(len(f), p + w)
        out.append(lo + int(np.argmax(np.abs(f[lo:hi]))))
    return np.unique(np.array(out, dtype=int)), f


def features(x):
    r, f = detect_r(x)
    if len(r) < 3:
        rr = np.array([1.0, 1.0])
    else:
        rr = np.diff(r) / FS
    hr = 60 / np.median(rr)
    drr = np.abs(np.diff(rr)) if len(rr) > 1 else np.array([0.0])
    feats = [hr / 100, np.std(rr) / np.mean(rr), np.mean(drr) / np.mean(rr),
             np.max(drr) / np.mean(rr), np.min(rr) / np.median(rr), np.max(rr) / np.median(rr),
             np.mean(drr > 0.12 * np.mean(rr)), len(r) / 20]
    # QRS width proxy and beat template (normalised)
    seg = []
    for p in r:
        if p - 50 >= 0 and p + 75 <= len(f):
            seg.append(x[p - 50:p + 75])
    if seg:
        seg = np.array(seg); tmpl = seg.mean(0); var = seg.std(0).mean()
    else:
        tmpl = np.zeros(125); var = 0.0
    tmpl_raw = tmpl
    tmpl = (tmpl - tmpl.mean()) / (np.abs(tmpl).max() + 1e-6)
    above = np.abs(tmpl) > 0.5
    width = above.sum() / 125
    beat = _beat_features(x, r, rr, seg if len(seg) else None, tmpl_raw)
    return np.concatenate([feats, [width, var], beat, tmpl[::5]]).astype(np.float32)


def _max_run(mask):
    run = best = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def _beat_features(x, r, rr, seg, tmpl):
    """Per-beat morphology/prematurity features that expose isolated or
    consecutive ventricular beats (diluted in the average template)."""
    if seg is None or len(seg) < 3:
        return np.zeros(10)
    med = np.median(seg, 0)
    medc = med - med.mean()
    corr, wid = [], []
    for b in seg:
        bc = b - b.mean()
        corr.append(float(bc @ medc / (np.linalg.norm(bc) * np.linalg.norm(medc) + 1e-9)))
        qrs = np.abs(b[35:70] - np.median(b))                 # -60 ms .. +80 ms around R
        wid.append(float((qrs > 0.35 * qrs.max()).sum()))
    corr, wid = np.array(corr), np.array(wid)
    wr = wid / (np.median(wid) + 1e-6)
    amp = np.array([np.ptp(b[35:70]) for b in seg]); ar = amp / (np.median(amp) + 1e-6)
    abn = (corr < 0.8) | (wr > 1.5)
    rr_n = rr / np.median(rr)
    prem = rr_n < 0.85
    return np.array([abn.mean(), _max_run(abn) / 10, corr.min(), corr.mean(), wr.max(), ar.max(),
                     ar.min(), prem.mean(), _max_run(prem) / 10,
                     float(np.any(prem[:-1] & (rr_n[1:] > 1.15))) if len(rr_n) > 1 else 0.0])


def triage_priority(x, acuity):
    """On-hub rule (cheap): emergency if HR>130 or <40 or marked RR irregularity;
    otherwise urgent for ICU-acuity patients, routine for ward/home."""
    r, _ = detect_r(x)
    if len(r) >= 3:
        rr = np.diff(r) / FS
        hr = 60 / np.median(rr)
        irr = np.std(rr) / np.mean(rr)
        if hr > 130 or hr < 40 or irr > 0.18:
            return 0
    return 1 if acuity == 'icu' else 2


class ECGClassifier:
    """MLP analytics model (the computation that is offloaded)."""

    def __init__(self, din, seed=0):
        rng = np.random.default_rng(seed)
        self.net = MLP([din, 64, 64, 4], rng, out_scale=0.1)
        self.mu = None

    def fit(self, F, y, epochs=150, lr=2e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-6
        Z = (F - self.mu) / self.sd
        opt = Adam(self.net.params, lr=lr)
        w = 1.0 / np.bincount(y, minlength=4).clip(1); w = w / w.mean()
        for ep in range(epochs):
            idx = rng.permutation(len(Z))
            for b in range(0, len(Z), 128):
                j = idx[b:b + 128]
                lo = self.net.forward(Z[j])
                p = softmax(lo)
                g = p.copy(); g[np.arange(len(j)), y[j]] -= 1
                g *= w[y[j]][:, None] / len(j)
                opt.step(self.net.backward(g))
        return self

    def predict(self, F):
        return self.net.forward((F - self.mu) / self.sd).argmax(1)


def _pick_examples(X, y, pred):
    """One illustrative, correctly classified window per class: for the ventricular
    classes the window with the largest share of abnormal-morphology beats."""
    out = []
    for k in range(4):
        idx = np.where((y == k) & (pred == k))[0]
        if len(idx) == 0:
            idx = np.where(y == k)[0]
        if k >= 2:
            score = [features(X[i])[10] for i in idx[:400]]
            out.append(int(idx[int(np.argmax(score))]))
        else:
            out.append(int(idx[0]))
    return out


def build_task_pool(source='auto', seed=0, out='results/ecg_pool.npz'):
    """Feature extraction, patient-wise split, classifier training/eval and the
    pool of (true label, predicted label, priority-relevant info) used by the
    network simulator. Returns a dict and writes ``out``."""
    beat_metrics = None
    use_mitdb = source == 'mitdb' or (source == 'auto' and os.path.isdir('data/mitdb'))
    if use_mitdb:
        # two-stage beat->window analytics, two-fold inter-patient CV (DS1 <-> DS2)
        from .mitdb_pipeline import run_pipeline, DS1, DS2
        missing = mitdb_missing('data/mitdb', DS1 + DS2)
        if missing:
            raise RuntimeError(f'MIT-BIH incomplete: {missing}. Run `python download_mitdb.py` first.')
        X, y, pid, te, pred, beat_metrics, _, rec_id = run_pipeline('data/mitdb', seed=seed)
        used = 'mitdb'
    else:
        X, y, pid, used = load_corpus('synthetic', seed)
        F = np.stack([features(x) for x in X])
        rng = np.random.default_rng(seed)
        pats = np.unique(pid); rng.shuffle(pats)
        test_p = set(pats[:len(pats) // 3])
        te = np.array([p in test_p for p in pid])
        clf = ECGClassifier(F.shape[1], seed).fit(F[~te], y[~te], seed=seed)
        pred = clf.predict(F)
    # triage flag computed on the raw window (priority 0 = emergency)
    flag = np.array([triage_priority(x, 'ward') == 0 for x in X])
    res = dict(y=y, pred=pred, pid=pid, test=te, flag=flag, source=used)
    yt, pt = y[te], pred[te]
    cm = np.zeros((4, 4), int)
    for a, b in zip(yt, pt):
        cm[a, b] += 1
    abn_t, abn_p = yt > 0, pt > 0
    res['metrics'] = dict(
        acc=float((yt == pt).mean()),
        sens_abnormal=float((abn_p & abn_t).sum() / max(abn_t.sum(), 1)),
        spec_abnormal=float((~abn_p & ~abn_t).sum() / max((~abn_t).sum(), 1)),
        per_class_recall=[float(cm[k, k] / max(cm[k].sum(), 1)) for k in range(4)],
        triage_sens=float((flag[te] & abn_t).sum() / max(abn_t.sum(), 1)),
        triage_flag_rate=float(flag[te].mean()),
        alarm_any_vt=float((pt[yt == 3] > 0).mean()) if (yt == 3).any() else 0.0,
        n_windows=int(len(y)), n_test=int(te.sum()), cm=cm.tolist(),
        class_counts=np.bincount(y, minlength=4).tolist(), beat=beat_metrics)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.savez(out, y=y, pred=pred, pid=pid, test=te, flag=flag, source=used,
             example=X[_pick_examples(X, y, pred)])
    return res
