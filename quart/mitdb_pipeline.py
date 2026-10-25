"""Two-stage, patient-independent arrhythmia analytics on the MIT-BIH Arrhythmia Database.

Stage 1 (beat level): every annotated beat is described by RR-interval features
    (pre-/post-/local-/record-normalised RR) and patient-adaptive morphology features
    (down-sampled beat, correlation with the patient's dominant template, QRS-width,
    amplitude and energy ratios), in the spirit of de Chazal et al. (IEEE TBME 2004).
    An MLP classifies beats into N / S / V classes.
Stage 2 (window level): each 10-s window is summarised by the stage-1 posteriors
    (mean/max V probability, V count, longest run of consecutive V beats, S share) and
    by rhythm-irregularity features; a second MLP outputs NSR / AF / PVC / VT-run.
Stage 2 is trained on out-of-fold stage-1 posteriors (patient-wise cross-fitting), so the
reported held-out performance is not optimistically biased.
Beat fiducial points are taken from the database annotations, as is common practice.
"""
import os
import numpy as np
from .nn import MLP, Adam, softmax

FS = 250
WIN = 10 * FS
N_SYM = {'N', 'L', 'R', 'e', 'j'}
S_SYM = {'A', 'a', 'J', 'S'}
V_SYM = {'V', 'E'}
BEAT_SYM = N_SYM | S_SYM | V_SYM | {'F', '/', 'f', 'Q'}
PRE, POST, DS = 50, 90, 4          # -200 ms .. +360 ms, every 16 ms


def _filter(sig):
    from scipy.ndimage import median_filter
    from scipy.signal import butter, filtfilt
    base = median_filter(median_filter(sig, 51), 151)       # 200-ms / 600-ms baseline
    b, a = butter(3, 35 / (FS / 2), 'low')
    out = filtfilt(b, a, sig - base)
    return out / (np.std(out) + 1e-9)


def _record_beats(sig, samp, sym):
    """Beat features and labels for one record."""
    keep = [i for i, s in enumerate(sym) if s in BEAT_SYM and PRE <= samp[i] < len(sig) - POST]
    r = samp[keep]
    s = [sym[i] for i in keep]
    lab = np.array([0 if x in N_SYM else 1 if x in S_SYM else 2 if x in V_SYM else 3 for x in s])
    rr = np.diff(r) / FS
    pre = np.r_[rr[0], rr]
    post = np.r_[rr, rr[-1]]
    n = len(r)
    loc = np.array([pre[max(0, i - 5):i + 6].mean() for i in range(n)])
    glob = np.array([np.median(pre[max(0, i - 300):i + 1]) for i in range(n)])
    segs = np.stack([sig[p - PRE:p + POST] for p in r])
    # patient-adaptive dominant template, refreshed every 50 beats from +-150 beats
    tmpl = np.zeros_like(segs)
    for c in range(0, n, 50):
        lo, hi = max(0, c - 150), min(n, c + 200)
        tmpl[c:c + 50] = np.median(segs[lo:hi], 0)
    qrs = slice(PRE - 25, PRE + 30)
    def _w(b):
        q = np.abs(b[qrs] - np.median(b))
        return (q > 0.3 * q.max()).sum()
    bc = segs - segs.mean(1, keepdims=True); tc = tmpl - tmpl.mean(1, keepdims=True)
    corr = (bc * tc).sum(1) / (np.linalg.norm(bc, axis=1) * np.linalg.norm(tc, axis=1) + 1e-9)
    wid = np.array([_w(b) for b in segs]); twid = np.array([_w(t) for t in tmpl])
    amp = np.ptp(segs[:, qrs], 1); tamp = np.ptp(tmpl[:, qrs], 1)
    en = (segs[:, qrs] ** 2).sum(1); ten = (tmpl[:, qrs] ** 2).sum(1)
    feats = np.column_stack([
        pre / glob, post / glob, loc / glob, pre / post, pre / loc, glob,
        corr, wid / (twid + 1e-6), amp / (tamp + 1e-6), np.log((en + 1e-6) / (ten + 1e-6)),
        segs[:, ::DS] / (tamp[:, None] + 1e-6), (segs - tmpl)[:, ::DS] / (tamp[:, None] + 1e-6)])
    return r, lab, feats.astype(np.float32)


def load_mitdb_beats(cache_dir, records):
    """Returns per-record beat data and the labelled 10-s windows (same rules as ecg.load_mitdb)."""
    import wfdb
    from scipy.signal import resample_poly
    recs, wins = [], []
    for pi, rec in enumerate(records):
        rd = wfdb.rdrecord(os.path.join(cache_dir, str(rec)))
        a = wfdb.rdann(os.path.join(cache_dir, str(rec)), 'atr')
        raw = resample_poly(rd.p_signal[:, 0], 25, 36)
        sig = _filter(raw)
        samp = np.round(np.asarray(a.sample) * 250 / 360).astype(int)
        rpos, blab, bfeat = _record_beats(sig, samp, list(a.symbol))
        recs.append(dict(r=rpos, lab=blab, X=bfeat))
        rhythm = np.empty(len(raw), dtype=object); cur = '(N'; last = 0
        for s_, sym, aux in zip(samp, a.symbol, a.aux_note):
            if sym == '+' and aux:
                rhythm[last:s_] = cur; cur = aux.strip('\x00'); last = s_
        rhythm[last:] = cur
        for st in range(0, len(raw) - WIN, WIN):
            rh = set(rhythm[st:st + WIN])
            m = (samp >= st) & (samp < st + WIN)
            syms = [a.symbol[i] for i in np.where(m)[0] if a.symbol[i] not in ('+', '~', '|', '"', 'x', '!')]
            run = best = 0
            for sy in syms:
                run = run + 1 if sy in V_SYM else 0
                best = max(best, run)
            if '(VT' in rh or '(VFL' in rh or best >= 3:
                lab = 3
            elif rh <= {'(AFIB', '(AFL'}:
                lab = 1
            elif rh == {'(N'} and any(s in V_SYM for s in syms):
                lab = 2
            elif rh == {'(N'} and set(syms) <= N_SYM:
                lab = 0
            else:
                continue
            bidx = np.where((rpos >= st) & (rpos < st + WIN))[0]
            w = raw[st:st + WIN]
            wins.append(dict(pid=pi, y=lab, bidx=bidx, x=((w - w.mean()) / (w.std() + 1e-6)).astype(np.float32)))
    return recs, wins


class _MLPClf:
    def __init__(self, din, nout, seed=0, hidden=(64, 64)):
        self.net = MLP([din, *hidden, nout], np.random.default_rng(seed), out_scale=0.1)
        self.nout = nout

    def fit(self, F, y, epochs=40, lr=2e-3, batch=256, seed=0, weight_pow=0.5):
        rng = np.random.default_rng(seed)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-6
        Z = (F - self.mu) / self.sd
        opt = Adam(self.net.params, lr=lr)
        cnt = np.bincount(y, minlength=self.nout).clip(1)
        w = (cnt.sum() / cnt) ** weight_pow; w = w / w[y].mean()
        for _ in range(epochs):
            idx = rng.permutation(len(Z))
            for b in range(0, len(Z), batch):
                j = idx[b:b + batch]
                p = softmax(self.net.forward(Z[j]))
                g = p.copy(); g[np.arange(len(j)), y[j]] -= 1
                g *= w[y[j]][:, None] / len(j)
                opt.step(self.net.backward(g))
        return self

    def proba(self, F):
        return softmax(self.net.forward((F - self.mu) / self.sd))


def _window_feats(win, P):
    """Window descriptor from beat posteriors P (n_beats x 3) and RR rhythm."""
    pv, ps = P[:, 2], P[:, 1]
    isv = pv > 0.5
    run = best = 0
    for v in isv:
        run = run + 1 if v else 0
        best = max(best, run)
    r = win['r']
    rr = np.diff(r) / FS if len(r) > 2 else np.array([1.0, 1.0])
    nv = ~isv[1:] if len(isv) > 1 else np.ones(len(rr), bool)
    rrn = rr[nv[:len(rr)]] if nv[:len(rr)].sum() > 2 else rr       # RR between non-V beats
    d = np.abs(np.diff(rrn)) if len(rrn) > 1 else np.array([0.0])
    return np.array([pv.mean(), pv.max(), isv.sum() / 10, best / 5, float(best >= 3), ps.mean(), ps.max(),
                     len(r) / 20, 60 / np.median(rr) / 100, np.std(rrn) / np.mean(rrn), d.mean() / np.mean(rrn),
                     np.mean(d > 0.1 * np.mean(rrn)), rr.min() / np.median(rr), rr.max() / np.median(rr)],
                    dtype=np.float32)


# de Chazal et al. inter-patient split (paced records 102, 104, 107, 217 excluded)
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


def run_pipeline(cache_dir, records=None, seed=0, n_folds=3, cache=None):
    """Two-fold inter-patient cross-validation (train DS1 -> test DS2 and train DS2 ->
    test DS1): every window receives a prediction from a model that never saw its patient."""
    records = records or (DS1 + DS2)
    if cache is None:
        cache = load_mitdb_beats(cache_dir, records)
    recs, wins = cache
    out = {}
    for test_set in (DS2, DS1):
        out[tuple(test_set)] = _one_fold(recs, wins, records, test_set, seed, n_folds)
    y = np.array([w['y'] for w in wins]); pid = np.array([w['pid'] for w in wins])
    pred = np.zeros(len(wins), int); WF = None
    for ts, (pr, te_mask, bm, wf) in out.items():
        pred[te_mask] = pr[te_mask]
        WF = wf if WF is None else np.where(te_mask[:, None], wf, WF)
    te = np.ones(len(wins), bool)
    bms = [o[2] for o in out.values()]
    beat_metrics = {k: float(sum(b[k] * b['n_beats_test'] for b in bms) / sum(b['n_beats_test'] for b in bms))
                    for k in bms[0] if k != 'n_beats_test'}
    beat_metrics['n_beats_test'] = int(sum(b['n_beats_test'] for b in bms))
    beat_metrics['per_fold'] = bms
    X = np.stack([w['x'] for w in wins])
    rec_id = np.array([records[p] for p in pid])
    return X, y, pid, te, pred, beat_metrics, WF, rec_id


def _one_fold(recs, wins, records, test_set, seed, n_folds):
    test_p = {i for i, r in enumerate(records) if r in test_set}
    train_p = [i for i, r in enumerate(records) if r not in test_set]
    for w in wins:
        w['r'] = recs[w['pid']]['r'][w['bidx']]
    train_p = list(train_p)

    def beat_data(plist):
        X = np.concatenate([recs[p]['X'][recs[p]['lab'] < 3] for p in plist])
        y = np.concatenate([recs[p]['lab'][recs[p]['lab'] < 3] for p in plist])
        return X, y

    # ---- out-of-fold beat posteriors for training patients (cross-fitting)
    post = {}
    folds = [train_p[k::n_folds] for k in range(n_folds)]
    for k in range(n_folds):
        tr = [p for p in train_p if p not in folds[k]]
        clf = _MLPClf(recs[0]['X'].shape[1], 3, seed + k).fit(*beat_data(tr), seed=seed + k)
        for p in folds[k]:
            post[p] = clf.proba(recs[p]['X'])
    beat_clf = _MLPClf(recs[0]['X'].shape[1], 3, seed).fit(*beat_data(train_p), seed=seed)
    for p in test_p:
        post[p] = beat_clf.proba(recs[p]['X'])
    # ---- window level
    WF = np.stack([_window_feats(w, post[w['pid']][w['bidx']]) if len(w['bidx']) else np.zeros(14, np.float32)
                   for w in wins])
    y = np.array([w['y'] for w in wins]); pid = np.array([w['pid'] for w in wins])
    te = np.array([p in test_p for p in pid])
    win_clf = _MLPClf(WF.shape[1], 4, seed, hidden=(32, 32)).fit(WF[~te], y[~te], epochs=200, batch=128,
                                                                   seed=seed, weight_pow=0.7)
    pred = win_clf.proba(WF).argmax(1)
    # beat-level held-out metrics (for reporting)
    Xb, yb = beat_data(sorted(test_p))
    pb = beat_clf.proba(Xb).argmax(1)
    beat_metrics = dict(
        n_beats_test=int(len(yb)),
        V_sens=float(((pb == 2) & (yb == 2)).sum() / max((yb == 2).sum(), 1)),
        V_ppv=float(((pb == 2) & (yb == 2)).sum() / max((pb == 2).sum(), 1)),
        S_sens=float(((pb == 1) & (yb == 1)).sum() / max((yb == 1).sum(), 1)),
        N_spec_V=float(((pb != 2) & (yb != 2)).sum() / max((yb != 2).sum(), 1)))
    return pred, te, beat_metrics, WF
