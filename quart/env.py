"""Healthcare-5.0 WBAN / medical-IoT cooperative-offloading simulator with
strategic deceptive insiders and a (quantum-secured) reputation layer.

Each slot:
  1. channels evolve (distance, body-shadowing Markov state, fading);
  2. every compromised node picks a behaviour b in {0..5} (cooperate, refuse,
     eavesdrop, reputation forgery, jamming, deception/spoofing) with a
     Q-learning policy conditioned on its *own reputation level* - i.e. the
     attackers adapt to the defence (indirect-reciprocity game);
  3. honest WBAN hubs with an ECG analytics task choose an execution target
     (local / MEC / one of K bedside-peer helpers), transmit power and
     bandwidth share (decided by the method under test);
  4. the MEC admits requests (zero-trust gate), shares CPU and uplink
     bandwidth, peers serve or misbehave, outcomes are evaluated clinically;
  5. observers report behaviours -> reputation update -> table broadcast,
     which may be tampered with depending on the protection scheme.
"""
import numpy as np
from .config import Config

N_TGT = 5          # local, edge, peer0, peer1, peer2
N_POW = 3
N_BW = 3
OBS_DIM = 24

METHODS = {
    #            decision  trust     protect      classify gate  mask   lagr
    'quart':        ('ppo',  'global', 'qkd',       True,  True,  True,  True),
    'quart_c':      ('ppo',  'global', 'classical', True,  True,  True,  True),
    'quart_u':      ('ppo',  'global', 'qkd',       False, True,  True,  True),
    'quart_nomask': ('ppo',  'global', 'qkd',       True,  True,  False, True),
    'quart_nolag':  ('ppo',  'global', 'qkd',       True,  True,  True,  False),
    'ppo_nr':       ('ppo',  'none',   'none',      False, False, False, False),
    'dqn_r':        ('dqn',  'global', 'classical', False, True,  False, False),
    'brs':          ('rule', 'beta',   'classical', False, True,  True,  False),
    'drp':          ('rule', 'direct', 'none',      False, True,  True,  False),
    'greedy':       ('rule', 'none',   'none',      False, False, False, False),
}


def db2lin(x):
    return 10.0 ** (np.asarray(x) / 10.0)


class HealthEnv:
    def __init__(self, cfg: Config, method: str, pool: dict, seed: int = 0):
        self.cfg = cfg
        self.method = method
        (self.decision, self.trust_mode, self.protect, self.classify,
         self.gate, self.mask_on, self.lagr) = METHODS[method]
        self.rng = np.random.default_rng(seed)
        self.pool = pool
        c = cfg
        N = c.n_hubs
        self.N = N
        n_mal = int(round(c.mal_frac * N))
        perm = self.rng.permutation(N)
        self.mal = np.zeros(N, bool); self.mal[perm[:n_mal]] = True
        self.ward = self.rng.integers(0, c.n_wards, N)
        self.icu = self.rng.random(N) < c.icu_frac
        # each hub streams ECG windows of one (held-out) patient record
        te = np.where(pool['test'])[0]
        pids = np.unique(pool['pid'][te])
        self.hub_pid = pids[self.rng.integers(0, len(pids), N)]
        self.pid_idx = {p: te[pool['pid'][te] == p] for p in pids}
        # persistent trust state
        self.rep = np.full(N, c.rep_init)                       # global reputation
        self.ba = np.zeros(N); self.bb = np.zeros(N)            # beta reputation
        self.dir_a = np.zeros((N + 1, N)); self.dir_b = np.zeros((N + 1, N))  # direct (row N: MEC)
        # attacker learning state: Q[node, level, behaviour]
        # shared Q-table: a coordinated adversary controls all compromised nodes
        # shared Q-tables of a coordinated adversary controlling all compromised
        # nodes; one table per decision context (0: network, 1: request, 2: helper)
        self.Q = np.full((3, 2 * c.rep_levels, 6), c.att_q0)
        self.ctx_actions = {0: [0, 3, 4], 1: [0, 5], 2: [0, 1, 2, 5]}
        self.V = np.full(2 * c.rep_levels, c.att_q0)      # value of the attacker's standing
        self.quar = np.zeros(N, bool)                    # zero-trust quarantine with hysteresis
        self.d_edge = self.rng.uniform(*c.d_edge, N)
        self.nlos = self.rng.random(N) < c.p_nlos
        self.edge_load = 0.3
        self.lam = 0.0                                           # Lagrange multiplier
        self.noise_w_hz = db2lin(c.n0_dbm_hz + c.nf_db) * 1e-3
        self.slot_count = 0

    # ------------------------------------------------------------- trust ---
    def trust(self, viewer=None):
        """Trust vector seen by the defender (viewer = requester id for direct)."""
        if self.trust_mode == 'global':
            return self.rep
        if self.trust_mode == 'beta':
            return (self.ba + 1) / (self.ba + self.bb + 2)
        if self.trust_mode == 'direct':
            row = self.N if viewer is None else viewer
            a0, b0 = self.c_init_a(), self.c_init_b()
            return (self.dir_a[row] + a0) / (self.dir_a[row] + self.dir_b[row] + a0 + b0)
        return np.full(self.N, 0.5)

    def c_init_a(self):
        return 2 * self.cfg.rep_init

    def c_init_b(self):
        return 2 * (1 - self.cfg.rep_init)

    def level(self, t):
        L = self.cfg.rep_levels
        return np.clip((np.asarray(t) * L).astype(int), 0, L - 1)

    def att_state(self):
        """Reputation level that each attacker perceives about itself."""
        if self.trust_mode == 'none':
            return np.zeros(self.N, int)
        return self.level(self.trust(None)) + self.cfg.rep_levels * self.quar.astype(int)

    def _att_choose(self, ctx, s):
        c = self.cfg
        acts = self.ctx_actions[ctx]
        if self.rng.random() < c.att_eps:
            return int(self.rng.choice(acts))
        q = self.Q[ctx, s, acts]
        z = (q - q.max()) / c.att_temp
        p = np.exp(z); p /= p.sum()
        return int(acts[self.rng.choice(len(acts), p=p)])

    def untrusted(self, viewer=None):
        """Boolean vector of nodes treated as untrusted by `viewer`."""
        if self.trust_mode == 'none':
            return np.zeros(self.N, bool)
        if self.trust_mode == 'direct' and viewer is not None and viewer < self.N:
            return self.trust(viewer) < self.cfg.tau_q
        return self.quar

    def _update_quarantine(self):
        if self.trust_mode == 'none':
            return
        t = self.trust(self.N) if self.trust_mode == 'direct' else self.trust(None)
        self.quar = (self.quar | (t < self.cfg.tau_q)) & ~(t >= self.cfg.tau_r)

    # ---------------------------------------------------------- channels ---
    def _gain(self, d, nlos):
        c = self.cfg
        pl = c.pl0_db + 10 * c.pl_exp * np.log10(np.maximum(d, 0.5)) + \
            c.shadow_db * self.rng.standard_normal(np.shape(d)) + c.body_nlos_db * nlos
        fad = self.rng.exponential(1.0, np.shape(d))
        return db2lin(-pl) * fad

    def _evolve(self):
        c = self.cfg
        self.d_edge = np.clip(self.d_edge + self.rng.normal(0, 0.3, self.N), *c.d_edge)
        flip = self.rng.random(self.N) < 0.1
        self.nlos = np.where(flip, self.rng.random(self.N) < c.p_nlos, self.nlos)
        self.g_edge = self._gain(self.d_edge, self.nlos)
        self.g_hat = self.g_edge * np.abs(1 + np.sqrt(c.ce_err / 2) * (self.rng.standard_normal(self.N) + 1j * self.rng.standard_normal(self.N))) ** 2

    # -------------------------------------------------------------- slot ---
    def begin_slot(self):
        """Draw behaviours, tasks and observations. Returns the list of honest
        decision requests (dict) for the method's policy."""
        c = self.cfg
        rng = self.rng
        self._evolve()
        N = self.N
        # --- attacker decisions (network context: idle / forge / jam)
        self.s_att = self.att_state()
        self.decisions = []                      # (node, ctx, state, action, payoff)
        self.net_act = np.zeros(N, int)
        self.has_task = rng.random(N) < c.p_task
        self.req_act = np.zeros(N, int)
        for m in np.where(self.mal)[0]:
            self.net_act[m] = self._att_choose(0, self.s_att[m])
            if self.has_task[m]:
                self.req_act[m] = self._att_choose(1, self.s_att[m])
        # honest peers: occasional low-battery refusal
        self.low_batt = (~self.mal) & (rng.random(N) < c.p_honest_refuse)
        # --- jamming per ward
        self.jam_I = np.zeros(N)
        self.jammers = np.where(self.mal & (self.net_act == 4))[0]
        self.jam_eff = np.zeros(N, bool)
        tv_mec = self.trust(N) if self.trust_mode == 'direct' else self.trust(None)
        for j in self.jammers:
            same = self.ward == self.ward[j]
            d = rng.uniform(*c.d_jam, N)
            # reputation-triggered PHY countermeasure (channel hopping / AP-side
            # null steering) once the jammer is identified as untrusted
            supp = 0.01 if (self.gate and self.trust_mode != 'none' and self.quar[j]) else 1.0
            self.jam_eff[j] = supp == 1.0
            self.jam_I[same] += supp * db2lin(c.jam_dbm) * 1e-3 * self._gain(d, np.zeros(N, bool))[same]
        # --- tasks
        reqs = []
        tr_global = self.trust(None)
        self.task = {}
        for i in range(N):
            if not self.has_task[i]:
                continue
            idx = self.pid_idx[self.hub_pid[i]]
            k = idx[rng.integers(len(idx))]
            prio = 0 if self.pool['flag'][k] else (1 if self.icu[i] else 2)
            D = rng.uniform(*c.task_kbit) * 1e3 * c.task_scale
            C = D * rng.uniform(*c.cyc_per_bit)
            fake = bool(self.mal[i] and self.req_act[i] == 5)
            if fake:                      # deceptive resource grab
                prio, C, D = 0, C * 3.0, D
            peers = [p for p in np.where(self.ward == self.ward[i])[0] if p != i]
            if len(peers) < c.n_peers:
                peers = [p for p in range(N) if p != i]
            peers = list(rng.choice(peers, c.n_peers, replace=False))
            claim = np.array([c.f_peer * (rng.uniform(1.0, 1.5) if self.mal[p] else rng.uniform(0.6, 1.0)) for p in peers])
            g_peer = self._gain(rng.uniform(*c.d_peer, c.n_peers), rng.random(c.n_peers) < c.p_nlos)
            t = dict(i=i, k=k, y=int(self.pool['y'][k]), pred=int(self.pool['pred'][k]), prio=prio,
                     D=D, C=C, fake=fake, peers=peers, claim=claim, g_peer=g_peer)
            self.task[i] = t
            if not self.mal[i]:
                tv = self.trust(i) if self.trust_mode == 'direct' else tr_global
                ptrust = np.array([tv[p] for p in peers])
                unt = self.untrusted(i)
                t['ptrust'] = ptrust
                t['pok'] = np.array([not unt[p] for p in peers])
                t['obs'] = self._obs(t, ptrust, tr_global[i])
                m = np.ones(N_TGT, bool)
                if self.mask_on:
                    m[2:] = t['pok']
                t['mask'] = m
                reqs.append(t)
        return reqs

    def _obs(self, t, ptrust, own):
        c = self.cfg
        dl = c.deadline_s[t['prio']]
        o = np.zeros(OBS_DIM, np.float32)
        o[0] = t['D'] / 3e5
        o[1] = t['C'] / 3e8
        o[2] = dl
        o[3 + t['prio']] = 1
        o[6] = own if self.trust_mode != 'none' else 0.5
        o[7] = (10 * np.log10(self.g_hat[t['i']]) + 80) / 20
        o[8] = self.edge_load
        o[9] = float(self.icu[t['i']])
        o[10] = (self.jam_seen_ward[self.ward[t['i']]] if hasattr(self, 'jam_seen_ward') else 0.0)
        o[11] = c.bw_mhz / 20
        for j in range(c.n_peers):
            o[12 + 4 * j] = (10 * np.log10(t['g_peer'][j]) + 70) / 20
            o[13 + 4 * j] = ptrust[j] if self.trust_mode != 'none' else 0.5
            o[14 + 4 * j] = t['claim'][j] / c.f_peer
            o[15 + 4 * j] = float(t['pok'][j])
        return o

    # ---------------------------------------------------- rule baselines ---
    def rule_action(self, t):
        """Delay-greedy choice (greedy) or trust-filtered delay-greedy (BRS/DRP)."""
        c = self.cfg
        dl = c.deadline_s[t['prio']]
        P = db2lin(c.p_levels_dbm[-1]) * 1e-3
        n_edge = max(1.0, self.edge_load * self.N)
        W = c.bw_mhz * 1e6 / n_edge * 2
        r_e = W * np.log2(1 + P * self.g_hat[t['i']] / (W * self.noise_w_hz))
        est = np.full(N_TGT, np.inf)
        est[0] = t['C'] / c.f_local
        est[1] = t['D'] / r_e + t['C'] / (c.f_edge / n_edge)
        Wd = c.bw_d2d_mhz * 1e6 * c.bw_levels[-1]
        for j in range(c.n_peers):
            r = Wd * np.log2(1 + P * t['g_peer'][j] / (Wd * self.noise_w_hz))
            est[2 + j] = t['D'] / r + t['C'] / t['claim'][j]
        if self.trust_mode != 'none':
            est[2:][~t['pok']] = np.inf
            # prefer trustworthy peers: penalise by (1 - trust)
            est[2:] = est[2:] / np.maximum(t['ptrust'], 1e-3)
        feas = est <= dl
        tgt = int(np.argmin(est)) if not feas.any() else int(np.where(feas)[0][np.argmin(est[feas])])
        return np.array([tgt, N_POW - 1, N_BW - 1])

    # ------------------------------------------------------------- resolve ---
    def end_slot(self, reqs, actions):
        """Resolve the slot. actions: int array [n_req, 3]. Returns rewards and
        a dict of per-slot statistics."""
        c = self.cfg
        rng = self.rng
        N = self.N
        obs_list = []      # (node, risk level, observer)  observer: hub id or N (MEC)
        tr = self.trust(None)
        # ---- reputation-table tampering
        forged = np.zeros(N, bool)
        pay = np.zeros(N)                         # attacker payoff accumulated per node
        for m in np.where(self.mal)[0]:
            a0 = self.net_act[m]
            eff = False
            if a0 == 4:
                eff = bool(self.jam_eff[m])
            if a0 != 3:
                pay[m] = (c.gain[a0] if eff else 0.0) - c.cost[a0]
                self.decisions.append([m, 0, self.s_att[m], a0, pay[m]])
                continue
            if self.trust_mode in ('global', 'beta'):
                q = {'qkd': 2.0 ** (-c.mac_bits), 'classical': c.q_classic, 'none': c.q_none}[self.protect]
                if rng.random() < q:
                    forged[m] = True; eff = True
                    if self.trust_mode == 'global':
                        self.rep[m] = max(self.rep[m], 0.95)
                    else:
                        self.ba[m] += 3.0; self.bb[m] *= 0.5
                elif self.protect != 'none' and rng.random() < c.p_attr_forge:
                    obs_list.append((m, 3, N))
            self.decisions.append([m, 0, self.s_att[m], 3, (c.gain[3] if eff else 0.0) - c.cost[3]])
        # ---- jamming detection (AP-side SINR residual detector)
        jam_det = np.zeros(c.n_wards)
        for j in self.jammers:
            if rng.random() < c.p_det_jam:
                obs_list.append((j, 4, N)); jam_det[self.ward[j]] = 1
        self.jam_seen_ward = jam_det
        # ---- collect requests (honest decisions + malicious own/fake requests)
        jobs = []
        for t, a in zip(reqs, actions):
            jobs.append((t, int(a[0]), int(a[1]), int(a[2])))
        for i, t in self.task.items():
            if self.mal[i]:
                jobs.append((t, 1, N_POW - 1, N_BW - 1))   # selfish nodes always grab MEC
        # ---- MEC admission (zero-trust gate) and resource sharing
        edge_jobs = []
        for jb in jobs:
            t, tgt = jb[0], jb[1]
            if tgt != 1:
                continue
            i = t['i']
            ok = True
            if self.gate:
                if self.untrusted(N)[i]:
                    ok = False
            t['admitted'] = ok
            if ok:
                edge_jobs.append(jb)
        if edge_jobs:
            wsum_cpu = sum(c.prio_w[jb[0]['prio']] for jb in edge_jobs)
            wsum_bw = sum(c.bw_levels[jb[3]] for jb in edge_jobs)
        self.edge_load = 0.8 * self.edge_load + 0.2 * len(edge_jobs) / N
        # ---- peer assignment (one job per peer per slot, by priority)
        peer_req = {}
        for jb in jobs:
            t, tgt = jb[0], jb[1]
            if tgt >= 2:
                p = t['peers'][tgt - 2]
                peer_req.setdefault(p, []).append(jb)
        peer_winner = {}
        engaged = np.zeros(N, bool)
        for p, lst in peer_req.items():
            lst.sort(key=lambda jb: (jb[0]['prio'], rng.random()))
            peer_winner[p] = lst[0][0]['i']
            engaged[p] = True
        # ---- evaluate outcomes
        rewards = np.zeros(len(reqs))
        st = dict(n=0, ok=0, okw=0.0, w=0.0, emerg=0, emerg_viol=0, crit=0, crit_miss=0,
                  leak=0, wrong=0, energy=0.0, bits_ok=0.0, prio_n=np.zeros(3), prio_ok=np.zeros(3),
                  tgt=np.zeros(N_TGT), gated_honest=0, cmp_n=0, cmp_ok=0, all_w=0.0, all_okw=0.0)
        for jb in jobs:
            t, tgt, pw, bw = jb
            i = t['i']
            P = db2lin(c.p_levels_dbm[pw]) * 1e-3
            P = min(P, db2lin(c.p_max_dbm) * 1e-3)
            delay, energy = np.inf, 0.0
            delivered, wrong_res, leak = False, False, False
            if tgt == 0:
                delay = t['C'] / c.f_local
                energy = c.kappa * c.f_local ** 2 * t['C']
                delivered = True
            elif tgt == 1:
                if t.get('admitted', False):
                    W = c.bw_mhz * 1e6 * c.bw_levels[bw] / wsum_bw
                    sinr = P * self.g_edge[i] / (W * self.noise_w_hz + self.jam_I[i])
                    rate = W * np.log2(1 + sinr)
                    f = c.f_edge * c.prio_w[t['prio']] / wsum_cpu
                    delay = t['D'] / rate + t['C'] / f
                    energy = P * min(t['D'] / rate, c.deadline_s[t['prio']])   # radio aborts at the deadline
                    delivered = True
                    if t['fake']:
                        t['att_gain'] = c.gain[5]
                        if rng.random() < c.p_det_fake:
                            obs_list.append((i, 5, N))
                elif t['fake'] is False and not self.mal[i]:
                    st['gated_honest'] += 1
            else:
                p = t['peers'][tgt - 2]
                Wd = c.bw_d2d_mhz * 1e6 * c.bw_levels[bw]
                rate = Wd * np.log2(1 + P * t['g_peer'][tgt - 2] / (Wd * self.noise_w_hz))
                energy = P * min(t['D'] / rate, c.deadline_s[t['prio']])   # radio aborts at the deadline
                req_bad = self.untrusted(p)[i]
                if peer_winner.get(p) != i:
                    delay = np.inf                                   # helper busy
                elif not self.mal[p]:
                    if self.trust_mode != 'none' and self.gate and req_bad:
                        delay = np.inf                               # justified refusal (norm)
                        obs_list.append((p, -1, i))
                    elif self.low_batt[p]:
                        delay = np.inf; obs_list.append((p, 1, i))
                    else:
                        delay = t['D'] / rate + t['C'] / t['claim'][tgt - 2]; delivered = True
                        obs_list.append((p, 0, i))
                else:
                    s_p = self.s_att[p]
                    b = self._att_choose(2, s_p)           # helper context
                    f_act = c.f_peer * 0.8
                    u_h = 0.0 if b == 1 else -c.cost[b]
                    if b == 1:
                        delay = np.inf; obs_list.append((p, 1, i))
                    else:
                        delay = t['D'] / rate + t['C'] / f_act; delivered = True
                        if b == 2:
                            leak = True; u_h += c.gain[2]
                            obs_list.append((p, 2, i) if rng.random() < c.p_det_eav else (p, 0, i))
                        elif b == 5:
                            u_h += c.gain[5]
                            if rng.random() < c.p_det_spoof:
                                delivered = False; delay = np.inf; obs_list.append((p, 5, i))
                            else:
                                wrong_res = True; obs_list.append((p, 0, i))
                        else:
                            obs_list.append((p, 0, i))
                    self.decisions.append([p, 2, s_p, b, u_h])
            on_time = delivered and delay <= c.deadline_s[t['prio']]
            if self.mal[i]:
                u_r = t.get('att_gain', 0.0) - c.cost[self.req_act[i]]
                if not t['fake']:
                    u_r += c.serv_benefit * float(on_time)
                    # genuine ECG task of a patient whose device is compromised
                    okc = on_time and ((t['pred'] > 0) == (t['y'] > 0))
                    st['cmp_n'] += 1; st['cmp_ok'] += okc
                    st['all_w'] += c.prio_w[t['prio']]; st['all_okw'] += c.prio_w[t['prio']] * okc
                self.decisions.append([i, 1, self.s_att[i], self.req_act[i], u_r])
                continue
            # ---- clinical outcome for honest patients
            final = t['pred']
            if wrong_res:
                final = 0 if t['y'] > 0 else 1
            correct = (final > 0) == (t['y'] > 0)
            success = on_time and correct
            w = c.prio_w[t['prio']]
            st['n'] += 1; st['ok'] += success; st['w'] += w; st['okw'] += w * success
            st['all_w'] += w; st['all_okw'] += w * success
            st['prio_n'][t['prio']] += 1; st['prio_ok'][t['prio']] += on_time
            st['tgt'][tgt] += 1
            st['energy'] += energy
            st['leak'] += leak; st['wrong'] += wrong_res
            if on_time and tgt != 0:
                st['bits_ok'] += t['D']
            if t['prio'] == 0:
                st['emerg'] += 1; st['emerg_viol'] += (not on_time)
            if t['y'] > 0:
                st['crit'] += 1; st['crit_miss'] += not (on_time and final > 0)
            r = w * float(success) - c.c_fail * w * float(not on_time) - c.c_energy * energy / 0.05 \
                - c.c_priv * float(leak) - c.c_wrong * w * float(wrong_res and not correct)
            if self.lagr and t['prio'] == 0:
                r -= self.lam * float(not on_time)
            t['reward'] = r
        for n_, (t, a) in enumerate(zip(reqs, actions)):
            rewards[n_] = t['reward']
        # ---- probation audits of quarantined nodes (lets them rebuild trust)
        if self.gate and self.trust_mode == 'global':
            for m in np.where(self.untrusted(N))[0]:
                if rng.random() < c.p_probe and self.mal[m]:
                    b = self._att_choose(2, self.s_att[m])     # honeypot audit task
                    if b == 0:
                        obs_list.append((m, 0, N))
                    elif b == 1:
                        obs_list.append((m, 2, N))     # refusing an audit is treated as non-cooperation
                    elif rng.random() < c.p_det_audit:
                        obs_list.append((m, int(b), N))
                    else:
                        obs_list.append((m, 0, N))
                    self.decisions.append([m, 2, self.s_att[m], b, 0.0 if b == 1 else -c.cost[b]])
                elif rng.random() < c.p_probe:
                    obs_list.append((m, 0, N))           # honest node passes the audit
        # ---- reputation update
        self._update_trust(obs_list)
        self._update_quarantine()
        # ---- attacker payoff & Q-learning
        s_new = self.att_state()
        cnt = np.zeros((3, 6))
        U = np.zeros(N)
        for (m, ctx, s0, b, u) in self.decisions:
            U[m] += u
        # standing value V(s): TD(0) on the total per-slot payoff of each node
        for m in np.where(self.mal)[0]:
            s0, s1 = self.s_att[m], s_new[m]
            self.V[s0] += c.att_lr * (U[m] + c.att_gamma * self.V[s1] - self.V[s0])
        # action values per context: immediate payoff + discounted standing value
        for (m, ctx, s0, b, u) in self.decisions:
            target = u + c.att_gamma * self.V[s_new[m]]
            self.Q[ctx, s0, b] += c.att_lr * (target - self.Q[ctx, s0, b])
            cnt[ctx, b] += 1
        n_dec = cnt.sum()
        st['attack'] = float(cnt[:, 2:].sum() / max(n_dec, 1))
        st['n_dec'] = n_dec
        # per-type rates within the context(s) where the type is feasible
        st['att_type'] = np.array([
            cnt[2, 0] / max(cnt[2].sum(), 1),                       # cooperate (helper)
            cnt[2, 1] / max(cnt[2].sum(), 1),                       # refuse
            cnt[2, 2] / max(cnt[2].sum(), 1),                       # eavesdrop
            cnt[0, 3] / max(cnt[0].sum(), 1),                       # reputation forgery
            cnt[0, 4] / max(cnt[0].sum(), 1),                       # jamming
            (cnt[1, 5] + cnt[2, 5]) / max(cnt[1].sum() + cnt[2].sum(), 1)])   # deception / spoofing
        st['att_cnt'] = cnt
        trv = self.trust(N) if self.trust_mode == 'direct' else self.trust(None)
        st['rep_h'] = float(trv[~self.mal].mean())
        st['rep_m'] = float(trv[self.mal].mean()) if self.mal.any() else 0.0
        st['forged'] = int(forged.sum())
        self.slot_count += 1
        return rewards, st

    def _update_trust(self, obs_list):
        c = self.cfg
        for (n, l, obsr) in obs_list:
            e = 1.0 if l <= 0 else (1.0 - l / 5.0) ** 2      # convex severity map (Table I)
            good = 1.0 if l <= 0 else (0.5 if l == 1 else 0.0)
            if self.trust_mode == 'global':
                if self.classify:
                    th = c.theta[max(l, 0)]
                    if l <= 0 and self.quar[n]:
                        th = c.theta_recover          # slow forgiveness while quarantined
                else:
                    th = c.theta_uniform
                self.rep[n] = th * self.rep[n] + (1 - th) * e
            elif self.trust_mode == 'beta':
                self.ba[n] = 0.98 * self.ba[n] + good
                self.bb[n] = 0.98 * self.bb[n] + (1 - good)
            elif self.trust_mode == 'direct':
                self.dir_a[obsr, n] = 0.98 * self.dir_a[obsr, n] + good
                self.dir_b[obsr, n] = 0.98 * self.dir_b[obsr, n] + (1 - good)
