"""Default simulation parameters (Table II of the paper)."""
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # ---- population / topology
    n_hubs: int = 40                 # WBAN hubs (patients)
    n_wards: int = 4                 # clinical zones (ICU, two wards, home-care cell)
    mal_frac: float = 0.25           # fraction of compromised / selfish nodes
    icu_frac: float = 0.25           # fraction of ICU-acuity patients
    n_peers: int = 3                 # candidate D2D helpers per request
    # ---- time
    slots: int = 100                 # slots per episode
    slot_s: float = 0.1              # slot duration (s)
    p_task: float = 0.5              # task arrival probability per hub per slot
    # ---- task (ECG + multi-parameter window)
    task_kbit: tuple = (80.0, 320.0)  # payload range (kbit)
    task_scale: float = 1.0          # task-size multiplier (sweeps)
    cyc_per_bit: tuple = (600.0, 1000.0)
    deadline_s: tuple = (0.10, 0.30, 1.00)   # emergency / urgent / routine
    prio_w: tuple = (4.0, 2.0, 1.0)
    # ---- computing
    f_local: float = 0.6e9           # hub CPU share (cycles/s)
    f_peer: float = 2.0e9            # bedside / peer idle CPU
    f_edge: float = 50e9             # hospital MEC server
    kappa: float = 1e-27             # effective switched capacitance
    # ---- radio (6G small cell / WBAN off-body link)
    bw_mhz: float = 20.0             # shared uplink bandwidth to the AP
    bw_d2d_mhz: float = 2.0          # per D2D link (x share level)
    p_levels_dbm: tuple = (5.0, 12.0, 20.0)   # SAR-compliant hub transmit powers
    p_max_dbm: float = 20.0
    bw_levels: tuple = (1.0, 2.0, 4.0)        # bandwidth share weights
    n0_dbm_hz: float = -174.0
    nf_db: float = 9.0
    pl0_db: float = 40.0             # 1-m reference loss (2.4-6 GHz)
    pl_exp: float = 2.6
    shadow_db: float = 4.0
    body_nlos_db: float = 12.0       # extra body shadowing in NLOS posture
    p_nlos: float = 0.3
    ce_err: float = 0.05             # channel-estimation error variance (rel.)
    d_edge: tuple = (3.0, 25.0)
    d_peer: tuple = (1.0, 8.0)
    jam_dbm: float = 15.0
    d_jam: tuple = (4.0, 15.0)
    # ---- detection / audit
    p_det_eav: float = 0.25          # canary-token detection of PHI exfiltration
    p_det_spoof: float = 0.55        # result audit / plausibility check
    p_det_fake: float = 0.50         # fake-request verification at the MEC
    p_det_jam: float = 0.70          # SINR-residual anomaly detector + localisation
    p_det_audit: float = 0.60       # detection of misbehaviour on a honeypot audit
    p_attr_forge: float = 0.90       # attribution of a failed reputation tamper
    # ---- reputation
    rep_init: float = 0.7
    rep_levels: int = 6
    theta: tuple = (0.90, 0.85, 0.70, 0.60, 0.55, 0.50)   # forgetting per risk level
    theta_uniform: float = 0.85
    tau_q: float = 0.45              # zero-trust quarantine threshold
    tau_r: float = 0.60              # re-admission threshold (hysteresis)
    theta_recover: float = 0.80      # forgetting while quarantined (slow forgiveness)
    p_probe: float = 0.50            # probation-audit probability per quarantined node/slot
    p_honest_refuse: float = 0.05    # low-battery refusals of honest helpers
    # ---- reputation-table protection
    q_classic: float = 0.3           # P[forgery succeeds] vs classical PKI under a CRQC adversary
    q_none: float = 0.95
    mac_bits: int = 64
    # ---- attacker learning (Q-learning on own reputation level)
    att_lr: float = 0.05
    att_gamma: float = 0.95
    att_temp: float = 0.10
    att_eps: float = 0.02
    att_q0: float = 0.0             # optimistic initial Q-value
    gain: tuple = (0.0, 0.0, 1.0, 0.6, 1.2, 1.6)  # illicit gain per behaviour
    cost: tuple = (0.2, 0.0, 0.25, 0.3, 0.4, 0.3)  # effort cost per behaviour
    serv_benefit: float = 3.0
    # ---- reward shaping for the defender
    c_energy: float = 0.15
    c_fail: float = 0.5
    c_priv: float = 1.0
    c_wrong: float = 1.0
    delta_emerg: float = 0.05        # URLLC target: P[emergency violation] <= delta

    def asdict(self):
        return asdict(self)
