# QuaRT: Learning-Based Quantum-Secured Reputation for Deception-Resilient WBANs in Healthcare 5.0

This repository is the official implementation of the paper

> **Learning-Based Quantum-Secured Reputation for Deception-Resilient Wireless Body Area Networks in Healthcare 5.0**
> Miaojiang Chen, Meng Yi, Saif Al-Kuwari, H. Herbert Song, Ahmed Farouk.
> Submitted to *IEEE Journal of Biomedical and Health Informatics* (J-BHI), Special Issue on Advanced Wireless Networks for Healthcare 5.0.

QuaRT schedules ECG analytics tasks from wireless body area network (WBAN) hubs to a hospital MEC server or to bedside helper devices. Some of those devices are compromised insiders that adapt their attacks to their own reputation. QuaRT has three parts:

1. **Risk-classified indirect-reciprocity reputation with a zero-trust gate.** Quarantine uses hysteresis, and a quarantined device can return only through probation audits.
2. **A CV-QKD-protected reputation registry.** Registry updates use QPSK CV-QKD keys, an optional quantum-scissors amplifier, one-time-pad encryption and Wegman–Carter authentication. This keeps a quantum-capable adversary from forging reputations.
3. **Trust-masked, URLLC-constrained PPO.** A factorised policy chooses the execution target, transmit power and bandwidth. A Lagrangian constraint enforces the emergency deadline.

The clinical workload comes from the **MIT-BIH Arrhythmia Database**. A two-stage beat → window arrhythmia classifier is evaluated inter-patient on DS1 ↔ DS2.

Everything is implemented in **NumPy, SciPy and Matplotlib**. No deep-learning framework and no GPU are needed.

## Repository layout

```
quart/
  nn.py              NumPy MLP + Adam (from scratch)
  config.py          all simulation parameters (Table II of the paper)
  env.py             Healthcare-5.0 WBAN/MEC simulator: body-shadowed channels, jamming,
                     clinical priorities/deadlines, six-level threat taxonomy, adaptive
                     Q-learning adversary, reputation + zero-trust gate, registry protection
  agents.py          trust-masked Lagrangian PPO (QuaRT) and Double-DQN baseline
  runner.py          training / evaluation loops and rule-based baselines
  cvqkd.py           QPSK CV-QKD key rate: analytic covariance-matrix route and Fock-space
                     route with quantum scissors (self-test checks that they agree)
  ecg.py             ECG task pool, on-hub triage (QRS detection + rate/RR rules),
                     synthetic ECGSYN-style fallback corpus
  mitdb_pipeline.py  two-stage inter-patient MIT-BIH arrhythmia analytics
download_mitdb.py    resumable download of the MIT-BIH Arrhythmia Database
eval_ecg.py          builds results/ecg_pool.npz and prints the analytics metrics
run_experiments.py   network experiments (main comparison, sweeps, forgery sweep,
                     compromised-fraction sweep)
qkd_experiments.py   key rate vs. distance and registry key budget
make_figures.py      all result figures (paper_figures/) and LaTeX tables (paper_tables/)
figures_src/         TikZ sources of the schematic figures (Figs. 1-3)
results/             raw results used in the paper (JSON) and the ECG task pool
paper_figures/       figures as they appear in the paper (PDF + PNG)
paper_tables/        tables as they appear in the paper (LaTeX)
```

## Installation

Python ≥ 3.9 is required.

```bash
git clone https://github.com/<your-account>/QuaRT.git
cd QuaRT
pip install -r requirements.txt
python -m quart.cvqkd        # numerical self-test of the two CV-QKD routes
```

## Reproducing the paper

### Option A: regenerate figures and tables from the included results (about 1 min)

```bash
python make_figures.py
```

### Option B: full reproduction

**1. Download MIT-BIH** (about 90 MB, from PhysioNet). The download can be resumed; re-run the script until it reports 48/48 records:

```bash
python download_mitdb.py
```

**2. Build the ECG task pool and print the analytics metrics** (Table III):

```bash
python eval_ecg.py
```

**3. Run the network experiments, the QKD experiments and the figures:**

```bash
bash run_all.sh                       # Linux / macOS
powershell -File run_all.ps1          # Windows
```

Each job is single-threaded. Use `--procs` to choose the number of parallel jobs; the default is 2. With 2 processes the full pipeline takes about 3–4 h. On a 32-core machine it finishes in well under 1 h.

Set `OMP_NUM_THREADS=1` (on Windows, `$env:OMP_NUM_THREADS=1`) to stop NumPy from oversubscribing the cores.

The individual steps are:

```bash
python run_experiments.py main    --n_hubs 40 --seeds 0 1 2 --procs 8   # Table IV, Figs. 6-9 (N=40) + ablations
python run_experiments.py main    --n_hubs 80 --seeds 0 1 2 --procs 8   # N=80
python run_experiments.py sweep   --n_hubs 40 --procs 8                 # bandwidth / task-size / power sweeps
python run_experiments.py sweep   --n_hubs 80 --procs 8
python run_experiments.py qclass  --seeds 0 1 2 --procs 8               # classical forgery probability sweep
python run_experiments.py malfrac --seeds 0 1 --procs 8                 # compromised-device fraction sweep
python qkd_experiments.py                                               # Fig. 4 (key rate, key budget)
python make_figures.py
```

`sweep` reloads the trained models from `results/models/`, so run it after the corresponding `main` run. The trained models are not included in this repository because the DQN models are about 14 MB each.

If `data/mitdb` is absent, `quart/ecg.py` falls back to a synthetic ECGSYN-style corpus. The code still runs end to end in that case, but the numbers will differ from the paper.

## Methods

| name | description |
|---|---|
| `quart` | **QuaRT**: trust-masked PPO + Lagrangian URLLC constraint + risk-classified reputation + zero-trust gate (hysteresis, probation) + QKD-protected registry |
| `quart_c` | QuaRT with a classically signed registry (forgeable with probability `q_classic`) |
| `quart_u` | QuaRT with a single (unclassified) forgetting factor |
| `quart_nomask`, `quart_nolag` | ablations without the trust mask / the Lagrangian constraint |
| `dqn_r` | Double DQN with reputation features (classical registry) |
| `ppo_nr` | PPO without reputation |
| `brs` | Bayesian beta reputation system + trust-filtered delay-greedy rule |
| `drp` | direct reciprocity (private experience only) + delay-greedy rule |
| `greedy` | delay-greedy, trust-agnostic |

## Main results (N = 40 WBANs, 25% compromised, mean over 3 seeds)

| | QuaRT | DRP | DQN-R | BRS |
|---|---|---|---|---|
| Attack rate (%) | **2.8** | 11.1 | 69.7 | 90.3 |
| Weighted success, all patients (%) | **94.7** | 73.9 | 69.7 | 33.2 |
| Emergency deadline violation (%) | **2.9** | 2.0 | 31.6 | 76.9 |

ECG analytics on MIT-BIH, inter-patient over 6,459 windows: accuracy 90.0%, alarm sensitivity 96.1%, specificity 97.8%.

See the paper for the complete tables.

## Data

The MIT-BIH Arrhythmia Database is provided by PhysioNet under the [Open Data Commons Attribution License v1.0](https://physionet.org/content/mitdb/1.0.0/). `results/ecg_pool.npz` holds the derived labels and model predictions for every 10-s window, plus four example 10-s ECG excerpts used in Fig. 5. If you use the data, please also cite:

- G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," *IEEE Eng. Med. Biol. Mag.*, 20(3):45–50, 2001.
- A. L. Goldberger *et al.*, "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation*, 101(23):e215–e220, 2000.

## Citation

```bibtex
@article{chen2026quart,
  author  = {Chen, Miaojiang and Yi, Meng and Al-Kuwari, Saif and Song, H. Herbert and Farouk, Ahmed},
  title   = {Learning-Based Quantum-Secured Reputation for Deception-Resilient Wireless Body Area Networks in Healthcare 5.0},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {under review}
}
```

## License

The code is released under the [MIT License](LICENSE).
