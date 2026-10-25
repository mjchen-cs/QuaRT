# QuaRT — Learning-based Quantum-Secured Reputation for Deception-Resilient WBANs (Healthcare 5.0)

Reference implementation of every experiment in the paper
*"Learning-Based Quantum-Secured Reputation for Deception-Resilient Wireless Body Area Networks in Healthcare 5.0"* (IEEE J-BHI special issue on Advanced Wireless Networks for Healthcare 5.0).

The code runs with **NumPy / SciPy / Matplotlib only** (no deep-learning framework, no GPU).
`wfdb` is optional and only needed to use the real MIT-BIH Arrhythmia Database.

## Layout

```
quart/
  nn.py         NumPy MLP + Adam (from scratch)
  mitdb_pipeline.py  two-stage, inter-patient MIT-BIH arrhythmia analytics
  ecg.py        ECG workload: synthetic ECGSYN-style generator, MIT-BIH loader (wfdb),
                Pan–Tompkins-style QRS detection, on-hub triage, analytics MLP
  cvqkd.py      QPSK CV-QKD key rate: analytic CM route + Fock-space route with
                quantum scissors (self-test checks that the two routes agree)
  config.py     all simulation parameters (Table II of the paper)
  env.py        Healthcare-5.0 WBAN/MEC simulator: body-shadowed channels, jamming,
                clinical priorities/deadlines, six-level threat taxonomy, adaptive
                Q-learning adversary, reputation + zero-trust gate, registry protection
  agents.py     trust-masked Lagrangian PPO (QuaRT) and Double-DQN baseline
  runner.py     training / evaluation loops
run_experiments.py   main comparison, sweeps, classical-forgery sweep, compromised-fraction sweep
qkd_experiments.py   key-rate vs distance, key budget, ECG analytics metrics
make_figures.py      all result figures (paper_figures/) and LaTeX tables (paper_tables/)
figures_src/         TikZ sources of the schematic figures (system, framework, CV-QKD)
run_all.sh           full pipeline
```

## Methods

| name | description |
|---|---|
| `quart` | **QuaRT**: trust-masked PPO + Lagrangian URLLC constraint + risk-classified reputation + zero-trust gate with hysteresis/probation + QKD-protected registry |
| `quart_c` | QuaRT with a classically signed registry (forgeable with prob. `q_classic`) |
| `quart_u` | QuaRT with a single (unclassified) forgetting factor |
| `quart_nomask`, `quart_nolag` | ablations |
| `dqn_r` | Double-DQN with reputation features (classical registry) |
| `ppo_nr` | vanilla PPO without reputation |
| `brs` | beta reputation system + trust-filtered delay-greedy rule |
| `drp` | direct reciprocity (private experience only) + rule |
| `greedy` | delay-greedy, trust-agnostic |

## Reproduce

```bash
pip install numpy scipy matplotlib        # optional: pip install wfdb
python -m quart.cvqkd                     # numerical self-test of the QKD routes
bash run_all.sh                           # ~3-4 h on 2 CPU cores
```

### ECG workload (MIT-BIH)
```bash
pip install wfdb
python download_mitdb.py      # resumable; re-run until 48/48 records are complete
python eval_ecg.py            # builds results/ecg_pool.npz and prints the analytics metrics
```
When `data/mitdb` exists, `quart/ecg.py` uses the two-stage MIT-BIH pipeline in
`quart/mitdb_pipeline.py`: a beat classifier (N/S/V, de Chazal-style RR and morphology features)
followed by a window classifier (NSR/AF/PVC/VT-run), evaluated with two-fold inter-patient
cross-validation (DS1 <-> DS2, paced records excluded). Without the data it falls back to a
synthetic ECGSYN-style corpus.

On Windows/PowerShell, run the commands of `run_all.sh` one by one, and set
`$env:OMP_NUM_THREADS=1` before launching several processes.
