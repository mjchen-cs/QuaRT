#!/usr/bin/env bash
# Full reproduction pipeline. Usage: bash run_all.sh [n_procs]
set -e
P=${1:-2}
export OMP_NUM_THREADS=1
python download_mitdb.py
python eval_ecg.py
python run_experiments.py main    --n_hubs 40 --seeds 0 1 2 --procs $P
python run_experiments.py main    --n_hubs 80 --seeds 0 1 2 --procs $P
python run_experiments.py sweep   --n_hubs 40 --procs $P
python run_experiments.py sweep   --n_hubs 80 --procs $P
python run_experiments.py qclass  --seeds 0 1 2 --procs $P
python run_experiments.py malfrac --seeds 0 1 --procs $P
python qkd_experiments.py
python make_figures.py
