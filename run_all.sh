#!/bin/bash
# Full reproduction pipeline (approx. 3 h on 2 CPU cores)
set -e
python3 run_experiments.py main --n_hubs 40 --seeds 0 1 2
python3 run_experiments.py main --n_hubs 80 --seeds 0 1 2
python3 run_experiments.py sweep --n_hubs 40
python3 run_experiments.py sweep --n_hubs 80
python3 run_experiments.py qclass --seeds 0 1 2
python3 run_experiments.py malfrac --seeds 0 1
python3 qkd_experiments.py
python3 make_figures.py
