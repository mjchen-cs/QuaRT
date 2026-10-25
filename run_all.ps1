# Full reproduction pipeline (Windows PowerShell). Usage: powershell -File run_all.ps1 [n_procs]
param([int]$P = 2)
$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = 1
function Run { param([string[]]$a) & python @a; if ($LASTEXITCODE -ne 0) { throw "failed: python $a" } }
Run download_mitdb.py
Run eval_ecg.py
Run run_experiments.py, main, --n_hubs, 40, --seeds, 0, 1, 2, --procs, $P
Run run_experiments.py, main, --n_hubs, 80, --seeds, 0, 1, 2, --procs, $P
Run run_experiments.py, sweep, --n_hubs, 40, --procs, $P
Run run_experiments.py, sweep, --n_hubs, 80, --procs, $P
Run run_experiments.py, qclass, --seeds, 0, 1, 2, --procs, $P
Run run_experiments.py, malfrac, --seeds, 0, 1, --procs, $P
Run qkd_experiments.py
Run make_figures.py
