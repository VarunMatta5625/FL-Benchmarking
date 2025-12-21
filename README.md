# Differential Privacy in Federated Learning: Sent140 + EMNIST

This repository is a research-style implementation suite for studying **differential privacy (DP)** in **federated learning (FL)** across two common benchmarks: **Sent140** (text sentiment) and **EMNIST** (handwritten characters). The code explores multiple privacy mechanisms and aggregation strategies, compares privacy–utility tradeoffs, and logs results as CSVs and plots.

The focus is not a single production pipeline, but a collection of experiments that answer: *How do different DP mechanisms affect accuracy, convergence, and privacy loss in FL?*

---

## What This Project Accomplishes

The project implements a family of FL training loops, each with a different privacy mechanism or aggregation strategy. By running the scripts on Sent140 or EMNIST, you can:

- Measure accuracy vs. privacy loss (epsilon) across rounds.
- Compare centralized DP, local DP, and federated DP variants.
- Observe the effect of clipping and noise on convergence.
- Generate logs and plots that summarize privacy–utility tradeoffs.

Each script is a self-contained experiment or a trainer compatible with the unified runner.

---

## How It Works (High Level)

All experiments follow a similar pattern:

1) **Client sampling**: select a subset of clients for each round.  
2) **Local training**: each selected client trains a local model or produces gradients.  
3) **Privacy mechanism**: apply clipping and noise (or output perturbation).  
4) **Aggregation**: average gradients or model updates to form the global update.  
5) **Evaluation**: compute accuracy and log privacy metrics.  

The key differences across algorithms are where the noise is added (gradient vs. output), whether updates are centralized or local, and how aggregation is performed.

---

## Entry Points

### Unified Runner (Config-Driven)

`run_experiment.py` dynamically imports a trainer from `training/` based on `--algo` and runs it with a JSON config.

```bash
python run_experiment.py --algo dp_fedavg --dataset sent140 --config configs/sent140_dp_fedavg.json
```

This assumes a `configs/` folder and a `utils/` package exist (see **Known Gaps**).

### Standalone Scripts

Some scripts embed their own data loading, model creation, and logging and can be run directly:

```bash
python training/train_fedsyn.py
```

---

## Algorithms Implemented (Research Summary)

This is a conceptual overview of what each family does. The file names in `training/` map directly.

### DP-FedAvg (Sent140 / EMNIST)
Implements federated averaging with **gradient clipping** and **Gaussian noise** added to client gradients before aggregation.  
Files: `training/train_dp_fedavg.py`, `training/train_dp_fedavg_emnist.py`

### DP-FTRL (Sent140 / EMNIST)
Uses a variant of Follow-the-Regularized-Leader updates with gradient clipping and noise for privacy.  
Files: `training/train_dp_ftrl.py`, `training/train_dp_ftrl_emnist.py`

### Output Perturbation (Sent140 / EMNIST)
Trains client models locally and adds noise directly to model parameters before averaging.  
Files: `training/train_dp_outputperturbation.py`, `training/train_outputperturbation_dp_fl_emnist.py`

### FedSyn + DP
Generates synthetic gradients from local updates with clipping and noise, then aggregates.  
Files: `training/train_dp_fedsyn.py`, `training/train_fedsyn.py`, `training/train_fedsyn_emnist.py`

### Bayesian DP-FL
Uses Bayesian-style local updates with DP noise to study posterior-like effects on privacy–utility.  
Files: `training/train_bayesian_dpfl.py`, `training/train_bayesian_dp_fl_emnist.py`

### Central DP vs Local DP
Compares central DP (noise after aggregation) vs local DP (noise at client).  
Files: `training/train_centraldp.py`, `training/train_localdp.py`, `training/train_central_dp_fl_emnist.py`, `training/train_local_dp_fl_emnist.py`

### RAPPOR-Style Local DP
Explores randomized response / RAPPOR-like mechanisms on client updates.  
Files: `training/train_rappor.py`, `training/train_rappor_emnist.py`

### Secure Aggregation + DP
Simulates secure aggregation with DP noise on federated updates.  
Files: `training/train_secureagg_dp.py`, `training/train_secureagg_dp_emnist.py`

### Strong / Weak FedFDP
Studies stronger vs weaker privacy constraints in federated DP variants.  
Files: `training/train_dp_strongfedfdp.py`, `training/train_strongfedfdp_emnist.py`, `training/train_weakfedfdp.py`, `training/train_weakfedfdp_emnist.py`

### DP-XGBoost (EMNIST + non-EMNIST)
Implements DP variants of gradient-boosted trees for FL experiments.  
Files: `training/train_dpxgb.py`, `training/train_dp_xgb_emnist.py`

---

## Data Overview (Lightweight)

### EMNIST
- Raw EMNIST files and processed tensors live under `data/emnist/`.  
- Client splits are serialized in `data/emnist/clients/`.  

### Sent140
- Raw CSVs and ZIPs are under `data/sent140/`.  
- Per-client JSON splits are in `data/sent140/clients/`.  

---

## Outputs and Logs

Experiments write metrics to `logs/` (and `logs/EMNIST/`). Typical outputs:

- CSVs with per-round accuracy, loss, and epsilon.
- PNG plots for accuracy vs rounds and privacy vs accuracy.

---

## Known Gaps (Current Repo Snapshot)

- **Missing `utils/` package**: several scripts import `utils.*` (model loaders, DP utilities, logging). These modules are not present here and must be restored for config-driven runs.
- **Config files missing**: `run_experiment.py` expects JSON configs in `configs/`.  
- **Requirements incomplete**: some scripts need `pandas` and `scikit-learn`, which are not listed in `requirements.txt`.

---

## Research Notes / Intended Usage

This project is best used as a research sandbox to compare DP mechanisms under controlled settings. The implementations are intentionally simple to make algorithmic differences visible and comparable across datasets.

If you want to turn this into a clean, reproducible research repo, the next steps are:

1) Restore or rewrite the missing `utils/` package.  
2) Add `configs/` with a baseline config per algorithm/dataset.  
3) Normalize logging and plotting for consistent output.  
4) Add a small experiments index (e.g., `docs/experiments.md`) with example runs and plots.
