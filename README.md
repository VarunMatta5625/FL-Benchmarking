# FL-Benchmarking: Differential Privacy in Federated Learning

A benchmarking suite that compares **differential privacy (DP) mechanisms in federated learning (FL)** on two standard benchmarks:

- **Sent140** — binary tweet sentiment (1.6M tweets, TF-IDF features, chance ≈ 50%)
- **EMNIST Balanced** — 47-class handwritten character recognition (chance ≈ 2.1%)

Each script in `training/` is a self-contained experiment implementing one privacy mechanism (gradient noise, output perturbation, randomized response, secure-aggregation simulation, …). Every run logs per-round accuracy and a privacy-loss estimate (ε) to `logs/`, plus PNG plots of accuracy-vs-rounds and accuracy-vs-ε.

The question the suite answers: **how much utility does each DP mechanism cost, and where in the pipeline is noise least damaging?**

---

## Repository layout

```
├── run_experiment.py        # config-driven runner (currently broken — see Known issues)
├── training/                # one self-contained script per (algorithm × dataset)
├── scripts/
│   └── prepare_emnist.py    # builds the EMNIST client shards the scripts expect
├── data/                    # git-ignored; see data/README.md
│   ├── emnist/              # built by scripts/prepare_emnist.py
│   └── sent140/             # downloaded manually (Sentiment140 CSV)
└── logs/                    # results: CSV metrics + PNG plots
    ├── *.csv, *.png         # Sent140 results
    └── EMNIST/              # EMNIST results
```

---

## Order of execution

Run everything from the **repo root**.

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare data

```bash
# EMNIST: downloads via torchvision and builds 50 IID client shards
python scripts/prepare_emnist.py

# Sent140: download the CSV manually (see data/README.md) and place it at
#   data/sent140/training.1600000.processed.noemoticon.csv
```

### 3. Run experiments

Each script is standalone — pick the algorithm/dataset you want:

```bash
python training/train_dp_fedavg_emnist.py     # example: DP-FedAvg on EMNIST
python training/train_centraldp.py            # example: central DP on Sent140
```

### 4. Inspect results

Each run writes a CSV (per-round `Round, Accuracy, Epsilon`) and PNG plots to `logs/` (Sent140) or `logs/EMNIST/` (EMNIST).

---

## Script index

### EMNIST (all expect `data/emnist/` from step 2)

| Script | Mechanism | Output CSV (`logs/EMNIST/`) |
|---|---|---|
| `train_dp_fedavg_emnist.py` | FedAvg + per-client update clipping & Gaussian noise | `dp_fedavg_metrics.csv` |
| `train_dp_ftrl_emnist.py` | DP-FTRL (noisy accumulated updates) | `dp_ftrl_metrics.csv` |
| `train_central_dp_fl_emnist.py` | Central DP (noise added after aggregation) | `central_dp_metrics.csv` |
| `train_local_dp_fl_emnist.py` | Local DP (noise at each client) | `local_dp_metrics.csv` |
| `train_secureagg_dp_emnist.py` | Secure-aggregation simulation + DP | `mpc_fl_metrics.csv` |
| `train_pldp_fl_emnist.py` | Personalised local DP (per-client ε) | `pldp_fl_metrics.csv` |
| `train_strongfedfdp_emnist.py` / `train_weakfedfdp_emnist.py` | FedFDP with strong/weak privacy budget | `strongfedfdp_metrics.csv` / `weakfedfdp_metrics.csv` |
| `train_rappor_emnist.py` | RAPPOR-style randomized response on labels | `rappor_metrics.csv` |
| `train_bayesian_dp_fl_emnist.py` | Bayesian DP-FL, swept over ε | `bayesian_dp_fl_emnist.csv` |
| `train_outputperturbation_dp_fl_emnist.py` | Output perturbation, swept over ε | `outputperturbation_dp_fl_emnist.csv` |
| `train_fedsyn_emnist.py` | FedSyn (synthetic-gradient aggregation) | `fedsyn_emnist_results.csv` |
| `train_dp_xgb_emnist.py` | "DP"-XGBoost (see Known issues) | `dp_xgb_emnist_results.csv` |

### Sent140 (all expect the Sentiment140 CSV from step 2)

| Script | Mechanism | Output CSV (`logs/`) |
|---|---|---|
| `train_centraldp.py` | Central DP | `centraldp_metrics.csv` |
| `train_localdp.py` | Local DP | `localdp_metrics.csv` |
| `train_secureagg_dp.py` | Secure-aggregation simulation + DP | `secureagg_metrics.csv` |
| `train_pldp_fl.py` | Personalised local DP | `pldpfl_metrics.csv` |
| `train_weakfedfdp.py` | FedFDP (weak budget) | `weakfedfdp_metrics.csv` |
| `train_rappor.py` | RAPPOR-style randomized response on TF-IDF features | `rappor_metrics.csv` |
| `train_bayesian_dpfl.py` | Bayesian DP-FL | `bayesian_dpfl_metrics.csv` |
| `train_fedsyn.py` | FedSyn | `fedsyn_metrics.csv` |
| `train_dpxgb.py` | "DP"-XGBoost (see Known issues) | `dpxgb_metrics.csv` |

### Broken / legacy (import a `utils/` package not present in the repo)

`run_experiment.py`, `train_dp_fedavg.py`, `train_dp_ftrl.py`, `train_dp_fedsyn.py`, `train_dp_outputperturbation.py`, `train_dp_strongfedfdp.py` — these were driven by a config-based runner whose `utils/` helpers (data loaders, DP utils, logger) were never committed. Their historical outputs survive in `logs/dp_*_sent140.csv`. The standalone scripts above cover the same algorithms.

---

## Results: what the logs show

Numbers below are read directly from the CSVs in `logs/`. "Best" is the best test accuracy over the run; ε values are the scripts' own (simplified) accounting — see the caveats section.

### Sent140 (binary; coin-flip = 50%)

| Method | Best acc | Final acc | Reading |
|---|---|---|---|
| FedAvg, no/weak DP (`fedavg_metrics.csv`, legacy) | **82.5%** | 68.1% | Utility ceiling for this model/feature setup. |
| Output perturbation (`outputperturbation_metrics.csv`, legacy) | **74.1%** | 74.1% | Best private result — noise on final weights, at low logged ε (≤6), retains almost all utility. |
| DP-FTRL (`dpftrl_metrics.csv`, legacy) | 73.6% | 53.0% | Learns well, then accumulated noise degrades it; best at round 56 of 100. |
| "DP"-XGBoost (`dpxgb_metrics.csv`) | 70.7% | 70.7% | Flat across all ε because the DP flags are silently ignored (see Known issues) — effectively a non-private baseline. |
| FedSyn (`fedsyn_metrics.csv`) | 60.7% | 55.9% | Modest learning over 30 rounds. |
| WeakFedFDP (`weakfedfdp_metrics.csv`) | 62.0% | 54.1% | Weak privacy budget preserves some utility. |
| Central DP (`centraldp_metrics.csv`) | 57.1% | 53.6% | Starts above chance, decays as ε accumulates — noise slowly erodes the model. |
| StrongFedFDP (`strongfdp_metrics.csv`, legacy) | 55.5% | 52.1% | Stronger budget ⇒ measurably worse than WeakFedFDP, as expected. |
| Bayesian DP-FL (`bayesian_dpfl_metrics.csv`) | 51.8% | 49.9% | Never escapes chance — noise overwhelms the signal. |
| PLDP-FL (`pldpfl_metrics.csv`) | 50.9% | 50.2% | Chance level throughout. |
| RAPPOR (`rappor_metrics.csv`) | 50.0% | 50.0% | Randomizing the TF-IDF **input features** destroys the signal entirely — pure coin-flip. |

Runner-era logs (`dp_*_sent140.csv`, 5 rounds each): DP-FedAvg peaked at 66.5%, DP-FTRL at 54.3%, output perturbation started at 60.4% and decayed; `dp_fedsyn_sent140.csv` shows ~0% accuracy with rising loss — that run's evaluation was broken and it should be disregarded.

### EMNIST Balanced (47 classes; chance ≈ 2.1%)

| Method | Best acc | Final acc | Reading |
|---|---|---|---|
| "DP"-XGBoost (`dp_xgb_emnist_results.csv`) | **80.2%** | 80.2% | Identical at every ε — DP flags ignored (see Known issues); treat as a non-private ceiling. |
| RAPPOR on labels (`rappor_metrics.csv`) | **67.6%** | 67.6% | Best genuinely-private result. Randomized response keeps each label correct w.p. 0.75, so learning survives. Still climbing at round 30. |
| FedSyn (`fedsyn_emnist_results.csv`) | 66.5% | 66.5% | Healthy convergence in 10 rounds (no DP noise on this variant). |
| Output perturbation ε-sweep (`outputperturbation_dp_fl_emnist.csv`) | 64.2% @ ε=10 | — | Classic privacy–utility curve: 2.4% at ε=0.1 → 53.2% at ε=5 → 64.2% at ε=10. |
| Bayesian DP-FL ε-sweep (`bayesian_dp_fl_emnist.csv`) | 63.4% @ ε=10 | — | Same shape: chance at ε≤0.5, usable from ε≈5. |
| DP-FedAvg / DP-FTRL / Central DP / Local DP / SecureAgg / PLDP / Strong- & WeakFedFDP (per-round CSVs) | 2–6% | 1.5–3.4% | **All stuck at chance for all 30 rounds.** See analysis below. |

### What this means

1. **Where you add noise matters more than how much.** The two consistent winners put randomness *away from the gradients*: RAPPOR's label randomization (EMNIST, 67.6%) and output perturbation on final weights (Sent140 74.1%; EMNIST 64.2% at ε=10). Per-gradient noise was the most destructive placement in every comparison.

2. **The per-round EMNIST DP-FL family never learned, and the logs explain why.** Those scripts clip each client update to total norm 1.0, then add Gaussian noise with σ=1.0 **per parameter**. With ~430K parameters, the injected noise has norm ~√430K ≈ 650× the signal ceiling — the aggregated update is essentially pure noise, so accuracy pins at the 47-class chance rate (~2.1%) for all 30 rounds. This is a *calibration* failure (noise not scaled to the clipping norm / parameter count), not evidence that DP-FL can't work on EMNIST.

3. **The ε-sweep experiments show the textbook privacy–utility tradeoff.** Bayesian DP-FL and output perturbation on EMNIST go from chance at ε≤0.5 to ~63–64% at ε=10 — a clean S-curve. If you want one plot that summarizes the project, it's `logs/EMNIST/outputperturbation_dp_fl_emnist.png`.

4. **A flat accuracy-vs-ε line is a red flag, not a win.** DP-XGBoost is bit-identical across every ε on both datasets because `dp_epsilon`/`dp_enabled` are not real XGBoost parameters — XGBoost silently ignores them. Those runs are non-private baselines (70.7% / 80.2%), useful as ceilings but providing **no** privacy.

5. **Privacy budget direction is consistent.** Weak budget beats strong budget (Sent140: 62.0% vs 55.5%), more rounds ⇒ more accumulated ε with eventual utility decay (DP-FTRL peaks mid-run then falls; central DP decays monotonically). The mechanisms interact with ε in the expected direction wherever the noise is calibrated sanely.

6. **Dataset difficulty interacts with DP.** Binary Sent140 tolerates noise (many methods float a few points above the 50% floor), while 47-class EMNIST is all-or-nothing: a method either learns (>60%) or sits at chance (2%). High-class-count tasks are far less forgiving of badly calibrated noise.

### Caveats on the ε values

The logged ε is a **simplified per-script estimate** (typically `ε = round × noise multiplier`, in one case a literal placeholder `ε = round`), not output of a real accountant (RDP / moments / zCDP). Use it to compare *relative* privacy levels within one script, not as a formal guarantee or across scripts.

---

## Known issues

- **DP-XGBoost applies no DP.** `train_dpxgb.py` / `train_dp_xgb_emnist.py` pass `dp_epsilon`, `dp_enabled`, … to `XGBClassifier`, which are not XGBoost parameters and are silently ignored — hence identical accuracy at every ε. Fixing this requires an actual DP-GBDT implementation.
- **EMNIST per-round DP noise is uncalibrated.** σ=1.0 per parameter against a total-norm-1.0 clip ⇒ noise dominates by ~3 orders of magnitude; all eight per-round EMNIST DP-FL runs are at chance. Scale σ relative to `CLIP / parameters` (or use per-layer clipping) to get meaningful curves.
- **`utils/` package and `configs/` were never committed**, so `run_experiment.py` and the five `utils.*`-importing trainers don't run. The standalone scripts are the supported path.
- **`logs/pldpfl_metrics.csv` (Sent140) may be SecureAgg output.** `train_secureagg_dp.py` used to write to PLDP-FL's filenames and could have overwritten that log; fixed now (it writes `secureagg_metrics.csv`), but the committed CSV's provenance is ambiguous. Both runs sat at ~50% regardless.
- **Orphan legacy logs.** `fedavg_metrics.csv`, `dpftrl_metrics.csv`, `outputperturbation_metrics.csv`, `strongfdp_metrics.csv` were produced by earlier script versions no longer in the repo. They're kept because they contain the best Sent140 results, but they can't be regenerated exactly.
- **`dp_fedsyn_sent140.csv` is a broken run** (0% accuracy on a binary task, rising loss) — ignore it.

## Next steps

1. Recalibrate EMNIST noise (σ ∝ clip-norm / √params) and re-run the per-round family to get real privacy–utility curves.
2. Replace the linear ε bookkeeping with a proper accountant (e.g. Opacus' RDP accountant).
3. Implement real DP for the XGBoost experiments or relabel them as non-private baselines.
4. Restore/rewrite `utils/` + `configs/` if the unified runner is still wanted; otherwise delete the five broken trainers.
