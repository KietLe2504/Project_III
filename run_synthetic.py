"""Phase 1 - controlled experiments on simulated Heston data (true variance known).

python run_synthetic.py --quick                  # smoke test (~vai phut)
python run_synthetic.py                          # full experiment
python run_synthetic.py --models naive garch har gp   # skip LSTM (no torch needed)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from volfc.experiments import run_experiment
from volfc.heston import REGIMES, simulate_heston
from volfc.models import build_models
from volfc.plots import plot_calibration, plot_forecasts, plot_synthetic_summary

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--h", type=int, default=5, help="forecast horizon (days)")
ap.add_argument("--n-test", type=int, default=600)
ap.add_argument("--n-train", type=int, nargs="+", default=[500, 1500])
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--refit", type=int, default=100, help="refit every k days")
ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
ap.add_argument("--models", nargs="+", default=["naive", "garch", "har", "gp", "lstm"])
ap.add_argument("--out", default="results/synthetic")
a = ap.parse_args()
if a.quick:
    a.n_test, a.n_train, a.seeds, a.refit = 250, [600], 1, 125
    a.regimes = a.regimes[:2]

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

# --- sanity check: does the simulator reproduce stylised facts? ------------------------------
chk = []
for name in a.regimes:
    s = simulate_heston(4000, REGIMES[name], seed=123)
    r2 = s.returns ** 2
    chk.append({"regime": name, "ann_vol_%": float(np.sqrt(252 * s.true_var.mean())),
                "excess_kurtosis": float(kurtosis(s.returns)),
                "acf1_r2": float(np.corrcoef(r2[:-1], r2[1:])[0, 1]),
                "acf10_r2": float(np.corrcoef(r2[:-10], r2[10:])[0, 1]),
                "corr(r, dvar)": float(np.corrcoef(s.returns[1:], np.diff(s.true_var))[0, 1])})
chk = pd.DataFrame(chk); chk.to_csv(out / "simulator_check.csv", index=False)
print("Simulator stylised facts:\n", chk.round(3).to_string(index=False), "\n")

# --- main experiment -------------------------------------------------------------------------
all_rows, example = [], None
for regime in a.regimes:
    for ntr in a.n_train:
        for seed in range(a.seeds):
            print(f"[{regime}] n_train={ntr} seed={seed}", flush=True)
            sim = simulate_heston(ntr + a.n_test, REGIMES[regime], seed=100 + seed)  # seeds disjoint from pilot tuning
            models = build_models(a.models, seed=seed, quick=a.quick)
            df, preds, origins, targets = run_experiment(
                sim.returns, models, a.h, test_start=ntr, refit_every=a.refit, true_var=sim.true_var)
            df.insert(0, "seed", seed); df.insert(0, "n_train", ntr); df.insert(0, "regime", regime)
            all_rows.append(df)
            if example is None or (regime == a.regimes[0] and ntr == max(a.n_train) and seed == 0):
                example = (regime, ntr, preds, origins, targets)
            pd.concat(all_rows).to_csv(out / "synthetic_runs.csv", index=False)   # save progressively

runs = pd.concat(all_rows, ignore_index=True)
runs.to_csv(out / "synthetic_runs.csv", index=False)
num = [c for c in runs.columns if c not in ("regime", "n_train", "seed", "model", "target")]
summary = runs.groupby(["target", "regime", "n_train", "model"])[num].mean().reset_index()
summary.to_csv(out / "synthetic_summary.csv", index=False)

print("\n=== Mean over seeds: error vs TRUE variance (ground truth) ===")
print(summary[summary.target == "true"][["regime", "n_train", "model", "rmse", "mae", "qlike"]]
      .round(4).to_string(index=False))

plot_synthetic_summary(runs, out / "fig_summary_qlike.png", "qlike")
plot_synthetic_summary(runs, out / "fig_summary_rmse.png", "rmse")
plot_calibration(runs[runs.target == "proxy"], out / "fig_calibration_proxy.png", "Interval calibration (vs proxy)")
regime, ntr, preds, origins, targets = example
plot_forecasts(origins, preds, targets, out / "fig_forecast_example.png",
               title=f"Heston [{regime}], n_train={ntr}, h={a.h}")
print(f"\nSaved results to {out}/")
