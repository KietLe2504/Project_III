"""Phase 2 - real VN data (target = realised-variance proxy from squared returns).

python run_real.py --symbols VCB FPT HPG VNINDEX
Place CSVs (columns: date,close) in data/raw/<SYMBOL>.csv to skip vnstock.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from volfc.data import VN30_EXAMPLE, load_prices, log_returns_pct
from volfc.experiments import by_vol_regime, run_experiment
from volfc.models import build_models
from volfc.plots import plot_calibration, plot_forecasts, plot_regime_bars

ap = argparse.ArgumentParser()
ap.add_argument("--symbols", nargs="+", default=VN30_EXAMPLE + ["VNINDEX"])
ap.add_argument("--start", default="2015-01-01")
ap.add_argument("--end", default="2026-10-30")
ap.add_argument("--h", type=int, default=5)
ap.add_argument("--train-frac", type=float, default=0.5)
ap.add_argument("--refit", type=int, default=100)
ap.add_argument("--models", nargs="+", default=["naive", "garch", "har", "gp", "lstm"])
ap.add_argument("--csv-dir", default="data/raw")
ap.add_argument("--out", default="results/real")
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
runs, regimes = [], []
for i, sym in enumerate(a.symbols):
    print(f"[{sym}]", flush=True)
    try:
        prices = load_prices(sym, a.start, a.end, a.csv_dir)
    except Exception as e:                      # keep going if one symbol fails
        print(f"  skipped: {e}"); continue
    r, dates, n_drop = log_returns_pct(prices)
    test_start = max(750, int(a.train_frac * len(r)))
    print(f"  {len(r)} returns, {dates[0].date()} -> {dates[-1].date()}, dropped {n_drop} outliers, "
          f"test from {dates[test_start].date()}")
    models = build_models(a.models, seed=0, quick=a.quick)
    df, preds, origins, targets = run_experiment(r, models, a.h, test_start, a.refit)
    df.insert(0, "symbol", sym); runs.append(df)
    reg = by_vol_regime(r, preds, targets["proxy"], origins); reg.insert(0, "symbol", sym); regimes.append(reg)
    plot_forecasts(origins, preds, targets, out / f"fig_forecast_{sym}.png", title=f"{sym} (h={a.h})")
    pd.concat(runs).to_csv(out / "real_runs.csv", index=False)

if runs:
    runs = pd.concat(runs, ignore_index=True); regimes = pd.concat(regimes, ignore_index=True)
    runs.to_csv(out / "real_runs.csv", index=False); regimes.to_csv(out / "real_by_regime.csv", index=False)
    print("\n=== Real data vs proxy ===")
    print(runs[["symbol", "model", "rmse", "mae", "qlike", "dm_p_vs_garch"]].round(4).to_string(index=False))
    print("\nMean over symbols:\n", runs.groupby("model")[["rmse", "mae", "qlike"]].mean().round(4))
    plot_calibration(runs, out / "fig_calibration.png", "Interval calibration (real data)")
    plot_regime_bars(regimes, out / "fig_regime_qlike.png", "qlike", "QLIKE: calm vs turbulent periods")
    print(f"\nSaved results to {out}/")
