"""Thi nghiem bo sung.

  --exp leverage-synth   Heston voi rho = 0 / -0.5 / -0.9: hieu ung don bay co giup GJR, HAR-L, GP-L khong? (co phuong sai that)
  --exp leverage         GARCH vs GJR, HAR vs HAR-L, GP vs GP-L tren du lieu that
  --exp horizon          h = 1, 5, 22 tren du lieu that
  --exp learning         cung tap test, doi cua so train (250/500/1000/1500/toan bo) tren du lieu that

Vi du:  python run_extra.py --exp leverage-synth
        python run_extra.py --exp leverage --symbols VCB FPT HPG VNM VNINDEX
        python run_extra.py --exp horizon  --symbols VCB FPT HPG VNM VNINDEX
        python run_extra.py --exp learning --symbols VCB FPT HPG VNM VNINDEX
Them --quick de chay thu.  Ket qua: results/extra/<exp>/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from volfc.data import VN30_EXAMPLE, load_prices, log_returns_pct
from volfc.experiments import run_experiment
from volfc.heston import LEVERAGE_REGIMES, simulate_heston
from volfc.models import build_models
from volfc.plots import plot_extra

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True, choices=["leverage-synth", "leverage", "horizon", "learning"])
ap.add_argument("--symbols", nargs="+", default=VN30_EXAMPLE + ["VNINDEX"])
ap.add_argument("--csv-dir", default="data/raw")
ap.add_argument("--start", default="2015-01-01")
ap.add_argument("--end", default="2026-10-30")
ap.add_argument("--h", type=int, default=5)
ap.add_argument("--train-frac", type=float, default=0.5)
ap.add_argument("--refit", type=int, default=100)
ap.add_argument("--models", nargs="+", default=None)
ap.add_argument("--windows", type=int, nargs="+", default=[250, 500, 1000, 1500], help="learning: train windows")
ap.add_argument("--seeds", type=int, default=3, help="leverage-synth")
ap.add_argument("--n-train", type=int, default=1500, help="leverage-synth")
ap.add_argument("--n-test", type=int, default=600, help="leverage-synth")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="results/extra")
a = ap.parse_args()
out = Path(a.out) / a.exp; out.mkdir(parents=True, exist_ok=True)


def real_series(sym):
    prices = load_prices(sym, a.start, a.end, a.csv_dir)
    r, dates, n_drop = log_returns_pct(prices)
    print(f"  {len(r)} returns, dropped {n_drop} outliers", flush=True)
    return r


def tag(df, **cols):
    for i, (k, v) in enumerate(cols.items()):
        df.insert(i, k, v)
    return df


rows = []
if a.exp == "leverage-synth":
    names = a.models or ["garch", "gjr", "har", "har_l", "gp", "gp_l"]
    seeds, n_test, n_train = (1, 250, 600) if a.quick else (a.seeds, a.n_test, a.n_train)
    for regime, params in LEVERAGE_REGIMES.items():
        for seed in range(seeds):
            print(f"[{regime}] seed={seed}", flush=True)
            sim = simulate_heston(n_train + n_test, params, seed=200 + seed)
            df, *_ = run_experiment(sim.returns, build_models(names, seed, a.quick), a.h,
                                    n_train, a.refit, true_var=sim.true_var)
            rows.append(tag(df, regime=regime, seed=seed))
            pd.concat(rows).to_csv(out / "runs.csv", index=False)
    runs = pd.concat(rows, ignore_index=True)
    t = runs[runs.target == "true"]
    plot_extra(t, "regime", out / "fig_qlike_true.png", "qlike", "Leverage experiment (Heston), QLIKE vs true variance")
    print("\nQLIKE vs TRUE variance (mean over seeds):")
    print(t.groupby(["regime", "model"]).qlike.mean().unstack().round(3).to_string())

else:
    default = {"leverage": ["garch", "gjr", "har", "har_l", "gp", "gp_l"],
               "horizon": ["naive", "garch", "har", "gp", "lstm"],
               "learning": ["garch", "har", "gp", "lstm"]}[a.exp]
    names = a.models or default
    for sym in a.symbols:
        print(f"[{sym}]", flush=True)
        try:
            r = real_series(sym)
        except Exception as e:
            print(f"  skipped: {e}"); continue
        n = len(r)
        base_start = max(750, int(a.train_frac * n))
        if a.exp == "leverage":
            df, *_ = run_experiment(r, build_models(names, 0, a.quick), a.h, base_start, a.refit)
            rows.append(tag(df, symbol=sym))
        elif a.exp == "horizon":
            for h in (1, 5, 22):
                print(f"  h={h}", flush=True)
                df, *_ = run_experiment(r, build_models(names, 0, a.quick), h, base_start, a.refit)
                rows.append(tag(df, symbol=sym, h=h))
        else:  # learning curve: identical test period for every window
            ts0 = max(max(a.windows), int(a.train_frac * n))
            for w in a.windows + [0]:
                print(f"  window={'all' if w == 0 else w}", flush=True)
                df, *_ = run_experiment(r, build_models(names, 0, a.quick), a.h, ts0, a.refit,
                                        train_window=None if w == 0 else w)
                rows.append(tag(df, symbol=sym, train_window="all" if w == 0 else w))
        pd.concat(rows).to_csv(out / "runs.csv", index=False)

    runs = pd.concat(rows, ignore_index=True)
    t = runs[runs.target == "proxy"]
    if a.exp == "leverage":
        plot_extra(t, "symbol", out / "fig_qlike.png", "qlike", "Leverage: QLIKE per symbol")
        print("\nQLIKE (proxy) per symbol:"); print(t.pivot_table(index="symbol", columns="model", values="qlike").round(4).to_string())
        print("\nMean:", t.groupby("model").qlike.mean().round(4).to_dict())
    elif a.exp == "horizon":
        plot_extra(t, "h", out / "fig_qlike_by_h.png", "qlike", "QLIKE by forecast horizon")
        print("\nQLIKE by horizon:"); print(t.groupby(["h", "model"]).qlike.mean().unstack().round(4).to_string())
    else:
        t = t.copy(); t["train_window"] = t["train_window"].astype(str)
        order = [str(w) for w in a.windows] + ["all"]
        t["train_window"] = pd.Categorical(t["train_window"], order, ordered=True)
        plot_extra(t, "train_window", out / "fig_learning_curve.png", "qlike", "Learning curve (same test period)", line=True)
        print("\nQLIKE by train window:"); print(t.groupby(["train_window", "model"], observed=True).qlike.mean().unstack().round(4).to_string())
print(f"\nSaved to {out}/")
