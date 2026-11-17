"""Run all models on one return series; compute metrics against proxy (and true variance if known)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import coverage, dm_test, losses, summarize, walk_forward
from .features import EPS, future_mean, future_mean_sq


def run_experiment(r, models, h, test_start, refit_every, true_var=None, verbose=True, train_window=None):
    """Returns (metrics DataFrame, dict name -> Forecast, origins array, targets dict)."""
    n = len(r)
    origins = np.arange(test_start, n - h)
    mask = np.zeros(n, dtype=bool)
    mask[origins] = True
    targets = {"proxy": future_mean_sq(r, h)}
    if true_var is not None:
        targets["true"] = future_mean(true_var, h)

    preds = {}
    for m in models:
        if verbose:
            print(f"    fitting/forecasting {m.name} ...", flush=True)
        preds[m.name] = walk_forward(m, r, h, test_start, refit_every, train_window)

    rows = []
    for tname, tv in targets.items():
        garch_ql = losses(preds["GARCH"].var[mask], tv[mask])[2] if "GARCH" in preds else None
        for name, f in preds.items():
            row = {"model": name, "target": tname, **summarize(f.var, tv, mask)}
            ql = losses(f.var[mask], tv[mask])[2]
            if garch_ql is not None and name != "GARCH":
                stat, p = dm_test(ql, garch_ql, h)
                row.update(dm_stat_vs_garch=stat, dm_p_vs_garch=p)
            if f.log_sd is not None and not np.all(np.isnan(f.log_sd)):
                cov = coverage(f.log_mu[mask], f.log_sd[mask], np.log(tv[mask] + EPS))
                row.update({f"cov{int(k * 100)}": v for k, v in cov.items()})
            rows.append(row)
    return pd.DataFrame(rows), preds, origins, targets


def by_vol_regime(r, preds, target, origins, window=22):
    """Split test origins into calm / turbulent days by trailing realised variance (known at t)."""
    trail = pd.Series(np.asarray(r) ** 2).rolling(window, min_periods=1).mean().to_numpy()
    thr = np.median(trail[origins])
    rows = []
    for label, sel in (("calm", trail[origins] <= thr), ("turbulent", trail[origins] > thr)):
        mask = np.zeros(len(r), dtype=bool)
        mask[origins[sel]] = True
        for name, f in preds.items():
            rows.append({"regime": label, "model": name, **summarize(f.var, target, mask)})
    return pd.DataFrame(rows)