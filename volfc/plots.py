from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .heston import TRADING_DAYS

ORDER = ["Naive", "GARCH", "GJR", "HAR", "HAR-L", "GP", "GP-L", "LSTM"]
COLORS = {"Naive": "#999999", "GARCH": "#1f77b4", "GJR": "#17becf", "HAR": "#2ca02c", "HAR-L": "#98df8a",
          "GP": "#d62728", "GP-L": "#ff9896", "LSTM": "#ff7f0e"}


def _ann_vol(v):
    return np.sqrt(TRADING_DAYS * np.maximum(v, 1e-12))


def plot_forecasts(origins, preds, targets, out, title="", last=300):
    """Forecast vs ground truth (synthetic) / proxy (real); GP 95% band if available."""
    sel = origins[-last:]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(sel, _ann_vol(targets["proxy"][sel]), color="#cccccc", lw=1, label="realised-vol proxy")
    if "true" in targets:
        ax.plot(sel, _ann_vol(targets["true"][sel]), color="black", lw=1.8, label="true Heston vol")
    for name in ORDER:
        if name in preds:
            ax.plot(sel, _ann_vol(preds[name].var[sel]), color=COLORS[name], lw=1.1, label=name)
    if "GP" in preds and not np.all(np.isnan(preds["GP"].log_sd)):
        mu, sd = preds["GP"].log_mu[sel], preds["GP"].log_sd[sel]
        ax.fill_between(sel, _ann_vol(np.exp(mu - 1.96 * sd)), _ann_vol(np.exp(mu + 1.96 * sd)),
                        color=COLORS["GP"], alpha=0.12, label="GP 95% band")
    ax.set_xlabel("day index"); ax.set_ylabel("annualised volatility (%)"); ax.set_title(title)
    ax.legend(ncol=4, fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def plot_calibration(rows_df, out, title=""):
    """Empirical coverage of GP/HAR predictive intervals vs nominal level."""
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    for name in ("HAR", "GP"):
        d = rows_df[(rows_df.model == name)]
        cols = [c for c in d.columns if c.startswith("cov")]
        if d.empty or not cols:
            continue
        lv = [int(c[3:]) / 100 for c in cols]
        ax.plot(lv, d[cols].mean().to_numpy(), "o-", color=COLORS[name], label=name)
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage"); ax.set_title(title)
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def plot_synthetic_summary(runs, out, metric="qlike"):
    df = runs[runs.target == "true"]
    models = [m for m in ORDER if m in set(df.model)]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    p = df.groupby(["regime", "model"])[metric].mean().unstack("model")[models]
    p.plot.bar(ax=axes[0], color=[COLORS[m] for m in models], rot=0)
    axes[0].set_title(f"{metric.upper()} vs true variance, by Heston regime"); axes[0].set_ylabel(metric)
    q = df.groupby(["n_train", "model"])[metric].mean().unstack("model")[models]
    q.plot(ax=axes[1], marker="o", color=[COLORS[m] for m in models])
    axes[1].set_title(f"{metric.upper()} vs training-set size"); axes[1].set_xlabel("n_train (days)")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def plot_regime_bars(reg_df, out, metric="qlike", title=""):
    models = [m for m in ORDER if m in set(reg_df.model)]
    p = reg_df.groupby(["regime", "model"])[metric].mean().unstack("model")[models]
    ax = p.plot.bar(figsize=(6.5, 4.5), color=[COLORS[m] for m in models], rot=0)
    ax.set_title(title); ax.set_ylabel(metric)
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()


def plot_extra(df, x, out, metric="qlike", title="", line=False):
    """Mean metric by model across the levels of column `x`."""
    models = [m for m in ORDER if m in set(df.model)]
    p = df.groupby([x, "model"])[metric].mean().unstack("model")[models]
    if line:
        ax = p.plot(marker="o", figsize=(7, 4.5), color=[COLORS[m] for m in models])
    else:
        ax = p.plot.bar(figsize=(8, 4.5), color=[COLORS[m] for m in models], rot=0)
    ax.set_title(title); ax.set_ylabel(metric); ax.set_xlabel(x)
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()