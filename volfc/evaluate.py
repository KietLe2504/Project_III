"""Walk-forward evaluation, loss functions, Diebold-Mariano test, interval calibration."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .heston import TRADING_DAYS
from .models import Forecast, Forecaster


def walk_forward(model: Forecaster, r: np.ndarray, h: int, test_start: int, refit_every: int,
                 train_window: int | None = None) -> Forecast:
    """Expanding-window walk-forward. At origin block start s the model is refit on r[:s+1];
    forecasts for origins t in [s, s+refit_every) use only r[:t+1]. If `train_window` is given the model
    only sees the last `train_window` days up to s (rolling window) - used for the learning-curve study."""
    n = len(r)
    last = n - h                              # last origin whose h-day target is fully observed
    var = np.full(n, np.nan)
    lmu = np.full(n, np.nan)
    lsd = np.full(n, np.nan)
    for s in range(test_start, last, refit_every):
        e = min(s + refit_every, last)
        off = 0 if train_window is None else max(0, s - train_window + 1)
        rs = r[off:]
        model.fit(rs, h, end=s - off)
        ts = np.arange(s, e)
        f = model.predict(rs, ts - off)
        var[ts] = f.var
        if f.log_mu is not None:
            lmu[ts] = f.log_mu
        if f.log_sd is not None:
            lsd[ts] = f.log_sd
    return Forecast(var=var, log_mu=lmu, log_sd=lsd)


def losses(pred_var: np.ndarray, truth_var: np.ndarray):
    """Per-observation losses: squared/abs error on annualised vol (%), and QLIKE on variance."""
    p, q = np.maximum(pred_var, 1e-6), np.maximum(truth_var, 1e-6)
    vp, vq = np.sqrt(TRADING_DAYS * p), np.sqrt(TRADING_DAYS * q)
    return (vp - vq) ** 2, np.abs(vp - vq), q / p - np.log(q / p) - 1.0


def summarize(pred_var, truth_var, mask) -> dict:
    se, ae, ql = losses(pred_var[mask], truth_var[mask])
    return {"rmse": float(np.sqrt(se.mean())), "mae": float(ae.mean()), "qlike": float(ql.mean()),
            "n": int(mask.sum())}


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano test (H0: equal expected loss), Bartlett/Newey-West HAC with lag h-1.
    Negative statistic => model A has lower loss than model B."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    T = len(d)
    dc = d - d.mean()
    lrv = float(dc @ dc) / T
    for k in range(1, h):
        w = 1.0 - k / h
        lrv += 2.0 * w * float(dc[k:] @ dc[:-k]) / T
    stat = d.mean() / np.sqrt(max(lrv, 1e-18) / T)
    return float(stat), float(2.0 * (1.0 - norm.cdf(abs(stat))))


def coverage(log_mu, log_sd, log_target, levels=(0.5, 0.8, 0.9, 0.95)) -> dict[float, float]:
    """Empirical share of targets inside the central predictive interval of the log-variance."""
    ok = ~(np.isnan(log_mu) | np.isnan(log_sd) | np.isnan(log_target))
    out = {}
    for lv in levels:
        z = norm.ppf(0.5 + lv / 2)
        out[lv] = float(np.mean(np.abs(log_target[ok] - log_mu[ok]) <= z * log_sd[ok]))
    return out