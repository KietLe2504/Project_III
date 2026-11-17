"""Causal features and forward-looking targets. All rolling features at day t use data <= t only."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-3  # log floor (variance units: %^2)


def har_features(r: np.ndarray, windows=(1, 5, 22), leverage: bool = False) -> np.ndarray:
    """log of trailing mean squared return over several windows (HAR-style). NaN until warmed up.
    leverage=True appends the standardised, clipped signed return of day t and the 5-day mean of its negative
    part: squared returns alone cannot tell a drop from a rise, so they cannot capture the leverage effect."""
    r = np.asarray(r, dtype=float)
    r2 = pd.Series(r ** 2)
    cols = [np.log(r2.rolling(w, min_periods=w).mean().to_numpy() + EPS) for w in windows]
    if leverage:
        # standardise by trailing 22-day vol (known at t) and clip, so the features are bounded and
        # the linear/GP extrapolation cannot explode on a single extreme return
        vol22 = np.sqrt(r2.rolling(22, min_periods=22).mean().to_numpy() + EPS)
        z = np.clip(r / vol22, -3.0, 3.0)
        cols.append(z)
        cols.append(pd.Series(np.minimum(z, 0.0)).rolling(5, min_periods=5).mean().to_numpy())
    return np.column_stack(cols)


def future_mean(x: np.ndarray, h: int) -> np.ndarray:
    """out[t] = mean(x[t+1 .. t+h]); NaN where the window is not fully available."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(n, np.nan)
    idx = np.arange(n - h)
    out[idx] = (c[idx + 1 + h] - c[idx + 1]) / h
    return out


def future_mean_sq(r: np.ndarray, h: int) -> np.ndarray:
    """Realised-variance proxy of the next h days from squared returns."""
    return future_mean(np.asarray(r) ** 2, h)