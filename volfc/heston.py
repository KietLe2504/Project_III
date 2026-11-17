"""Heston stochastic-volatility simulator (Euler, full truncation).

dS/S = mu dt + sqrt(v) dW1
dv   = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho

Each trading day is split into `n_sub` sub-steps. Besides daily log-returns we keep the
*true* daily integrated variance (the ground truth that a real market never reveals).
Everything is returned in percent units: returns in %, variance in %^2 per day.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

TRADING_DAYS = 252


@dataclass(frozen=True)
class HestonParams:
    mu: float = 0.05
    kappa: float = 3.0    # mean-reversion speed
    theta: float = 0.04   # long-run variance (annual) -> 20% vol
    xi: float = 0.5       # vol of vol
    rho: float = -0.7     # leverage correlation
    v0: float | None = None


# Regimes used in the experiments (each stresses a different aspect of the volatility process)
REGIMES: dict[str, HestonParams] = {
    "baseline": HestonParams(),
    "fast_meanrev": replace(HestonParams(), kappa=10.0, xi=0.6),
    "high_volvol": replace(HestonParams(), kappa=2.0, xi=1.0),
    "crisis": replace(HestonParams(), kappa=3.0, theta=0.09, xi=0.8, rho=-0.8),
}


# Leverage experiment: same dynamics, only the return/variance correlation rho changes
LEVERAGE_REGIMES: dict[str, HestonParams] = {
    f"rho{rho}": replace(HestonParams(), kappa=3.0, xi=0.8, rho=rho) for rho in (0.0, -0.5, -0.9)
}


@dataclass
class HestonSim:
    returns: np.ndarray   # daily log-returns, %
    true_var: np.ndarray  # daily integrated variance, %^2
    params: HestonParams


def simulate_heston(n_days: int, params: HestonParams, n_sub: int = 10, seed: int = 0,
                    burn: int = 250) -> HestonSim:
    rng = np.random.default_rng(seed)
    total = n_days + burn
    n = total * n_sub
    dt = 1.0 / (TRADING_DAYS * n_sub)
    z_v = rng.standard_normal(n)
    z_s = params.rho * z_v + np.sqrt(1.0 - params.rho ** 2) * rng.standard_normal(n)

    v_cur = params.theta if params.v0 is None else params.v0
    var_path = np.empty(n)
    logret = np.empty(n)
    for i in range(n):
        v_pos = v_cur if v_cur > 0.0 else 0.0           # full truncation
        sq = np.sqrt(v_pos * dt)
        logret[i] = (params.mu - 0.5 * v_pos) * dt + sq * z_s[i]
        v_cur = v_cur + params.kappa * (params.theta - v_pos) * dt + params.xi * sq * z_v[i]
        var_path[i] = v_pos

    daily_ret = logret.reshape(total, n_sub).sum(axis=1)
    daily_iv = (var_path * dt).reshape(total, n_sub).sum(axis=1)   # integrated variance per day
    return HestonSim(returns=100.0 * daily_ret[burn:], true_var=1e4 * daily_iv[burn:], params=params)