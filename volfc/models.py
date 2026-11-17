"""Forecasters. Common interface:

    fit(r, h, end)   -> learn using ONLY r[:end+1] (training pairs satisfy t + h <= end, no leakage)
    predict(r, ts)   -> forecast, made at the close of day t (uses r[:t+1]), of the average daily
                        variance over days t+1 .. t+h, for every origin t in `ts`.

All variances are in %^2 per day.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .features import EPS, future_mean_sq, har_features


@dataclass
class Forecast:
    var: np.ndarray
    log_mu: np.ndarray | None = None   # predictive mean of log-variance (HAR, GP)
    log_sd: np.ndarray | None = None   # predictive sd of log-variance   (HAR, GP)


class Forecaster:
    name = "base"

    def fit(self, r: np.ndarray, h: int, end: int) -> None:
        raise NotImplementedError

    def predict(self, r: np.ndarray, ts: np.ndarray) -> Forecast:
        raise NotImplementedError


# ----------------------------------------------------------------------------- baselines
class RollingVar(Forecaster):
    """Naive baseline: tomorrow's variance = trailing `window`-day mean squared return."""

    def __init__(self, window: int = 22):
        self.window = window
        self.name = "Naive"

    def fit(self, r, h, end):
        pass

    def predict(self, r, ts):
        rv = pd.Series(np.asarray(r) ** 2).rolling(self.window, min_periods=1).mean().to_numpy()
        return Forecast(var=rv[ts])


class Garch11(Forecaster):
    """GARCH(1,1) with Student-t innovations. Parameters by MLE (arch package); the h-day average
    variance forecast uses the closed-form mean-reverting recursion with fixed parameters.
    With `asym=True` it is GJR-GARCH(1,1,1): sigma2_t = w + (a + g*1[e<0]) e^2 + b sigma2."""

    name = "GARCH"

    def __init__(self, dist: str = "t", max_train: int | None = None, asym: bool = False):
        self.dist, self.max_train, self.asym = dist, max_train, asym
        if asym:
            self.name = "GJR"
        self.gamma = 0.0

    def fit(self, r, h, end):
        from arch import arch_model

        x = np.asarray(r[: end + 1], dtype=float)
        if self.max_train:
            x = x[-self.max_train:]
        self.h, self.s0 = h, float(np.var(x))
        am = arch_model(x, mean="Constant", vol="GARCH", p=1, o=1 if self.asym else 0, q=1,
                        dist=self.dist, rescale=False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = am.fit(disp="off", show_warning=False).params
            self.mu, self.omega = float(p["mu"]), float(p["omega"])
            self.alpha, self.beta = float(p["alpha[1]"]), float(p["beta[1]"])
            self.gamma = float(p["gamma[1]"]) if self.asym else 0.0
        except Exception:  # optimiser failed: keep previous parameters, else a sane default
            if not hasattr(self, "omega"):
                self.mu, self.alpha, self.beta, self.gamma = float(np.mean(x)), 0.05, 0.90, 0.0
                self.omega = self.s0 * 0.05

    def predict(self, r, ts):
        t_max = int(np.max(ts))
        eps = np.asarray(r[: t_max + 1], dtype=float) - self.mu
        eps2, neg = eps ** 2, (eps < 0).astype(float)
        s2 = np.empty(t_max + 1)
        s2[0] = self.s0
        for t in range(1, t_max + 1):
            s2[t] = self.omega + (self.alpha + self.gamma * neg[t - 1]) * eps2[t - 1] + self.beta * s2[t - 1]
        s2_next = self.omega + (self.alpha + self.gamma * neg) * eps2 + self.beta * s2   # sigma^2_{t+1}
        persist = min(self.alpha + 0.5 * self.gamma + self.beta, 0.9999)
        long_run = self.omega / (1.0 - persist)
        decay = np.mean(persist ** np.arange(self.h))                                    # avg over horizon
        return Forecast(var=(long_run + (s2_next - long_run) * decay)[ts])


class LogHAR(Forecaster):
    """HAR-type linear regression on log realised variances (1/5/22-day), log-normal bias corrected."""

    name = "HAR"

    def __init__(self, leverage: bool = False):
        self.lev = leverage
        if leverage:
            self.name = "HAR-L"

    def fit(self, r, h, end):
        rr = np.asarray(r[: end + 1], dtype=float)
        X, y = har_features(rr, leverage=self.lev), np.log(future_mean_sq(rr, h) + EPS)
        m = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        self.lr = LinearRegression().fit(X[m], y[m])
        self.s2 = float(np.var(y[m] - self.lr.predict(X[m])))

    def predict(self, r, ts):
        X = har_features(np.asarray(r[: int(np.max(ts)) + 1], dtype=float), leverage=self.lev)[ts]
        mu = self.lr.predict(X)
        return Forecast(var=np.exp(mu + 0.5 * self.s2), log_mu=mu,
                        log_sd=np.full(len(ts), np.sqrt(self.s2)))


# ----------------------------------------------------------------------------- Gaussian process
class GPVol(Forecaster):
    """Gaussian-process regression of log future realised variance on the HAR features, with a
    *linear (HAR) mean function* and a GP on the residuals:

        log RV_{t+1:t+h} = w'x_t + f(x_t) + eps,   f ~ GP(0, k),  k = c * Matern-5/2 (ARD),  eps ~ N(0, s2)

    The linear part is fitted on all available pairs; the GP learns the non-linear remainder from the
    last `max_train` pairs. Length-scales are bounded below (smoothness prior), since the target is very
    noisy and an unconstrained kernel just interpolates noise. Output is a predictive *distribution*."""

    name = "GP"

    def __init__(self, max_train: int = 750, n_restarts: int = 1, seed: int = 0,
                 min_length_scale: float = 2.0, max_amplitude: float = 1.0, leverage: bool = False):
        self.max_train, self.n_restarts, self.seed = max_train, n_restarts, seed
        self.min_ls, self.max_amp, self.lev = min_length_scale, max_amplitude, leverage
        if leverage:
            self.name = "GP-L"

    def fit(self, r, h, end):
        rr = np.asarray(r[: end + 1], dtype=float)
        X, y = har_features(rr, leverage=self.lev), np.log(future_mean_sq(rr, h) + EPS)
        m = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[m], y[m]
        self.lin = LinearRegression().fit(X, y)
        res = y - self.lin.predict(X)
        Xg, rg = X[-self.max_train:], res[-self.max_train:]
        self.scaler = StandardScaler().fit(Xg)
        kernel = (ConstantKernel(0.1, (1e-3, self.max_amp))
                  * Matern(length_scale=np.full(X.shape[1], 3.0),
                           length_scale_bounds=(self.min_ls, 1e2), nu=2.5)
                  + WhiteKernel(0.5, (1e-3, 5.0)))
        self.gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=self.n_restarts,
                                           random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(self.scaler.transform(Xg), rg)

    def predict(self, r, ts):
        X = har_features(np.asarray(r[: int(np.max(ts)) + 1], dtype=float), leverage=self.lev)[ts]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f, sd = self.gp.predict(self.scaler.transform(X), return_std=True)
        mu = self.lin.predict(X) + f
        return Forecast(var=np.exp(mu + 0.5 * sd ** 2), log_mu=mu, log_sd=sd)

    def kernel_str(self) -> str:
        return str(self.gp.kernel_)


# ----------------------------------------------------------------------------- LSTM
class LSTMVol(Forecaster):
    """LSTM over the last `seq_len` days of [standardised log r^2, standardised r]. Output is the log
    of the h-day variance forecast, trained with the QLIKE loss (so exp(output) targets the conditional
    mean of realised variance; no log-bias correction needed)."""

    name = "LSTM"

    def __init__(self, seq_len: int = 22, hidden: int = 16, epochs: int = 60, lr: float = 5e-3,
                 batch: int = 32, max_train: int = 1500, patience: int = 15, seed: int = 0,
                 weight_decay: float = 1e-4):
        self.L, self.hidden, self.epochs, self.lr = seq_len, hidden, epochs, lr
        self.batch, self.max_train, self.patience, self.seed = batch, max_train, patience, seed
        self.weight_decay = weight_decay
        self.net = None

    def _windows(self, r):
        F = np.column_stack([(np.log(r ** 2 + EPS) - self.m1) / self.s1, r / self.s2]).astype(np.float32)
        W = sliding_window_view(F, self.L, axis=0)            # (N-L+1, 2, L); window i ends at t=i+L-1
        return np.ascontiguousarray(np.transpose(W, (0, 2, 1)))

    def fit(self, r, h, end):
        import copy

        import torch
        import torch.nn as nn

        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        rr = np.asarray(r[: end + 1], dtype=float)
        lr2 = np.log(rr ** 2 + EPS)
        self.m1, self.s1, self.s2 = float(lr2.mean()), float(lr2.std() + 1e-8), float(rr.std() + 1e-8)
        W = self._windows(rr)
        y = np.log(future_mean_sq(rr, h) + EPS)[self.L - 1:]
        m = ~np.isnan(y)
        W, y = W[m][-self.max_train:], y[m][-self.max_train:].astype(np.float32)
        n_val = max(int(0.15 * len(y)), 30)
        Wtr, ytr, Wva, yva = W[:-n_val], y[:-n_val], W[-n_val:], y[-n_val:]

        class Net(nn.Module):
            def __init__(s, d_in, hidden):
                super().__init__()
                s.lstm = nn.LSTM(d_in, hidden, batch_first=True)
                s.head = nn.Linear(hidden, 1)

            def forward(s, x):
                o, _ = s.lstm(x)
                return s.head(o[:, -1]).squeeze(-1)

        net = Net(W.shape[2], self.hidden)
        with torch.no_grad():
            net.head.bias.fill_(float(np.log(np.mean(np.exp(ytr)))))   # QLIKE-optimal constant forecast
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def qlike(pred, target):               # pred, target are log-variances
            d = target - pred
            return (torch.exp(d) - d - 1.0).mean()

        Xtr, Ytr = torch.from_numpy(Wtr), torch.from_numpy(ytr)
        Xva, Yva = torch.from_numpy(Wva), torch.from_numpy(yva)
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(len(Xtr), generator=g)
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                qlike(net(Xtr[b]), Ytr[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(qlike(net(Xva), Yva))
            if v < best - 1e-5:
                best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net = net

    def predict(self, r, ts):
        import torch

        rr = np.asarray(r[: int(np.max(ts)) + 1], dtype=float)
        W = self._windows(rr)
        idx = np.asarray(ts) - (self.L - 1)
        with torch.no_grad():
            yhat = self.net(torch.from_numpy(W[idx])).numpy().astype(float)
        return Forecast(var=np.exp(yhat), log_mu=yhat)


def build_models(names, seed: int = 0, quick: bool = False) -> list[Forecaster]:
    registry = {
        "naive": lambda: RollingVar(22),
        "garch": lambda: Garch11(),
        "gjr": lambda: Garch11(asym=True),
        "har": lambda: LogHAR(),
        "har_l": lambda: LogHAR(leverage=True),
        "gp": lambda: GPVol(max_train=400 if quick else 750, n_restarts=0 if quick else 1, seed=seed),
        "gp_l": lambda: GPVol(max_train=400 if quick else 750, n_restarts=0 if quick else 1, seed=seed,
                              leverage=True),
        "lstm": lambda: LSTMVol(epochs=25 if quick else 60, seed=seed),
    }
    return [registry[n.lower()]() for n in names]