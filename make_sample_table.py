"""In bang mau (LaTeX) cho phan Data Sample cua bao cao.

  python make_sample_table.py --heston baseline --start 300 --rows 6          # chuoi Heston (co dap an IV)
  python make_sample_table.py --symbol VCB --csv-dir data/raw --start 300 --rows 6   # du lieu that
Dan phan in ra vao report.tex."""
from __future__ import annotations

import argparse

import numpy as np

from volfc.data import load_prices, log_returns_pct
from volfc.features import EPS, future_mean, future_mean_sq, har_features
from volfc.heston import REGIMES, simulate_heston

ap = argparse.ArgumentParser()
ap.add_argument("--heston", choices=list(REGIMES), default=None)
ap.add_argument("--symbol", default=None)
ap.add_argument("--csv-dir", default="data/raw")
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--n", type=int, default=2100)
ap.add_argument("--start", type=int, default=300)
ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--h", type=int, default=5)
a = ap.parse_args()

if a.heston:
    sim = simulate_heston(a.n, REGIMES[a.heston], seed=a.seed)
    r, truth, dates = sim.returns, future_mean(sim.true_var, a.h), None
else:
    df = load_prices(a.symbol, "1990-01-01", "2100-01-01", a.csv_dir)
    r, d, _ = log_returns_pct(df)
    truth, dates = None, d
X = har_features(r)
q = future_mean_sq(r, a.h)
y = np.log(q + EPS)
rows = []
for t in range(a.start, a.start + a.rows):
    lead = f"{t}" if dates is None else dates[t].strftime("%Y-%m-%d")
    cells = [lead, f"{r[t]:.2f}", f"{X[t, 0]:.2f}", f"{X[t, 1]:.2f}", f"{X[t, 2]:.2f}", f"{q[t]:.2f}", f"{y[t]:.2f}"]
    if truth is not None:
        cells.append(f"{truth[t]:.2f}")
    rows.append(" & ".join(c.replace("-", "$-$") if c.startswith("-") else c for c in cells) + r" \\")
head = ("Day" if dates is None else "Date") + r" & $r_t$ & $\ln(\bar q^{(1)}_t{+}\varepsilon)$ & $\ln(\bar q^{(5)}_t{+}\varepsilon)$ & $\ln(\bar q^{(22)}_t{+}\varepsilon)$ & $\tilde q_{t+1:t+" + str(a.h) + r"}$ & $y_t$"
if truth is not None:
    head += r" & $\overline{IV}_{t+1:t+" + str(a.h) + "}$"
ncol = 8 if truth is not None else 7
print(r"\begin{tabular}{@{}" + "c" * ncol + r"@{}}")
print(r"\toprule"); print(head + r" \\"); print(r"\midrule")
print("\n".join(rows)); print(r"\bottomrule"); print(r"\end{tabular}")
