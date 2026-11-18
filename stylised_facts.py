"""Quan sat 3 stylised facts (tu cum, duoi day, don bay) tren du lieu that.

  python stylised_facts.py --symbols VCB FPT HPG VNM VNINDEX

Ket qua: results/stylised/fig_<ma>.png, results/stylised/stylised_facts.csv,
va in san cac dong LaTeX de dan vao bao cao.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm

from volfc.data import load_prices, log_returns_pct

H = 5              # cua so "volatility sau do" cho hieu ung don bay
LAGS = (1, 5, 22)


def acf(x, k):
    x = np.asarray(x, float) - np.mean(x)
    return float(np.dot(x[:-k], x[k:]) / np.dot(x, x))


def future_sq(r, h=H):
    """mean r^2 over t+1..t+h (NaN o cuoi)."""
    c = np.concatenate([[0.0], np.cumsum(r ** 2)])
    out = np.full(len(r), np.nan)
    i = np.arange(len(r) - h)
    out[i] = (c[i + 1 + h] - c[i + 1]) / h
    return out


def decile_stats(r, h=H):
    """Chia cac ngay thanh 10 nhom theo loi suat hom nay (dung thu hang de khong bi trung bien
    khi co nhieu ngay loi suat bang 0). Tra ve: trung binh r^2 cua h ngay sau, trung binh loi suat, trung binh chung."""
    fut = future_sq(r, h)
    m = ~np.isnan(fut)
    rr, ff = r[m], fut[m]
    grp = pd.qcut(pd.Series(rr).rank(method="first"), 10, labels=False).to_numpy()
    mean_fut = np.array([ff[grp == g].mean() for g in range(10)])
    mean_ret = np.array([rr[grp == g].mean() for g in range(10)])
    return mean_fut, mean_ret, float(ff.mean())


def stats(r):
    n = len(r)
    z = (r - r.mean()) / r.std()
    dec, dret, overall = decile_stats(r)
    maxz = float(np.abs(z).max())
    return {
        "n": n,
        "ann_vol": float(r.std() * np.sqrt(252)),
        "zero_share": float((r == 0).mean()),
        "kurt": float(kurtosis(r)),
        "acf": {"r": [acf(r, k) for k in LAGS],
                "abs": [acf(np.abs(r), k) for k in LAGS],
                "sq": [acf(r ** 2, k) for k in LAGS]},
        "band": float(1.96 / np.sqrt(n)),
        "tails": [(int((np.abs(z) > k).sum()), n * 2 * norm.sf(k)) for k in (2, 3, 4)],
        "maxz": maxz, "pmax": float(2 * norm.sf(maxz)),
        "dec": dec, "dret": dret, "overall": overall,
    }


def tex_num(x, nd=3):
    return f"${x:.{nd}f}$" if x < 0 else f"{x:.{nd}f}"


def print_latex(name, st):
    print(f"\n% ===================== {name} (n = {st['n']} returns) =====================")
    print("% ACF table body (lag 1, 5, 22)")
    for lab, key in (("$r_t$", "r"), ("$|r_t|$", "abs"), ("$r_t^2$", "sq")):
        print(f"{lab} & " + " & ".join(tex_num(v, 3) for v in st["acf"][key]) + r" \\")
    print(f"% 95% band of an uncorrelated series: +-{st['band']:.3f}")
    print("% Tails table body (observed, expected if normal)")
    for k, (obs, exp) in zip((2, 3, 4), st["tails"]):
        print(f"$|z|>{k}$ & {obs} & {exp:.1f} \\\\")
    print(f"% excess kurtosis = {st['kurt']:.2f}; largest |z| = {st['maxz']:.2f}; normal probability of such a day = {st['pmax']:.1e}")
    print("% Decile table body (D1 = worst days ... D10 = best days)")
    print("Next 5 days & " + " & ".join(f"{v:.2f}" for v in st["dec"]) + r" \\")
    print(f"% overall mean = {st['overall']:.2f}; D1 average return = {st['dret'][0]:.2f}; "
          f"D10 average return = {st['dret'][-1]:.2f}; D1 / D10 = {st['dec'][0] / st['dec'][-1]:.2f}")


def plot(r, name, out):
    n = len(r)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    a = ax[0, 0]
    sel = slice(max(0, n - 750), n)
    a.plot(np.arange(n)[sel], r[sel], lw=0.7, color="#1f77b4")
    a.axhline(0, color="k", lw=0.5)
    a.set_title("(a) Daily returns: calm and stormy stretches (clustering)")
    a.set_xlabel("day"); a.set_ylabel("return (%)")

    b = ax[0, 1]
    ks = np.arange(1, 31)
    b.plot(ks, [acf(r, k) for k in ks], "o-", ms=3, color="#999999", label="return r")
    b.plot(ks, [acf(np.abs(r), k) for k in ks], "o-", ms=3, color="#d62728", label="|r|")
    b.plot(ks, [acf(r ** 2, k) for k in ks], "o-", ms=3, color="#ff7f0e", label="r$^2$")
    band = 1.96 / np.sqrt(n)
    b.axhspan(-band, band, color="#cccccc", alpha=0.5, label="~0 (95% band)")
    b.set_title("(b) Autocorrelation: sign unpredictable, size predictable")
    b.set_xlabel("lag (days)"); b.set_ylabel("ACF"); b.legend(fontsize=7, loc="center right")

    c = ax[1, 0]
    z = (r - r.mean()) / r.std()
    c.hist(z, bins=np.linspace(-8, 8, 65), density=True, color="#1f77b4", alpha=0.6, label="data (standardised)")
    xs = np.linspace(-8, 8, 400)
    c.plot(xs, norm.pdf(xs), "k--", lw=1.2, label="normal")
    c.set_yscale("log"); c.set_ylim(1e-5, 1)
    c.set_xlabel("standardised return"); c.set_ylabel("density (log scale)")
    c.set_title(f"(c) Heavy tails: excess kurtosis = {kurtosis(r):.2f}; |z|>3 on {100 * (np.abs(z) > 3).mean():.2f}% of days (normal: 0.27%)", fontsize=9)
    c.legend(fontsize=8)

    d = ax[1, 1]
    dec, _, overall = decile_stats(r)
    d.bar(np.arange(10), dec, color=["#d62728" if i < 5 else "#1f77b4" for i in range(10)])
    d.axhline(overall, color="k", ls="--", lw=0.8, label="overall mean")
    d.set_xticks(np.arange(10)); d.set_xticklabels([f"D{i + 1}" for i in range(10)])
    d.set_xlabel("today's return decile (D1 = worst days, D10 = best days)")
    d.set_ylabel(f"mean r$^2$ over the next {H} days")
    d.set_title("(d) Leverage: after bad days, later volatility is higher"); d.legend(fontsize=8)

    fig.suptitle(f"Stylised facts: {name}  (n = {n} days)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=140); plt.close(fig)


ap = argparse.ArgumentParser()
ap.add_argument("--symbols", nargs="+", default=["VCB", "FPT", "HPG", "VNM", "VNINDEX"])
ap.add_argument("--csv-dir", default="data/raw")
ap.add_argument("--out", default="results/stylised")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

rows = []
for s in a.symbols:
    try:
        r = log_returns_pct(load_prices(s, "1990-01-01", "2100-01-01", a.csv_dir))[0]
    except Exception as e:
        print(f"[{s}] bo qua: {e}")
        continue
    st = stats(r)
    plot(r, s, out / f"fig_{s}.png")
    print_latex(s, st)
    rows.append({"series": s, "n": st["n"], "ann_vol_%": st["ann_vol"], "zero_return_%": 100 * st["zero_share"],
                 "excess_kurtosis": st["kurt"],
                 "acf1_r": st["acf"]["r"][0], "acf1_abs": st["acf"]["abs"][0], "acf1_sq": st["acf"]["sq"][0],
                 "acf22_abs": st["acf"]["abs"][2],
                 "days>2sd": st["tails"][0][0], "days>3sd": st["tails"][1][0], "days>4sd": st["tails"][2][0],
                 "max|z|": st["maxz"],
                 "D1": st["dec"][0], "D10": st["dec"][-1], "D1/D10": st["dec"][0] / st["dec"][-1]})

if rows:
    tab = pd.DataFrame(rows).set_index("series")
    tab.to_csv(out / "stylised_facts.csv")
    print("\n================ SUMMARY ================")
    print(tab.round(3).T.to_string())
    print(f"\nSaved to {out}/")