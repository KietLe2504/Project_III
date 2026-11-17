"""Kiem tra nhanh CSV truoc khi chay thi nghiem:  python check_data.py VCB FPT HPG VNINDEX"""
import sys

import numpy as np

from volfc.data import load_prices, log_returns_pct

for sym in sys.argv[1:] or ["VCB"]:
    try:
        df = load_prices(sym, start="1990-01-01", end="2100-01-01")
    except Exception as e:
        print(f"[{sym}] LOI: {e}\n"); continue
    r, d, n_drop = log_returns_pct(df)
    gaps = d.to_series().diff().dt.days.max()
    print(f"[{sym}] {len(df)} dong gia, {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}, "
          f"{len(r)} return, loai {n_drop} outlier (>7.5%), khoang trong lon nhat {int(gaps)} ngay")
    print(f"      close: min={df['close'].min():.2f} max={df['close'].max():.2f} | "
          f"vol quy nam={np.std(r) * np.sqrt(252):.1f}% | ngay return=0: {(r == 0).mean():.1%}")
    if len(r) < 1500:
        print("      CANH BAO: < 1500 return (~6 nam); nen tai nhieu nam hon (>= 8-10 nam).")
    if n_drop > 5:
        print("      CANH BAO: nhieu outlier -> nghi gia chua dieu chinh (chia tach/co phieu thuong).")
    print()
