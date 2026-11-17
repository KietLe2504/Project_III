"""Load VN price data from CSV (primary route) or, optionally, vnstock v3-style API.

Primary route: put a CSV per symbol in data/raw/<SYMBOL>.csv. The loader is tolerant:
  * delimiter auto-detected (, ; tab), UTF-8 with/without BOM
  * date column: date / time / ngay / thoi gian ...  (ISO yyyy-mm-dd or dd/mm/yyyy)
  * price column: close / dong cua / gia dong cua / price / last / adj close ...
  * numbers like "1,234.56", "1.234,56", "85.200" are parsed
Only date and close are used (everything else is ignored).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

VN30_EXAMPLE = ["VCB", "FPT", "HPG", "VNM", "MWG"]
LIMIT_PCT = 7.5   # HOSE daily price limit is +/-7%; larger moves in adjusted data = adjustment artefacts

DATE_NAMES = ["date", "time", "ngay", "thoi gian", "ngay giao dich", "datetime", "trading date"]
CLOSE_NAMES = ["close", "gia dong cua", "dong cua", "gia dieu chinh", "adj close", "adjusted close",
               "price", "gia", "last", "closing price", "close price"]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def _find_col(columns, names) -> str | None:
    cols = {_norm(c): c for c in columns}
    for n in names:                      # exact match first, in priority order
        if n in cols:
            return cols[n]
    for n in names:                      # then substring match
        for k, c in cols.items():
            if n in k:
                return c
    return None


def _parse_number(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    s = s.where(~s.isin(["", "nan", "None", "-"]))
    nn = s.dropna()
    if nn.empty:
        return pd.Series(np.nan, index=series.index)

    def conv(x: str) -> float:
        if "," in x and "." in x:                       # last separator is the decimal one
            x = x.replace(".", "").replace(",", ".") if x.rfind(",") > x.rfind(".") else x.replace(",", "")
        elif "," in x:
            x = x.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+", x) else x.replace(",", ".")
        elif "." in x and re.fullmatch(r"\d{1,3}(\.\d{3})+", x) and nn.str.fullmatch(r"\d{1,3}(\.\d{3})+").all():
            x = x.replace(".", "")                      # every value looks like 85.200 -> thousands
        return float(x)

    return s.map(lambda v: conv(v) if isinstance(v, str) else np.nan)


def _parse_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):               # epoch seconds / milliseconds
        unit = "ms" if series.dropna().abs().max() > 1e11 else "s"
        return pd.to_datetime(series, unit=unit)
    s = series.astype(str).str.strip()
    if s.str.match(r"^\d{4}-\d{2}-\d{2}").all():
        return pd.to_datetime(s.str[:10], format="%Y-%m-%d")
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def read_price_csv(path: str | Path) -> pd.DataFrame:
    df = None
    for enc in ("utf-8-sig", "utf-16", "cp1258", "latin-1"):   # PowerShell '>' writes UTF-16
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
            if df.shape[1] >= 2:
                break
        except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
    if df is None or df.shape[1] < 2:
        raise ValueError(f"{path}: khong doc duoc file (kiem tra file co noi dung CSV, dung la 'date,close,...').")
    dcol, ccol = _find_col(df.columns, DATE_NAMES), _find_col(df.columns, CLOSE_NAMES)
    if dcol is None or ccol is None:
        raise ValueError(f"{path}: khong tim thay cot ngay/gia dong cua. Cac cot hien co: {list(df.columns)}. "
                         "Doi ten cot thanh 'date' va 'close'.")
    out = pd.DataFrame({"date": _parse_dates(df[dcol]), "close": _parse_number(df[ccol])})
    out = out.dropna()
    return out[out["close"] > 0].sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _from_vnstock(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Old vnstock (<=3.x) API. vnstock 6 no longer has `Vnstock` (it needs per-broker Open API keys)."""
    try:
        from vnstock import Vnstock
    except ImportError as e:
        raise RuntimeError(
            f"Khong co file data/raw/{symbol}.csv. vnstock hien tai (>=6) khong con ham Vnstock() nen khong tu tai duoc. "
            "Hay tai CSV thu cong va luu vao data/raw/ (xem README, muc 'Du lieu thuc')."
        ) from e
    df = Vnstock().stock(symbol=symbol, source="VCI").quote.history(start=start, end=end, interval="1D")
    tcol = "time" if "time" in df.columns else "date"
    return pd.DataFrame({"date": pd.to_datetime(df[tcol]), "close": df["close"].astype(float)})


def load_prices(symbol: str, start: str = "2015-01-01", end: str = "2026-10-30",
                csv_dir: str | Path = "data/raw") -> pd.DataFrame:
    path = Path(csv_dir) / f"{symbol}.csv"
    if path.exists():
        df = read_price_csv(path)
    else:
        df = _from_vnstock(symbol, start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df.reset_index(drop=True)


def log_returns_pct(df: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex, int]:
    """Daily log-returns in %. Returns (returns, dates, n_dropped_outliers)."""
    r = 100.0 * np.log(df["close"]).diff()
    dates = pd.DatetimeIndex(df["date"])
    keep = r.notna() & (r.abs() <= LIMIT_PCT)
    n_drop = int((r.notna() & ~keep).sum())
    return r[keep].to_numpy(), dates[keep.to_numpy()], n_drop