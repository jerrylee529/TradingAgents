"""Shared helpers for Tushare Pro data access."""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from typing import Callable, Optional, TypeVar

import pandas as pd
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

_TUSHARE_PRO = None

T = TypeVar("T")

# Yahoo Finance uses .SS for Shanghai; Tushare uses .SH
_SUFFIX_ALIASES = {
    ".SS": ".SH",
}


def get_tushare_token() -> Optional[str]:
    return os.environ.get("TUSHARE_TOKEN") or os.environ.get("TUSHARE_API_TOKEN")


def get_pro():
    """Return a cached Tushare Pro API client."""
    global _TUSHARE_PRO
    token = get_tushare_token()
    if not token:
        raise RuntimeError(
            "TUSHARE_TOKEN (or TUSHARE_API_TOKEN) environment variable is not set"
        )
    if _TUSHARE_PRO is None:
        import tushare as ts

        _TUSHARE_PRO = ts.pro_api(token)
    return _TUSHARE_PRO


def normalize_ts_code(symbol: str) -> str:
    """Normalize ticker symbols to Tushare ``ts_code`` format (e.g. 600519.SH)."""
    s = symbol.strip().upper()
    for old, new in _SUFFIX_ALIASES.items():
        if s.endswith(old):
            s = s[: -len(old)] + new
            break

    if re.match(r"^\d{6}\.(SH|SZ|BJ)$", s):
        return s

    if s.endswith(".SZ"):
        return s

    if re.match(r"^\d{6}$", s):
        if s.startswith(("6", "9")):
            return f"{s}.SH"
        if s.startswith(("0", "3")):
            return f"{s}.SZ"
        if s.startswith(("4", "8")):
            return f"{s}.BJ"

    return s


def is_ashare_ts_code(ts_code: str) -> bool:
    return bool(re.match(r"^\d{6}\.(SH|SZ|BJ)$", ts_code.upper()))


def to_yyyymmdd(date_str: str) -> str:
    return date_str.replace("-", "")


def to_iso_date(value) -> str:
    s = str(value)
    if "-" in s:
        return s[:10]
    if len(s) >= 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def ts_retry(func: Callable[[], T], max_retries: int = 3, base_delay: float = 2.0) -> T:
    """Retry Tushare calls on transient failures."""
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            transient = any(
                token in message
                for token in (
                    "timeout",
                    "timed out",
                    "connection",
                    "429",
                    "too many",
                    "频率",
                    "limit",
                    "busy",
                )
            )
            if not transient or attempt >= max_retries:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                "Tushare request failed, retrying in %.0fs (attempt %d/%d): %s",
                delay,
                attempt + 1,
                max_retries,
                exc,
            )
            time.sleep(delay)
    raise last_error  # pragma: no cover


def normalize_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Map Tushare OHLCV columns to the project-standard schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    work = df.copy()
    rename_map = {}
    for col in work.columns:
        lower = str(col).lower()
        if lower in ("trade_date", "date"):
            rename_map[col] = "Date"
        elif lower == "open":
            rename_map[col] = "Open"
        elif lower == "high":
            rename_map[col] = "High"
        elif lower == "low":
            rename_map[col] = "Low"
        elif lower == "close":
            rename_map[col] = "Close"
        elif lower in ("vol", "volume"):
            rename_map[col] = "Volume"
    work = work.rename(columns=rename_map)

    if "Date" not in work.columns:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    work["Date"] = pd.to_datetime(work["Date"].astype(str), errors="coerce")
    work = work.dropna(subset=["Date"])
    work = work.sort_values("Date")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in work.columns]
    return work[cols].reset_index(drop=True)


def fetch_ohlcv_dataframe(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    adj: str = "qfq",
) -> pd.DataFrame:
    """Fetch OHLCV history for ``symbol`` between ``start_date`` and ``end_date``."""
    ts_code = normalize_ts_code(symbol)
    start = to_yyyymmdd(start_date)
    end = to_yyyymmdd(end_date)

    def _fetch():
        import tushare as ts

        pro = get_pro()
        try:
            df = ts.pro_bar(
                ts_code=ts_code,
                adj=adj,
                start_date=start,
                end_date=end,
                freq="D",
            )
            if df is not None and not df.empty:
                return normalize_ohlcv_dataframe(df)
        except Exception as exc:
            logger.debug("ts.pro_bar failed for %s: %s", ts_code, exc)

        daily = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        if daily is None or daily.empty:
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        return normalize_ohlcv_dataframe(daily)

    return ts_retry(_fetch)


def latest_trade_date_on_or_before(ts_code: str, curr_date: str) -> Optional[str]:
    """Return YYYY-MM-DD for the latest trading day <= curr_date."""
    pro = get_pro()
    end = to_yyyymmdd(curr_date)
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start = to_yyyymmdd((curr_dt - relativedelta(years=1)).strftime("%Y-%m-%d"))

    def _fetch():
        df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        if df is None or df.empty:
            return None
        dates = sorted(df["trade_date"].astype(str).tolist())
        return to_iso_date(dates[-1])

    return ts_retry(_fetch)


def format_csv_header(title: str, **meta) -> str:
    lines = [f"# {title}"]
    for key, value in meta.items():
        lines.append(f"# {key}: {value}")
    lines.append(f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    return "\n".join(lines)
