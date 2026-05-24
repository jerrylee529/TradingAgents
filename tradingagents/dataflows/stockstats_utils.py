import time
import logging

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)


def _get_core_stock_vendor() -> str:
    config = get_config()
    tool_vendors = config.get("tool_vendors", {})
    if "get_stock_data" in tool_vendors:
        return tool_vendors["get_stock_data"].split(",")[0].strip()
    return config.get("data_vendors", {}).get("core_stock_apis", "yfinance").split(",")[0].strip()


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


def _load_ohlcv_yfinance(symbol: str, start_str: str, end_str: str, data_file: str) -> pd.DataFrame:
    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
    else:
        data = yf_retry(lambda: yf.download(
            symbol,
            start=start_str,
            end=end_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        ))
        data = data.reset_index()
        data.to_csv(data_file, index=False, encoding="utf-8")
    return data


def _load_ohlcv_tushare(symbol: str, start_str: str, end_str: str, data_file: str) -> pd.DataFrame:
    from .tushare_common import fetch_ohlcv_dataframe, normalize_ts_code

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
    else:
        ts_code = normalize_ts_code(symbol)
        data = fetch_ohlcv_dataframe(ts_code, start_str, end_str)
        data.to_csv(data_file, index=False, encoding="utf-8")
    return data


def fetch_ohlcv_range(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OHLCV for a date range using the configured core stock vendor."""
    vendor = _get_core_stock_vendor()
    if vendor == "tushare":
        from .tushare_common import fetch_ohlcv_dataframe, normalize_ts_code

        data = fetch_ohlcv_dataframe(normalize_ts_code(symbol), start_date, end_date)
    else:
        data = yf_retry(lambda: yf.download(
            symbol,
            start=start_date,
            end=end_date,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        ))
        data = data.reset_index()
    return _clean_dataframe(data)


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads 5 years of data up to today and caches per symbol. On
    subsequent calls the cache is reused. Rows after curr_date are
    filtered out so backtests never see future prices.
    """
    # Reject ticker values that would escape the cache directory when
    # interpolated into the cache filename (e.g. ``../../tmp/x``).
    safe_symbol = safe_ticker_component(symbol)

    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)
    vendor = _get_core_stock_vendor()

    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    vendor_tag = "Tushare" if vendor == "tushare" else "YFin"
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_symbol}-{vendor_tag}-data-{start_str}-{end_str}.csv",
    )

    if vendor == "tushare":
        data = _load_ohlcv_tushare(symbol, start_str, end_str, data_file)
    else:
        data = _load_ohlcv_yfinance(symbol, start_str, end_str, data_file)

    data = _clean_dataframe(data)

    # Filter to curr_date to prevent look-ahead bias in backtesting
    data = data[data["Date"] <= curr_date_dt]

    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop financial statement columns (fiscal period timestamps) after curr_date.

    yfinance financial statements use fiscal period end dates as columns.
    Columns after curr_date represent future data and are removed to
    prevent look-ahead bias.
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        data = load_ohlcv(symbol, curr_date)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
