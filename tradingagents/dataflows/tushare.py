"""Tushare Pro vendor implementations for A-share market data."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional

import pandas as pd
from dateutil.relativedelta import relativedelta

from .stockstats_utils import filter_financials_by_date
from .tushare_common import (
    fetch_ohlcv_dataframe,
    format_csv_header,
    get_pro,
    is_ashare_ts_code,
    latest_trade_date_on_or_before,
    normalize_ts_code,
    to_iso_date,
    to_yyyymmdd,
    ts_retry,
)
from .y_finance import get_stock_stats_indicators_window

# Reuse indicator window logic; it reads OHLCV via load_ohlcv (vendor-aware).
get_tushare_stock_stats_indicators_window = get_stock_stats_indicators_window


def get_tushare_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    ts_code = normalize_ts_code(symbol)
    data = fetch_ohlcv_dataframe(ts_code, start_date, end_date)

    if data.empty:
        return (
            f"No data found for symbol '{symbol}' ({ts_code}) "
            f"between {start_date} and {end_date}"
        )

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    csv_string = data.to_csv(index=False)
    header = format_csv_header(
        f"Stock data for {ts_code} from {start_date} to {end_date}",
        Total_records=len(data),
    )
    return header + csv_string


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
):
    ts_code = normalize_ts_code(ticker)
    try:
        pro = get_pro()
        basic = ts_retry(
            lambda: pro.stock_basic(
                ts_code=ts_code,
                fields="ts_code,name,area,industry,market,list_date",
            )
        )
        if basic is None or basic.empty:
            return f"No fundamentals data found for symbol '{ticker}' ({ts_code})"

        row = basic.iloc[0]
        trade_date = None
        daily_row = None
        if curr_date:
            trade_date = latest_trade_date_on_or_before(ts_code, curr_date)
            if trade_date:
                daily = ts_retry(
                    lambda: pro.daily_basic(
                        ts_code=ts_code,
                        trade_date=to_yyyymmdd(trade_date),
                        fields=(
                            "ts_code,trade_date,close,turnover_rate,pe_ttm,pb,ps_ttm,"
                            "dv_ttm,total_mv,circ_mv"
                        ),
                    )
                )
                if daily is not None and not daily.empty:
                    daily_row = daily.iloc[0]

        fina_row = None
        fina = ts_retry(
            lambda: pro.fina_indicator(
                ts_code=ts_code,
                fields=(
                    "ts_code,end_date,eps,dt_eps,bps,roe,roa,grossprofit_margin,"
                    "netprofit_margin,debt_to_assets,current_ratio,quick_ratio,"
                    "or_yoy,netprofit_yoy"
                ),
            )
        )
        if fina is not None and not fina.empty:
            fina_sorted = fina.sort_values("end_date", ascending=False)
            if curr_date:
                fina_sorted = fina_sorted[
                    fina_sorted["end_date"].astype(str)
                    <= to_yyyymmdd(curr_date)
                ]
            if not fina_sorted.empty:
                fina_row = fina_sorted.iloc[0]

        fields = [
            ("Name", row.get("name")),
            ("TS Code", row.get("ts_code")),
            ("Area", row.get("area")),
            ("Industry", row.get("industry")),
            ("Market", row.get("market")),
            ("List Date", row.get("list_date")),
        ]
        if daily_row is not None:
            fields.extend(
                [
                    ("Close", daily_row.get("close")),
                    ("PE Ratio (TTM)", daily_row.get("pe_ttm")),
                    ("Price to Book", daily_row.get("pb")),
                    ("PS (TTM)", daily_row.get("ps_ttm")),
                    ("Dividend Yield (TTM)", daily_row.get("dv_ttm")),
                    ("Total Market Cap", daily_row.get("total_mv")),
                    ("Circulating Market Cap", daily_row.get("circ_mv")),
                    ("Turnover Rate", daily_row.get("turnover_rate")),
                    ("Valuation Date", daily_row.get("trade_date")),
                ]
            )
        if fina_row is not None:
            fields.extend(
                [
                    ("EPS", fina_row.get("eps")),
                    ("BPS", fina_row.get("bps")),
                    ("ROE", fina_row.get("roe")),
                    ("ROA", fina_row.get("roa")),
                    ("Gross Profit Margin", fina_row.get("grossprofit_margin")),
                    ("Net Profit Margin", fina_row.get("netprofit_margin")),
                    ("Debt to Assets", fina_row.get("debt_to_assets")),
                    ("Current Ratio", fina_row.get("current_ratio")),
                    ("Quick Ratio", fina_row.get("quick_ratio")),
                    ("Revenue YoY", fina_row.get("or_yoy")),
                    ("Net Profit YoY", fina_row.get("netprofit_yoy")),
                    ("Latest Report Period", fina_row.get("end_date")),
                ]
            )

        lines = [f"{label}: {value}" for label, value in fields if value is not None]
        header = format_csv_header(f"Company Fundamentals for {ts_code}")
        return header + "\n".join(lines)
    except Exception as exc:
        return f"Error retrieving fundamentals for {ticker}: {exc}"


def _pivot_financial_statement(df: pd.DataFrame, curr_date: Optional[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    meta_cols = {
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "update_flag",
        "report_date",
    }
    value_cols = [c for c in df.columns if c not in meta_cols]
    if "end_date" not in df.columns or not value_cols:
        return pd.DataFrame()

    work = df.copy()
    work["end_date"] = work["end_date"].astype(str)
    if curr_date:
        work = work[work["end_date"] <= to_yyyymmdd(curr_date)]

    if work.empty:
        return pd.DataFrame()

    pivoted = work.set_index("end_date")[value_cols].T
    pivoted.columns = pd.to_datetime(pivoted.columns.astype(str), format="%Y%m%d", errors="coerce")
    pivoted = pivoted.loc[:, pivoted.columns.notna()]
    return filter_financials_by_date(pivoted, curr_date)


def _get_financial_statement(
    ticker: str,
    freq: str,
    curr_date: Optional[str],
    report_type: str,
    title: str,
):
    ts_code = normalize_ts_code(ticker)
    try:
        pro = get_pro()
        fetcher = {
            "balance": pro.balancesheet,
            "income": pro.income,
            "cashflow": pro.cashflow,
        }[report_type]

        raw = ts_retry(lambda: fetcher(ts_code=ts_code))
        if raw is None or raw.empty:
            return f"No {title.lower()} data found for symbol '{ticker}' ({ts_code})"

        if "end_date" in raw.columns:
            end_dates = raw["end_date"].astype(str)
            if freq.lower() == "annual":
                raw = raw[end_dates.str.endswith("1231")]
            else:
                raw = raw[
                    end_dates.str.endswith(("0331", "0630", "0930", "1231"))
                ]

        data = _pivot_financial_statement(raw, curr_date)
        if data.empty:
            return f"No {title.lower()} data found for symbol '{ticker}' ({ts_code})"

        header = format_csv_header(f"{title} for {ts_code} ({freq})")
        return header + data.to_csv()
    except Exception as exc:
        return f"Error retrieving {title.lower()} for {ticker}: {exc}"


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
):
    return _get_financial_statement(ticker, freq, curr_date, "balance", "Balance Sheet")


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
):
    return _get_financial_statement(ticker, freq, curr_date, "cashflow", "Cash Flow")


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
):
    return _get_financial_statement(ticker, freq, curr_date, "income", "Income Statement")


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"],
):
    ts_code = normalize_ts_code(ticker)
    try:
        pro = get_pro()
        data = ts_retry(lambda: pro.stk_holdertrade(ts_code=ts_code))
        if data is None or data.empty:
            return f"No insider transactions data found for symbol '{ticker}' ({ts_code})"
        header = format_csv_header(f"Insider Transactions for {ts_code}")
        return header + data.to_csv(index=False)
    except Exception as exc:
        return f"Error retrieving insider transactions for {ticker}: {exc}"


def get_news_tushare(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    ts_code = normalize_ts_code(ticker)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    try:
        pro = get_pro()
        blocks = []

        if is_ashare_ts_code(ts_code):
            anns = ts_retry(
                lambda: pro.anns(
                    ts_code=ts_code,
                    start_date=to_yyyymmdd(start_date),
                    end_date=to_yyyymmdd(end_date),
                )
            )
            if anns is not None and not anns.empty:
                for _, row in anns.iterrows():
                    title = row.get("title") or row.get("ann_title") or "Announcement"
                    pub = row.get("ann_date") or row.get("pub_date") or ""
                    url = row.get("url") or ""
                    summary = row.get("content") or row.get("summary") or ""
                    blocks.append(
                        f"### {title} (source: Company Announcement)\n"
                        f"Date: {pub}\n"
                        f"{summary}\n"
                        f"Link: {url}\n"
                    )

        try:
            major = ts_retry(
                lambda: pro.major_news(
                    src="",
                    start_date=to_yyyymmdd(start_date),
                    end_date=to_yyyymmdd(end_date),
                )
            )
            if major is not None and not major.empty:
                for _, row in major.iterrows():
                    title = row.get("title", "News")
                    pub = row.get("pub_time") or row.get("pub_date") or ""
                    blocks.append(
                        f"### {title} (source: Major News)\n"
                        f"Date: {pub}\n"
                        f"{row.get('content', '')}\n"
                    )
        except Exception:
            pass

        if not blocks:
            return (
                f"No news found for {ticker} ({ts_code}) between {start_date} and {end_date}. "
                "Tushare news coverage may require higher API permissions."
            )

        return f"## {ts_code} News, from {start_date} to {end_date}:\n\n" + "\n".join(blocks)
    except Exception as exc:
        return f"Error fetching news for {ticker}: {exc}"


def get_global_news_tushare(
    curr_date: str,
    look_back_days: Optional[int] = None,
    limit: Optional[int] = None,
) -> str:
    from .config import get_config

    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    try:
        pro = get_pro()
        blocks = []
        try:
            major = ts_retry(
                lambda: pro.major_news(
                    src="",
                    start_date=to_yyyymmdd(start_date),
                    end_date=to_yyyymmdd(curr_date),
                )
            )
            if major is not None and not major.empty:
                for _, row in major.head(limit).iterrows():
                    title = row.get("title", "News")
                    pub = row.get("pub_time") or row.get("pub_date") or ""
                    blocks.append(
                        f"### {title} (source: Major News)\n"
                        f"Date: {pub}\n"
                        f"{row.get('content', '')}\n"
                    )
        except Exception:
            pass

        if not blocks:
            return (
                f"No global news found for {curr_date} via Tushare. "
                "Consider switching news_data vendor to yfinance for macro headlines."
            )

        return (
            f"## Global Market News, from {start_date} to {curr_date}:\n\n"
            + "\n".join(blocks[:limit])
        )
    except Exception as exc:
        return f"Error fetching global news: {exc}"
