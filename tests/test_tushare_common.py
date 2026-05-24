"""Unit tests for Tushare helpers."""

import copy
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import tradingagents.default_config as default_config
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.stockstats_utils import fetch_ohlcv_range, load_ohlcv
from tradingagents.dataflows.tushare_common import (
    fetch_ohlcv_dataframe,
    normalize_ohlcv_dataframe,
    normalize_ts_code,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.mark.unit
class TestNormalizeTsCode:
    def test_shanghai_yahoo_suffix(self):
        assert normalize_ts_code("600519.SS") == "600519.SH"

    def test_shenzhen_suffix(self):
        assert normalize_ts_code("000001.SZ") == "000001.SZ"

    def test_bare_shanghai_code(self):
        assert normalize_ts_code("600519") == "600519.SH"

    def test_bare_shenzhen_code(self):
        assert normalize_ts_code("000001") == "000001.SZ"

    def test_already_tushare_format(self):
        assert normalize_ts_code("600519.SH") == "600519.SH"


@pytest.mark.unit
class TestNormalizeOhlcvDataframe:
    def test_maps_tushare_columns(self):
        raw = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240103"],
                "open": [10.0, 10.5],
                "high": [10.8, 11.0],
                "low": [9.8, 10.2],
                "close": [10.2, 10.7],
                "vol": [1000, 1200],
            }
        )
        out = normalize_ohlcv_dataframe(raw)
        assert list(out.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
        assert len(out) == 2
        assert str(out["Date"].iloc[0].date()) == "2024-01-02"


@pytest.mark.unit
class TestFetchOhlcvDataframe:
    @patch("tradingagents.dataflows.tushare_common.ts_retry")
    @patch("tradingagents.dataflows.tushare_common.get_pro")
    def test_falls_back_to_daily(self, mock_get_pro, mock_retry):
        mock_retry.side_effect = lambda fn: fn()
        mock_pro = MagicMock()
        mock_get_pro.return_value = mock_pro
        mock_pro.daily.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102"],
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.05],
                "vol": [100],
            }
        )

        with patch("tushare.pro_bar", side_effect=RuntimeError("no pro_bar")):
            out = fetch_ohlcv_dataframe("600519.SH", "2024-01-01", "2024-01-10")

        assert len(out) == 1
        assert out["Close"].iloc[0] == 1.05


@pytest.mark.unit
class TestAshareBenchmark:
    def setup_method(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    def test_sh_suffix_maps_to_csi300(self):
        graph = MagicMock()
        graph.config = default_config.DEFAULT_CONFIG
        assert TradingAgentsGraph._resolve_benchmark(graph, "600519.SH") == "000300.SH"

    def test_sz_suffix_maps_to_sz_component(self):
        graph = MagicMock()
        graph.config = default_config.DEFAULT_CONFIG
        assert TradingAgentsGraph._resolve_benchmark(graph, "000001.SZ") == "399001.SZ"

    def test_yahoo_ss_suffix_maps_to_csi300(self):
        graph = MagicMock()
        graph.config = default_config.DEFAULT_CONFIG
        assert TradingAgentsGraph._resolve_benchmark(graph, "600519.SS") == "000300.SH"


@pytest.mark.unit
class TestVendorAwareOhlcv:
    def setup_method(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    @patch("tradingagents.dataflows.tushare_common.fetch_ohlcv_dataframe")
    def test_fetch_ohlcv_range_uses_tushare(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "Open": [10.0, 10.5],
                "High": [10.8, 11.0],
                "Low": [9.8, 10.2],
                "Close": [10.2, 10.7],
                "Volume": [1000, 1200],
            }
        )
        out = fetch_ohlcv_range("600519.SH", "2024-01-01", "2024-01-10")
        mock_fetch.assert_called_once()
        assert len(out) == 2

    @patch("tradingagents.dataflows.tushare_common.fetch_ohlcv_dataframe")
    def test_load_ohlcv_tushare_cache(self, mock_fetch, tmp_path):
        cfg = copy.deepcopy(default_config.DEFAULT_CONFIG)
        cfg["data_cache_dir"] = str(tmp_path)
        set_config(cfg)

        mock_fetch.return_value = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-02"]),
                "Open": [10.0],
                "High": [10.8],
                "Low": [9.8],
                "Close": [10.2],
                "Volume": [1000],
            }
        )
        out = load_ohlcv("600519.SH", "2024-06-01")
        assert len(out) == 1
        out2 = load_ohlcv("600519.SH", "2024-06-01")
        assert len(out2) == 1
        assert mock_fetch.call_count == 1
