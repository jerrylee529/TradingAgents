"""Vendor routing tests for Tushare."""

import copy
from unittest.mock import MagicMock, patch

import pytest

import tradingagents.default_config as default_config
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.interface import VENDOR_LIST, VENDOR_METHODS, route_to_vendor


@pytest.mark.unit
class TestTushareVendorRegistration:
    def setup_method(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    def test_tushare_in_vendor_list(self):
        assert "tushare" in VENDOR_LIST

    def test_all_methods_have_tushare_impl(self):
        for method, vendors in VENDOR_METHODS.items():
            assert "tushare" in vendors, f"{method} missing tushare vendor"

    def test_route_stock_data_to_tushare(self):
        mock_get = MagicMock(return_value="ok")
        with patch.dict(
            VENDOR_METHODS["get_stock_data"],
            {"tushare": mock_get},
        ):
            result = route_to_vendor(
                "get_stock_data", "600519.SH", "2024-01-01", "2024-01-31"
            )
        assert result == "ok"
        mock_get.assert_called_once_with("600519.SH", "2024-01-01", "2024-01-31")

    def test_route_fundamentals_to_tushare(self):
        mock_get = MagicMock(return_value="fundamentals")
        with patch.dict(
            VENDOR_METHODS["get_fundamentals"],
            {"tushare": mock_get},
        ):
            result = route_to_vendor("get_fundamentals", "600519.SH", "2024-06-01")
        assert result == "fundamentals"
        mock_get.assert_called_once()
