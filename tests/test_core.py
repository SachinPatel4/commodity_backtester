"""Unit tests — run with `pytest`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis import commodities, indicators
from src.backtest.engine import Backtester
from src.backtest.strategy import SmaCrossStrategy


def test_rsi_bounded() -> None:
    s = pd.Series(np.random.RandomState(0).normal(0, 1, 300)).cumsum() + 100
    r = indicators.rsi(s, 14).dropna()
    assert r.between(0, 100).all()


def test_crack_positive_when_products_rich() -> None:
    idx = pd.date_range("2020-01-01", periods=3)
    crude = pd.Series([50.0] * 3, index=idx)
    gas = pd.Series([2.0] * 3, index=idx)
    ho = pd.Series([2.0] * 3, index=idx)
    assert (commodities.crack_spread_321(crude, gas, ho) > 0).all()


def test_backtester_runs_and_reports() -> None:
    idx = pd.bdate_range("2021-01-01", periods=400)
    price = pd.Series(np.linspace(50, 90, 400), index=idx)  # uptrend
    df = price.to_frame("close"); df.attrs["symbol"] = "TEST"
    res = Backtester().run(df, SmaCrossStrategy(10, 30, allow_short=False))
    assert len(res.equity) == len(idx)
    assert set(res.metrics) >= {"sharpe", "max_drawdown", "total_return"}
