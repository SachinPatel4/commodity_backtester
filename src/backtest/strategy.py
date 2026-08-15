"""Strategy interface and reference implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.analysis import indicators


class Strategy(ABC):
    """Abstract base: map features to a target position in ``[-1, 1]``."""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return desired position (-1..1) per date; 'close' must be present."""


class SmaCrossStrategy(Strategy):
    """Trend-follower: long when fast SMA > slow SMA (optionally short)."""

    name = "sma_cross"

    def __init__(self, fast: int = 20, slow: int = 50, allow_short: bool = True):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast, self.slow, self.allow_short = fast, slow, allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        price = data["close"]
        fast, slow = indicators.sma(price, self.fast), indicators.sma(price, self.slow)
        valid = fast.notna() & slow.notna()
        sig = pd.Series(0.0, index=price.index)
        sig[valid & (fast > slow)] = 1.0
        if self.allow_short:
            sig[valid & (fast <= slow)] = -1.0
        return sig.rename("signal")


class MeanReversionZScore(Strategy):
    """Fade extremes: short high z-score, long low z-score. Good for spreads."""

    name = "zscore_reversion"

    def __init__(self, window: int = 20, entry: float = 1.5):
        self.window, self.entry = window, entry

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        z = indicators.zscore(data["close"], self.window)
        sig = pd.Series(0.0, index=data.index)
        sig[z > self.entry] = -1.0
        sig[z < -self.entry] = 1.0
        return sig.rename("signal")
