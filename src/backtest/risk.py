"""Position sizing and risk-control helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskParams:
    """Risk-management parameters.

    Attributes:
        stop_loss: Stop distance as fraction of entry price.
        take_profit: Target distance as fraction of entry price.
        max_leverage: Cap on absolute exposure relative to equity.
    """

    stop_loss: float = 0.07
    take_profit: float = 0.15
    max_leverage: float = 1.0


def volatility_target_position(
    signal: pd.Series, returns: pd.Series, target_vol: float = 0.15,
    window: int = 20, max_leverage: float = 1.0,
) -> pd.Series:
    """Scale signals so realised vol targets ``target_vol`` (annualised).

    The workhorse of systematic commodity funds: size inversely to recent
    volatility so each position contributes a similar risk budget.
    """
    realised = returns.rolling(window).std() * np.sqrt(252)
    scaler = (target_vol / realised).clip(upper=max_leverage)
    return (signal * scaler).fillna(0.0).rename("position")
