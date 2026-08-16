"""Position sizing and risk-control helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskParams:
    """Risk-management parameters (dollar-based, mode-agnostic).

    Attributes:
        stop_loss: Cut a trade when its mark-to-market loss exceeds this
            fraction of initial capital (e.g. 0.10 = -10%). None disables.
        take_profit: Close a trade at this profit fraction of initial capital.
            None disables (sensible for trend-following).
        max_leverage: Cap on absolute exposure (weight) in returns mode.
    """

    stop_loss: float | None = 0.10
    take_profit: float | None = None
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
