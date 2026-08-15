"""Commodity-specific analytics: crack spread, term structure, seasonality."""
from __future__ import annotations

import numpy as np
import pandas as pd


def crack_spread_321(
    crude: pd.Series, gasoline: pd.Series, heating_oil: pd.Series,
    gallons_per_barrel: float = 42.0,
) -> pd.Series:
    """3:2:1 crack spread (refinery-margin proxy) in USD/barrel.

    Three barrels of crude are assumed refined into two of gasoline and one
    of distillate. EIA product prices are USD/gallon, scaled to USD/barrel.

    Args:
        crude: Crude price, USD/bbl (e.g. WTI).
        gasoline: Gasoline price, USD/gal (e.g. RBOB).
        heating_oil: Distillate price, USD/gal.
    """
    gas_bbl = gasoline * gallons_per_barrel
    ho_bbl = heating_oil * gallons_per_barrel
    return ((2.0 * gas_bbl + ho_bbl - 3.0 * crude) / 3.0).rename("crack_321")


def term_structure_slope(front: pd.Series, deferred: pd.Series) -> pd.Series:
    """Curve slope ``(deferred - front) / front``.

    Positive => contango (carry cost); negative => backwardation.
    """
    return ((deferred - front) / front).rename("term_slope")


def market_state(slope: pd.Series, flat_band: float = 0.001) -> pd.Series:
    """Label each day 'contango', 'backwardation' or 'flat'."""
    return pd.cut(
        slope, bins=[-np.inf, -flat_band, flat_band, np.inf],
        labels=["backwardation", "flat", "contango"],
    )


def monthly_seasonality(series: pd.Series) -> pd.DataFrame:
    """Average value, std and count by calendar month across all years."""
    df = series.to_frame("value")
    df["month"] = df.index.month
    out = df.groupby("month")["value"].agg(["mean", "std", "count"])
    out.index = pd.to_datetime(out.index, format="%m").strftime("%b")
    return out
