"""Cleaning and alignment utilities for raw commodity price series."""
from __future__ import annotations

import pandas as pd

from src.config import DEFAULT_START, EIA_SERIES
from src.data.eia_client import EIAClient


def load_prices(
    series: dict[str, str] | None = None,
    start: str = DEFAULT_START,
    client: EIAClient | None = None,
) -> pd.DataFrame:
    """Download, clean and align several EIA series into one wide panel.

    Args:
        series: Mapping of ``symbol -> EIA series ID``.
        start: ISO date; earlier rows are discarded.
        client: Optional pre-built client (useful for testing).

    Returns:
        A wide DataFrame on a continuous business-day index, one column
        per symbol, short gaps forward-filled.
    """
    series = series or EIA_SERIES
    client = client or EIAClient()
    cols = [
        client.fetch_series(sid).set_index("date")["value"].rename(sym)
        for sym, sid in series.items()
    ]
    panel = pd.concat(cols, axis=1).sort_index()
    panel = panel.loc[panel.index >= pd.Timestamp(start)]
    return _make_continuous(panel)


def _make_continuous(panel: pd.DataFrame) -> pd.DataFrame:
    """Reindex onto a gap-free business-day calendar and forward-fill holidays.

    Spot series carry weekend/holiday gaps; a continuous calendar keeps
    indicator windows and trade timing predictable. Only short gaps are
    filled (limit=5) so genuine data outages remain visible as NaN.
    """
    full_idx = pd.bdate_range(panel.index.min(), panel.index.max())
    panel = panel.reindex(full_idx)
    panel.index.name = "date"
    return panel.ffill(limit=5)
