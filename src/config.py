"""Central configuration for the commodity backtesting project."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"
DB_PATH: Path = DATA_DIR / "commodities.db"
DATA_DIR.mkdir(exist_ok=True)

# Get a free key at https://www.eia.gov/opendata/  then:  export EIA_API_KEY="..."
EIA_API_KEY: str = os.environ.get("EIA_API_KEY", "")
EIA_BASE_URL: str = "https://api.eia.gov/v2"

# Legacy v1 series IDs work through the /seriesid compatibility route.
EIA_SERIES: dict[str, str] = {
    "WTI":      "PET.RWTC.D",   # Crude oil WTI, $/bbl
    "BRENT":    "PET.RBRTE.D",  # Crude oil Brent, $/bbl
    "RBOB":     "PET.EER_EPMRU_PF4_Y35NY_DPG.D",  # Gasoline, $/gal
    "HEATOIL":  "PET.EER_EPD2F_PF4_Y35NY_DPG.D",  # Heating oil, $/gal
    "HENRYHUB": "NG.RNGWHHD.D",  # Natural gas Henry Hub, $/MMBtu
    "CL1":      "PET.RCLC1.D",   # WTI futures contract 1 (front)
    "CL4":      "PET.RCLC4.D",   # WTI futures contract 4 (deferred)
}

DEFAULT_START: str = "2015-01-01"
Copy
src/data/eia_client.py
"""Thin, read-only client around the U.S. EIA open-data v2 REST API."""
from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

from config import EIA_API_KEY, EIA_BASE_URL


class EIAError(RuntimeError):
    """Raised when the EIA API returns an error or unexpected payload."""


class EIAClient:
    """Minimal client for the EIA v2 ``/seriesid`` route.

    Attributes:
        api_key: A valid EIA API key.
        session: A pooled session for connection reuse.
    """

    def __init__(self, api_key: str | None = None, timeout: int = 30) -> None:
        self.api_key: str = api_key or EIA_API_KEY
        if not self.api_key:
            raise EIAError("No EIA API key. Set the EIA_API_KEY env variable.")
        self.timeout: int = timeout
        self.session: requests.Session = requests.Session()

    def fetch_series(self, series_id: str, max_retries: int = 3) -> pd.DataFrame:
        """Download the full history of one EIA series.

        Args:
            series_id: Legacy identifier, e.g. ``"PET.RWTC.D"``.
            max_retries: Attempts on transient network failures.

        Returns:
            DataFrame with ``date`` and ``value`` columns, sorted ascending.

        Raises:
            EIAError: If the request fails or the payload cannot be parsed.
        """
        url = f"{EIA_BASE_URL}/seriesid/{series_id}"
        params = {"api_key": self.api_key}
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return self._parse(resp.json())
            except (requests.RequestException, ValueError) as exc:
                if attempt == max_retries:
                    raise EIAError(f"Failed to fetch {series_id}: {exc}") from exc
                time.sleep(2 ** attempt)  # exponential backoff
        raise EIAError("unreachable")  # pragma: no cover

    @staticmethod
    def _parse(payload: dict[str, Any]) -> pd.DataFrame:
        """Convert a raw EIA JSON payload into a tidy DataFrame."""
        records = payload.get("response", {}).get("data", [])
        if not records:
            raise EIAError("EIA payload contained no data rows.")
        frame = pd.DataFrame.from_records(records)[["period", "value"]]
        frame = frame.rename(columns={"period": "date"})
        frame["date"] = pd.to_datetime(frame["date"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        return frame.dropna().sort_values("date").reset_index(drop=True)
