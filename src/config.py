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
