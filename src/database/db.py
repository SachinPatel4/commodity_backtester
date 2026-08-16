"""SQLite persistence layer with a small, typed query API."""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from config import DB_PATH
from src.database.schema import SCHEMA_SQL


class Database:
    """Context-managed wrapper around a SQLite connection.

    Example:
        >>> with Database() as db:
        ...     db.initialise()
        ...     db.write_prices(panel)
    """

    def __init__(self, path: Path | str = DB_PATH) -> None:
        self.path: str = str(path)
        self.conn: sqlite3.Connection | None = None

    def __enter__(self) -> "Database":
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        # Register a safe SQRT so rolling-vol works on any SQLite build.
        self.conn.create_function(
            "SAFE_SQRT", 1, lambda x: math.sqrt(x) if x and x > 0 else 0.0
        )
        return self

    def __exit__(self, *exc: object) -> None:
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None

    def _c(self) -> sqlite3.Connection:
        if self.conn is None:
            raise RuntimeError("Use Database inside a 'with' block.")
        return self.conn

    # -- schema -----------------------------------------------------------
    def initialise(self) -> None:
        """Create all tables and indexes if absent."""
        self._c().executescript(SCHEMA_SQL)

    # -- writes -----------------------------------------------------------
    def upsert_instrument(self, symbol: str, desc: str = "", unit: str = "") -> int:
        """Insert an instrument if missing; return its primary key."""
        self._c().execute(
            """INSERT INTO instruments (symbol, description, unit)
               VALUES (?, ?, ?)
               ON CONFLICT(symbol) DO UPDATE SET
                   description = excluded.description, unit = excluded.unit;""",
            (symbol, desc, unit),
        )
        row = self._c().execute(
            "SELECT id FROM instruments WHERE symbol = ?;", (symbol,)
        ).fetchone()
        return int(row["id"])

    def write_prices(self, panel: pd.DataFrame) -> None:
        """Persist a wide price panel idempotently (upsert on conflict)."""
        for symbol in panel.columns:
            iid = self.upsert_instrument(symbol)
            rows = [
                (iid, d.strftime("%Y-%m-%d"), float(v))
                for d, v in panel[symbol].dropna().items()
            ]
            self._c().executemany(
                """INSERT INTO prices (instrument_id, date, close)
                   VALUES (?, ?, ?)
                   ON CONFLICT(instrument_id, date)
                   DO UPDATE SET close = excluded.close;""",
                rows,
            )

    def create_run(self, strategy: str, params: dict[str, Any]) -> int:
        """Register a backtest run and return its id."""
        cur = self._c().execute(
            "INSERT INTO runs (strategy, params) VALUES (?, ?);",
            (strategy, json.dumps(params, default=str)),
        )
        return int(cur.lastrowid)

    def write_trades(self, run_id: int, trades: pd.DataFrame) -> None:
        """Persist a trade ledger for a run."""
        if trades.empty:
            return
        rows = [
            (
                run_id, t.symbol, t.side,
                pd.Timestamp(t.entry_date).strftime("%Y-%m-%d"),
                pd.Timestamp(t.exit_date).strftime("%Y-%m-%d")
                if pd.notna(t.exit_date) else None,
                float(t.quantity), float(t.entry_price),
                float(t.exit_price) if pd.notna(t.exit_price) else None,
                float(t.pnl) if pd.notna(t.pnl) else None,
            )
            for t in trades.itertuples()
        ]
        self._c().executemany(
            """INSERT INTO trades (run_id, symbol, side, entry_date, exit_date,
                                   quantity, entry_price, exit_price, pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            rows,
        )

    def write_equity(self, run_id: int, equity: pd.Series) -> None:
        """Persist an equity curve for a run."""
        rows = [
            (run_id, d.strftime("%Y-%m-%d"), float(v)) for d, v in equity.items()
        ]
        self._c().executemany(
            "INSERT OR REPLACE INTO equity_curve (run_id, date, equity) "
            "VALUES (?, ?, ?);",
            rows,
        )

    # -- reads (the interview-relevant SQL) -------------------------------
    def read_prices(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """Return a wide price panel via a JOIN + pivot."""
        query = """
            SELECT p.date, i.symbol, p.close
            FROM prices AS p
            JOIN instruments AS i ON i.id = p.instrument_id
        """
        params: tuple[str, ...] = ()
        if symbols:
            placeholders = ",".join("?" * len(symbols))
            query += f" WHERE i.symbol IN ({placeholders})"
            params = tuple(symbols)
        long_df = pd.read_sql_query(
            query, self._c(), params=params, parse_dates=["date"]
        )
        return long_df.pivot(index="date", columns="symbol", values="close")

    def rolling_stats_sql(self, symbol: str, window: int = 20) -> pd.DataFrame:
        """Compute daily returns and rolling volatility *in SQL*.

        Demonstrates CTEs, ``LAG`` and a moving-frame ``AVG ... OVER`` — the
        window-function toolkit desks actually use.
        """
        query = """
            WITH base AS (
                SELECT p.date, p.close,
                       LAG(p.close) OVER (ORDER BY p.date) AS prev_close
                FROM prices AS p
                JOIN instruments AS i ON i.id = p.instrument_id
                WHERE i.symbol = :sym
            ),
            rets AS (
                SELECT date, close, (close / prev_close) - 1.0 AS daily_return
                FROM base WHERE prev_close IS NOT NULL
            )
            SELECT date, close, daily_return,
                   AVG(daily_return) OVER w AS avg_return,
                   SAFE_SQRT(
                       AVG(daily_return * daily_return) OVER w
                       - AVG(daily_return) OVER w * AVG(daily_return) OVER w
                   ) AS rolling_vol
            FROM rets
            WINDOW w AS (ORDER BY date ROWS BETWEEN :w PRECEDING AND CURRENT ROW)
            ORDER BY date;
        """
        return pd.read_sql_query(
            query, self._c(),
            params={"sym": symbol, "w": window - 1}, parse_dates=["date"],
        )

    def trade_summary(self, run_id: int) -> pd.DataFrame:
        """Per-symbol trade stats in dollars (GROUP BY + conditional aggregation)."""
        query = """
            SELECT symbol,
                   COUNT(*)                                 AS n_trades,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                   ROUND(AVG(pnl), 2)                       AS avg_pnl_gbp,
                   ROUND(SUM(pnl), 2)                       AS total_pnl_gbp,
                   ROUND(100.0 * SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)
                         / COUNT(*), 1)                     AS win_rate_pct
            FROM trades
            WHERE run_id = ?
            GROUP BY symbol
            ORDER BY total_pnl_gbp DESC;
        """
        return pd.read_sql_query(query, self._c(), params=(run_id,))

