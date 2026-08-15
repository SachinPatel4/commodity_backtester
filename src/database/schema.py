"""SQLite schema (DDL) for the commodity analytics store."""
from __future__ import annotations

SCHEMA_SQL: str = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS instruments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL UNIQUE,
    description TEXT,
    unit        TEXT
);

CREATE TABLE IF NOT EXISTS prices (
    instrument_id INTEGER NOT NULL,
    date          TEXT    NOT NULL,            -- 'YYYY-MM-DD'
    close         REAL    NOT NULL,
    PRIMARY KEY (instrument_id, date),
    FOREIGN KEY (instrument_id) REFERENCES instruments(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy   TEXT NOT NULL,
    params     TEXT,                            -- JSON
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    symbol      TEXT    NOT NULL,
    side        TEXT    NOT NULL CHECK (side IN ('LONG','SHORT')),
    entry_date  TEXT    NOT NULL,
    exit_date   TEXT,
    quantity    REAL    NOT NULL,
    entry_price REAL    NOT NULL,
    exit_price  REAL,
    pnl         REAL,                           -- trade return fraction
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS equity_curve (
    run_id INTEGER NOT NULL,
    date   TEXT    NOT NULL,
    equity REAL    NOT NULL,
    PRIMARY KEY (run_id, date),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
"""
