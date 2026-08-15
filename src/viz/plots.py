"""Matplotlib visualisations for report-quality output."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.engine import BacktestResult


def plot_equity(result: BacktestResult, path: Path | None = None) -> plt.Figure:
    """Equity curve with an underwater (drawdown) panel."""
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    result.equity.plot(ax=a1, color="navy", lw=1.3)
    a1.set_title(f"Equity curve — {result.strategy} "
                 f"(Sharpe {result.metrics['sharpe']}, "
                 f"MaxDD {result.metrics['max_drawdown']:.1%})")
    a1.set_ylabel("Equity"); a1.grid(alpha=0.3)
    dd = result.equity / result.equity.cummax() - 1.0
    a2.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    a2.set_ylabel("Drawdown"); a2.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_term_structure(slope: pd.Series, path: Path | None = None) -> plt.Figure:
    """Shade contango (blue) vs backwardation (red) over time."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(slope.index, slope, 0, where=slope >= 0,
                    color="steelblue", alpha=0.5, label="contango")
    ax.fill_between(slope.index, slope, 0, where=slope < 0,
                    color="tomato", alpha=0.5, label="backwardation")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title("WTI term-structure slope (CL4 vs CL1)")
    ax.set_ylabel("(deferred − front) / front"); ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_seasonality(table: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Monthly seasonality bars with std-dev whiskers."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(table.index, table["mean"], yerr=table["std"], capsize=3,
           color="slateblue", alpha=0.85)
    ax.set_title("Henry Hub natural gas — monthly seasonality")
    ax.set_ylabel("Avg price ($/MMBtu)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig
