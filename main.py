"""End-to-end pipeline: ingest -> store -> analyse -> backtest -> report."""
from __future__ import annotations

import pandas as pd

from src.analysis import commodities
from src.backtest.engine import Backtester
from src.backtest.risk import RiskParams
from src.backtest.strategy import MeanReversionZScore, SmaCrossStrategy
from src.data.loader import load_prices
from src.database.db import Database
from src.viz import plots


def main() -> None:
    """Run the full workflow and print a summary."""
    # 1. Ingest and persist
    panel = load_prices()
    with Database() as db:
        db.initialise()
        db.write_prices(panel)
        prices = db.read_prices()                 # read back via SQL JOIN

    # 2. Commodity analytics
    crack = commodities.crack_spread_321(
        prices["WTI"], prices["RBOB"], prices["HEATOIL"]
    ).dropna()
    slope = commodities.term_structure_slope(prices["CL1"], prices["CL4"]).dropna()
    seasonality = commodities.monthly_seasonality(prices["HENRYHUB"].dropna())

    # 3. Backtests
    bt = Backtester(commission=0.0003, risk=RiskParams(stop_loss=0.08))

    wti = prices[["WTI"]].rename(columns={"WTI": "close"}).dropna()
    wti.attrs["symbol"] = "WTI"
    trend = bt.run(wti, SmaCrossStrategy(fast=20, slow=60))

    crack_df = crack.to_frame("close")
    crack_df.attrs["symbol"] = "CRACK321"
    reversion = bt.run(crack_df, MeanReversionZScore(window=20, entry=1.5))

    # 4. Persist results and query them back with SQL
    with Database() as db:
        for result in (trend, reversion):
            run_id = db.create_run(result.strategy, result.params)
            db.write_trades(run_id, result.trades)
            db.write_equity(run_id, result.equity)
            print(f"\n=== {result.strategy} (run {run_id}) ===")
            print(result.metrics)
            print(db.trade_summary(run_id).to_string(index=False))
        print("\nWTI rolling stats (SQL window functions):")
        print(db.rolling_stats_sql("WTI", window=20).tail().to_string(index=False))

    # 5. Report
    plots.plot_equity(trend, path="data/equity_wti_trend.png")
    plots.plot_term_structure(slope, path="data/term_structure.png")
    plots.plot_seasonality(seasonality, path="data/gas_seasonality.png")
    print("\nCharts saved to data/. Done.")


if __name__ == "__main__":
    main()
