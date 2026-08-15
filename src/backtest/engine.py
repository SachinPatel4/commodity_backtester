"""Event-driven backtesting engine with basic risk controls."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.risk import RiskParams
from src.backtest.strategy import Strategy


@dataclass
class BacktestResult:
    """Outcome of a single backtest run."""

    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]
    strategy: str
    params: dict[str, Any] = field(default_factory=dict)


class Backtester:
    """Simulate a single-instrument strategy bar by bar.

    Signals are lagged implicitly (a position decided at close of day *t*
    earns day *t+1*'s return) to avoid look-ahead bias.

    Args:
        initial_cash: Starting equity.
        commission: Proportional cost on traded exposure change.
        risk: Stop/target/leverage parameters.
        periods_per_year: Annualisation factor (252 business days).
    """

    def __init__(
        self, initial_cash: float = 100_000.0, commission: float = 0.0002,
        risk: RiskParams | None = None, periods_per_year: int = 252,
    ) -> None:
        self.initial_cash = initial_cash
        self.commission = commission
        self.risk = risk or RiskParams()
        self.periods_per_year = periods_per_year

    def run(self, data: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        """Execute ``strategy`` over ``data`` (needs a 'close' column)."""
        if "close" not in data.columns:
            raise ValueError("data must contain a 'close' column")

        prices = data["close"].astype(float)
        signals = (strategy.generate_signals(data)
                   .reindex(prices.index).fillna(0.0))
        symbol = str(data.attrs.get("symbol", "ASSET"))
        idx = prices.index

        equity, pos = self.initial_cash, 0.0
        entry_price, entry_date = float("nan"), None
        equity_hist: list[float] = [equity]
        ret_hist: list[float] = [0.0]
        trades: list[dict[str, Any]] = []

        for i in range(1, len(idx)):
            price, prev = prices.iat[i], prices.iat[i - 1]
            gross = pos * (price / prev - 1.0)          # carry position return

            # stop-loss / take-profit on the open position
            forced_exit = False
            if pos != 0.0 and not math.isnan(entry_price):
                move = (price / entry_price - 1.0) * math.copysign(1.0, pos)
                forced_exit = (move <= -self.risk.stop_loss
                               or move >= self.risk.take_profit)

            target = float(np.clip(signals.iat[i], -self.risk.max_leverage,
                                   self.risk.max_leverage))
            if forced_exit:
                target = 0.0

            cost = abs(target - pos) * self.commission
            period_ret = gross - cost
            equity *= (1.0 + period_ret)

            opening = pos == 0.0 and target != 0.0
            closing = pos != 0.0 and target == 0.0
            flipping = (pos != 0.0 and target != 0.0
                        and math.copysign(1, target) != math.copysign(1, pos))

            if closing or flipping:
                trades.append({
                    "symbol": symbol,
                    "side": "LONG" if pos > 0 else "SHORT",
                    "entry_date": entry_date, "exit_date": idx[i],
                    "quantity": pos, "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": (price / entry_price - 1.0) * math.copysign(1, pos),
                })
            if opening or flipping:
                entry_price, entry_date = price, idx[i]
            elif closing:
                entry_price, entry_date = float("nan"), None

            pos = target
            equity_hist.append(equity)
            ret_hist.append(period_ret)

        # mark out any position still open at the end
        if pos != 0.0 and not math.isnan(entry_price):
            last = prices.iat[-1]
            trades.append({
                "symbol": symbol, "side": "LONG" if pos > 0 else "SHORT",
                "entry_date": entry_date, "exit_date": idx[-1], "quantity": pos,
                "entry_price": entry_price, "exit_price": last,
                "pnl": (last / entry_price - 1.0) * math.copysign(1, pos),
            })

        equity_s = pd.Series(equity_hist, index=idx, name="equity")
        returns_s = pd.Series(ret_hist, index=idx, name="returns")
        trades_df = pd.DataFrame(trades)
        metrics = self._metrics(returns_s, equity_s, trades_df)
        return BacktestResult(equity_s, returns_s, trades_df, metrics,
                              strategy.name, params=vars(strategy))

    def _metrics(
        self, returns: pd.Series, equity: pd.Series, trades: pd.DataFrame
    ) -> dict[str, float]:
        """Compute headline performance and risk metrics."""
        ann = self.periods_per_year
        n_years = max(len(returns) / ann, 1e-9)
        total_return = equity.iloc[-1] / self.initial_cash - 1.0
        cagr = (equity.iloc[-1] / self.initial_cash) ** (1 / n_years) - 1.0
        vol = returns.std() * math.sqrt(ann)
        sharpe = (returns.mean() * ann) / vol if vol > 0 else 0.0
        drawdown = equity / equity.cummax() - 1.0
        max_dd = float(drawdown.min())

        profit_factor, win_rate, n_trades = float("nan"), float("nan"), 0
        if not trades.empty and "pnl" in trades:
            n_trades = len(trades)
            wins = trades.loc[trades["pnl"] > 0, "pnl"].sum()
            losses = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
            profit_factor = wins / losses if losses > 0 else float("inf")
            win_rate = float((trades["pnl"] > 0).mean())

        return {
            "total_return": round(total_return, 4),
            "cagr": round(cagr, 4),
            "ann_vol": round(vol, 4),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4),
            "profit_factor": round(profit_factor, 3),
            "win_rate": round(win_rate, 3),
            "n_trades": n_trades,
        }
