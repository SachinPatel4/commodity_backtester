"""Event-driven backtester: vol targeting, dual PnL modes, reconciled ledger."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.backtest.risk import RiskParams
from src.backtest.strategy import Strategy

PnLMode = Literal["returns", "spread"]


@dataclass
class BacktestResult:
    """Outcome of a single backtest run."""

    equity: pd.Series
    returns: pd.Series          # per-period return on prior equity
    trades: pd.DataFrame        # 'pnl' is $ and sums to (equity_end - initial)
    metrics: dict[str, float]
    strategy: str
    params: dict[str, Any] = field(default_factory=dict)


class Backtester:
    """Simulate a single instrument or spread bar by bar.

    Signals are lagged one bar (decided on close of *t*, earn *t+1*) so there
    is no look-ahead. Two PnL conventions:

    * ``returns`` — multiplicative, for outright prices (WTI). Position is a
      weight in [-max_leverage, max_leverage]; equity compounds.
    * ``spread``  — additive $ PnL, for spreads (crack). Position is in units;
      equity grows by ``units * dPrice * contract_size``. This is the correct
      basis for a value that can approach or cross zero.

    Volatility targeting scales exposure inversely to recent realised vol, so
    risk contracts automatically in stressed regimes (e.g. April 2020).
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        commission: float = 0.0003,
        risk: RiskParams | None = None,
        periods_per_year: int = 252,
        pnl_mode: PnLMode = "returns",
        vol_target: float | None = 0.15,
        vol_window: int = 20,
        contract_size: float = 1.0,
    ) -> None:
        self.initial_cash = initial_cash
        self.commission = commission
        self.risk = risk or RiskParams()
        self.periods_per_year = periods_per_year
        self.pnl_mode = pnl_mode
        self.vol_target = vol_target
        self.vol_window = vol_window
        self.contract_size = contract_size

    def _sized_signal(self, raw: pd.Series, level: pd.Series) -> pd.Series:
        """Vol-target a raw [-1, 1] signal into a weight (returns) or units (spread)."""
        cap = self.risk.max_leverage
        if self.vol_target is None:
            return raw.clip(-cap, cap)

        if self.pnl_mode == "returns":
            realised = (level.pct_change().rolling(self.vol_window).std()
                        * math.sqrt(self.periods_per_year)).replace(0.0, np.nan)
            sized = (raw * (self.vol_target / realised)).clip(-cap, cap)
        else:  # spread: size units so daily $ risk ~ target fraction of capital
            sigma_d = ((level.diff() * self.contract_size)
                       .rolling(self.vol_window).std()).replace(0.0, np.nan)
            target_daily = self.vol_target / math.sqrt(self.periods_per_year) * self.initial_cash
            sized = raw * target_daily / sigma_d
        return sized.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def run(self, data: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        """Execute ``strategy`` over ``data`` (requires a 'close' column)."""
        if "close" not in data.columns:
            raise ValueError("data must contain a 'close' column")

        level = data["close"].astype(float)
        raw = strategy.generate_signals(data).reindex(level.index).fillna(0.0)
        target_pos = self._sized_signal(raw, level)
        symbol = str(data.attrs.get("symbol", "ASSET"))
        idx = level.index

        equity, pos = self.initial_cash, 0.0
        entry_price, entry_date, trade_pnl = float("nan"), None, 0.0
        equity_hist: list[float] = [equity]
        ret_hist: list[float] = [0.0]
        trades: list[dict[str, Any]] = []

        for i in range(1, len(idx)):
            price, prev = level.iat[i], level.iat[i - 1]

            # PnL from the position carried into today
            if self.pnl_mode == "returns":
                gross = equity * pos * (price / prev - 1.0)
            else:
                gross = pos * (price - prev) * self.contract_size

            # dollar-based stop / target on the open trade's running PnL
            running = trade_pnl + gross
            forced_exit = False
            if pos != 0.0:
                sl, tp = self.risk.stop_loss, self.risk.take_profit
                if sl is not None and running <= -sl * self.initial_cash:
                    forced_exit = True
                if tp is not None and running >= tp * self.initial_cash:
                    forced_exit = True

            target = 0.0 if forced_exit else float(target_pos.iat[i])

            # transaction cost on the change in exposure
            if self.pnl_mode == "returns":
                cost = abs(target - pos) * self.commission * equity
            else:
                cost = abs(target - pos) * self.commission * price * self.contract_size

            equity_prev = equity
            period_pnl = gross - cost
            equity += period_pnl
            trade_pnl += period_pnl
            ret_hist.append(period_pnl / equity_prev if equity_prev > 0 else 0.0)
            equity_hist.append(equity)

            opening = pos == 0.0 and target != 0.0
            closing = pos != 0.0 and target == 0.0
            flipping = (pos != 0.0 and target != 0.0
                        and math.copysign(1, target) != math.copysign(1, pos))

            if closing or flipping:
                trades.append({
                    "symbol": symbol, "side": "LONG" if pos > 0 else "SHORT",
                    "entry_date": entry_date, "exit_date": idx[i],
                    "quantity": pos, "entry_price": entry_price,
                    "exit_price": price, "pnl": trade_pnl,
                })
                trade_pnl = 0.0
            if opening or flipping:
                entry_price, entry_date = price, idx[i]
            elif closing:
                entry_price, entry_date = float("nan"), None

            pos = target

        # mark out any position still open at the end
        if pos != 0.0 and not math.isnan(entry_price):
            trades.append({
                "symbol": symbol, "side": "LONG" if pos > 0 else "SHORT",
                "entry_date": entry_date, "exit_date": idx[-1], "quantity": pos,
                "entry_price": entry_price, "exit_price": level.iat[-1],
                "pnl": trade_pnl,
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
        """Headline performance and risk metrics."""
        ann = self.periods_per_year
        n_years = max(len(returns) / ann, 1e-9)
        total_return = equity.iloc[-1] / self.initial_cash - 1.0
        cagr = (max(equity.iloc[-1], 1e-9) / self.initial_cash) ** (1 / n_years) - 1.0
        vol = returns.std() * math.sqrt(ann)
        sharpe = (returns.mean() * ann) / vol if vol > 0 else 0.0
        max_dd = float((equity / equity.cummax() - 1.0).min())

        profit_factor, win_rate, n_trades = float("nan"), float("nan"), 0
        if not trades.empty and "pnl" in trades:
            n_trades = len(trades)
            wins = trades.loc[trades["pnl"] > 0, "pnl"].sum()
            losses = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
            profit_factor = wins / losses if losses > 0 else float("inf")
            win_rate = float((trades["pnl"] > 0).mean())

        return {
            "total_return": round(total_return, 4), "cagr": round(cagr, 4),
            "ann_vol": round(vol, 4), "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4), "profit_factor": round(profit_factor, 3),
            "win_rate": round(win_rate, 3), "n_trades": n_trades,
        }
