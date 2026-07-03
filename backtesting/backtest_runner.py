"""
NexaTrade — Backtest Runner.

The BacktestRunner is the high-level orchestrator for
running backtests. It:
  - Fetches historical data from DataService
  - Instantiates the strategy and Backtester
  - Persists run metadata to PostgreSQL
  - Supports parameter sweeps (grid search)
  - Supports walk-forward analysis
  - Returns and stores BacktestResult objects

Usage:
    runner = BacktestRunner(data_service, pg_client)

    # Single run
    result = await runner.run(
        strategy_class=EMACrossoverStrategy,
        symbol="RELIANCE",
        exchange="NSE",
        interval="5minute",
        from_date="2023-01-01",
        to_date="2024-01-01",
        initial_capital=500_000.0,
    )
    print(result.summary())

    # Parameter sweep
    results = await runner.parameter_sweep(
        strategy_class=EMACrossoverStrategy,
        symbol="RELIANCE",
        exchange="NSE",
        interval="5minute",
        from_date="2023-01-01",
        to_date="2024-01-01",
        param_grid={
            "fast_period": [5, 9, 12],
            "slow_period": [20, 21, 26],
        },
    )
    best = max(results, key=lambda r: r.metrics["sharpe_ratio"])
"""

from __future__ import annotations

import asyncio
import itertools
from datetime import datetime
from typing import Any, Optional, Type

import shortuuid

from backtesting.backtester import Backtester, BacktestResult
from backtesting.performance import PerformanceAnalyser
from data.storage.postgres_client import PostgresClient
from services.data_service import DataService
from strategies.abstract_strategy import AbstractStrategy
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger(__name__)


class BacktestRunner:
    """
    NexaTrade Backtest Orchestrator.

    Manages data fetching, run registration, execution,
    result persistence, and multi-run workflows.
    """

    def __init__(
        self,
        data_service: DataService,
        pg_client: PostgresClient,
    ) -> None:
        self._data_svc = data_service
        self._pg       = pg_client
        self._results:  list[BacktestResult] = []

    # ─────────────────────────────────────────
    # Single Run
    # ─────────────────────────────────────────

    async def run(
        self,
        strategy_class: Type[AbstractStrategy],
        symbol: str,
        exchange: str,
        interval: str,
        from_date: str,
        to_date: str,
        initial_capital: float = 1_000_000.0,
        slippage_pct: float = 0.05,
        commission_pct: float = 0.03,
        warmup_bars: int = 50,
        parameters: Optional[dict[str, Any]] = None,
        broker_name: Optional[str] = None,
        progress_callback=None,
    ) -> BacktestResult:
        """
        Runs a single backtest for a strategy and symbol.

        Steps:
            1. Register run in PostgreSQL (PENDING)
            2. Fetch OHLCV data from DataService
            3. Instantiate strategy and Backtester
            4. Run the backtest
            5. Update run in PostgreSQL (COMPLETE/FAILED)
            6. Return BacktestResult

        Args:
            strategy_class: AbstractStrategy subclass (not instance).
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval.
            from_date: Backtest start date (YYYY-MM-DD).
            to_date: Backtest end date (YYYY-MM-DD).
            initial_capital: Starting capital in INR.
            slippage_pct: Slippage percentage per fill.
            commission_pct: Commission percentage per trade.
            warmup_bars: Warm-up period before signals start.
            parameters: Strategy parameter overrides.
            broker_name: Data source broker name.
            progress_callback: Async callable(pct, bar, total).

        Returns:
            BacktestResult with fills, equity curve, and metrics.

        Raises:
            ValueError: If data fetch fails or is empty.
            RuntimeError: If strategy setup fails.
        """
        run_id = shortuuid.uuid()[:10]
        strategy_name = strategy_class.STRATEGY_NAME

        logger.info(
            f"BacktestRunner: starting run | "
            f"run_id={run_id} | "
            f"strategy={strategy_name} | "
            f"symbol={symbol} | "
            f"interval={interval} | "
            f"range={from_date}→{to_date}"
        )

        # 1 ── Register in PostgreSQL ──────────
        try:
            await self._pg.insert_backtest_run(
                run_id=run_id,
                strategy_name=strategy_name,
                symbol=symbol,
                interval=interval,
                start_date=from_date,
                end_date=to_date,
                initial_capital=initial_capital,
                parameters=parameters or strategy_class.DEFAULT_PARAMETERS,
                broker_name=broker_name,
            )
            await self._pg.update_backtest_run(
                run_id=run_id, status="RUNNING"
            )
        except Exception as exc:
            logger.warning(f"Backtest DB register failed: {exc}")

        # 2 ── Fetch OHLCV data ────────────────
        try:
            df = await self._data_svc.get_ohlcv(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                from_date=from_date,
                to_date=to_date,
                broker_name=broker_name,
            )
            if df.empty:
                raise ValueError(
                    f"No data for {symbol} {interval} "
                    f"{from_date}→{to_date}"
                )
            logger.info(
                f"Data fetched | "
                f"symbol={symbol} | "
                f"bars={len(df)} | "
                f"range={df.index[0]}→{df.index[-1]}"
            )
        except Exception as exc:
            await self._pg.update_backtest_run(
                run_id=run_id, status="FAILED"
            )
            raise

        # 3 ── Instantiate strategy ────────────
        strategy = strategy_class()
        strategy.name = strategy_name
        if parameters:
            strategy.parameters.update(parameters)

        # 4 ── Build and run backtester ────────
        backtester = Backtester(
            strategy=strategy,
            df=df,
            initial_capital=initial_capital,
            slippage_pct=slippage_pct,
            commission_pct=commission_pct,
            warmup_bars=warmup_bars,
            symbol=symbol,
            exchange=exchange,
            interval=interval,
        )
        backtester.run_id = run_id

        try:
            result = await backtester.run(
                progress_callback=progress_callback
            )
        except Exception as exc:
            logger.error(
                f"Backtest execution failed | "
                f"run_id={run_id} | error={exc}"
            )
            await self._pg.update_backtest_run(
                run_id=run_id, status="FAILED"
            )
            raise

        # 5 ── Update PostgreSQL with results ──
        metrics = result.metrics
        try:
            await self._pg.update_backtest_run(
                run_id=run_id,
                status="COMPLETE",
                final_capital=metrics.get("final_capital"),
                total_pnl=metrics.get("total_pnl"),
                total_trades=metrics.get("total_trades"),
                win_rate=metrics.get("win_rate_pct", 0) / 100,
                max_drawdown=metrics.get("max_drawdown_pct", 0) / 100,
                sharpe_ratio=metrics.get("sharpe_ratio"),
            )
        except Exception as exc:
            logger.warning(f"Backtest DB update failed: {exc}")

        self._results.append(result)
        logger.info(
            f"BacktestRunner: run complete | "
            f"run_id={run_id} | "
            f"return={metrics.get('total_return_pct', 0):.2f}% | "
            f"sharpe={metrics.get('sharpe_ratio', 0):.4f} | "
            f"max_dd={metrics.get('max_drawdown_pct', 0):.2f}%"
        )
        return result

    # ─────────────────────────────────────────
    # Parameter Sweep (Grid Search)
    # ─────────────────────────────────────────

    async def parameter_sweep(
        self,
        strategy_class: Type[AbstractStrategy],
        symbol: str,
        exchange: str,
        interval: str,
        from_date: str,
        to_date: str,
        param_grid: dict[str, list[Any]],
        initial_capital: float = 1_000_000.0,
        slippage_pct: float = 0.05,
        commission_pct: float = 0.03,
        warmup_bars: int = 50,
        broker_name: Optional[str] = None,
        max_concurrent: int = 3,
        rank_by: str = "sharpe_ratio",
    ) -> list[BacktestResult]:
        """
        Runs a full grid search over the parameter space.

        Generates all combinations of the param_grid values
        and runs a separate backtest for each combination.
        Results are sorted by the rank_by metric.

        Args:
            strategy_class: Strategy class to test.
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval.
            from_date: Test start date.
            to_date: Test end date.
            param_grid: Dict of {param_name: [value1, value2, ...]}.
            initial_capital: Starting capital.
            slippage_pct: Slippage per fill.
            commission_pct: Commission per trade.
            warmup_bars: Warm-up period.
            broker_name: Data source broker.
            max_concurrent: Max concurrent backtest runs.
            rank_by: Metric to rank results by.

        Returns:
            List of BacktestResult sorted by rank_by metric (desc).

        Example:
            results = await runner.parameter_sweep(
                strategy_class=EMACrossoverStrategy,
                symbol="RELIANCE",
                exchange="NSE",
                interval="5minute",
                from_date="2023-01-01",
                to_date="2024-01-01",
                param_grid={
                    "fast_period": [5, 9, 12],
                    "slow_period": [20, 21, 26],
                    "atr_multiplier": [1.5, 2.0, 2.5],
                },
                rank_by="sharpe_ratio",
            )
            best = results[0]
            print(f"Best params: {best.parameters}")
        """
        # Generate all parameter combinations
        keys   = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        total = len(combinations)

        logger.info(
            f"Parameter sweep started | "
            f"strategy={strategy_class.STRATEGY_NAME} | "
            f"combinations={total} | "
            f"rank_by={rank_by}"
        )

        # Pre-fetch data once (shared across all runs)
        df = await self._data_svc.get_ohlcv(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            from_date=from_date,
            to_date=to_date,
            broker_name=broker_name,
        )
        if df.empty:
            raise ValueError(
                f"No data for sweep: {symbol} {interval}"
            )

        results: list[BacktestResult] = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run_single(
            combo: tuple, combo_idx: int
        ) -> Optional[BacktestResult]:
            params = dict(zip(keys, combo))
            async with semaphore:
                try:
                    logger.debug(
                        f"Sweep combo {combo_idx + 1}/{total} | "
                        f"params={params}"
                    )
                    strategy = strategy_class()
                    strategy.parameters.update(params)

                    backtester = Backtester(
                        strategy=strategy,
                        df=df.copy(),
                        initial_capital=initial_capital,
                        slippage_pct=slippage_pct,
                        commission_pct=commission_pct,
                        warmup_bars=warmup_bars,
                        symbol=symbol,
                        exchange=exchange,
                        interval=interval,
                    )
                    result = await backtester.run()
                    return result
                except Exception as exc:
                    logger.warning(
                        f"Sweep combo {combo_idx + 1} failed: {exc}"
                    )
                    return None

        # Run all combinations with concurrency control
        tasks = [
            _run_single(combo, idx)
            for idx, combo in enumerate(combinations)
        ]
        raw_results = await asyncio.gather(*tasks)
        results = [r for r in raw_results if r is not None]

        # Sort by rank_by metric (descending)
        results.sort(
            key=lambda r: r.metrics.get(rank_by, float("-inf")),
            reverse=True,
        )

        logger.info(
            f"Parameter sweep complete | "
            f"total={total} | "
            f"successful={len(results)} | "
            f"best_{rank_by}="
            f"{results[0].metrics.get(rank_by, 0):.4f}"
            if results else "no results"
        )
        return results

    # ─────────────────────────────────────────
    # Walk-Forward Analysis
    # ─────────────────────────────────────────

    async def walk_forward(
        self,
        strategy_class: Type[AbstractStrategy],
        symbol: str,
        exchange: str,
        interval: str,
        from_date: str,
        to_date: str,
        train_months: int = 6,
        test_months: int = 1,
        param_grid: Optional[dict[str, list[Any]]] = None,
        initial_capital: float = 1_000_000.0,
        broker_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Runs walk-forward analysis (anchored or rolling windows).

        Splits the date range into alternating train/test periods.
        For each window:
            - Optimises parameters on the train period
            - Validates on the out-of-sample test period

        Args:
            strategy_class: Strategy class.
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval.
            from_date: Analysis start date.
            to_date: Analysis end date.
            train_months: Months in each training window.
            test_months: Months in each test window.
            param_grid: Parameter grid for optimisation.
                        If None, uses DEFAULT_PARAMETERS only.
            initial_capital: Starting capital per window.
            broker_name: Data source broker.

        Returns:
            Dict with:
                windows: List of {train_result, test_result} dicts
                combined_metrics: Aggregated out-of-sample metrics
                best_params_per_window: List of winning params
        """
        from datetime import datetime
        from dateutil.relativedelta import relativedelta

        start_dt = datetime.strptime(from_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(to_date,   "%Y-%m-%d")

        windows: list[dict[str, Any]] = []
        oos_results: list[BacktestResult] = []
        best_params_list: list[dict] = []

        current = start_dt

        window_idx = 0
        while current < end_dt:
            train_start = current
            train_end   = current + relativedelta(
                months=train_months
            )
            test_start  = train_end
            test_end    = test_start + relativedelta(
                months=test_months
            )

            if test_end > end_dt:
                break

            window_idx += 1
            logger.info(
                f"Walk-forward window {window_idx} | "
                f"train={train_start.date()}→{train_end.date()} | "
                f"test={test_start.date()}→{test_end.date()}"
            )

            # ── Optimise on training data ──────
            best_params = dict(
                strategy_class.DEFAULT_PARAMETERS
            )
            if param_grid:
                try:
                    train_results = await self.parameter_sweep(
                        strategy_class=strategy_class,
                        symbol=symbol,
                        exchange=exchange,
                        interval=interval,
                        from_date=train_start.strftime("%Y-%m-%d"),
                        to_date=train_end.strftime("%Y-%m-%d"),
                        param_grid=param_grid,
                        initial_capital=initial_capital,
                        broker_name=broker_name,
                        rank_by="sharpe_ratio",
                    )
                    if train_results:
                        best_params = train_results[0].parameters
                except Exception as exc:
                    logger.warning(
                        f"WF window {window_idx} train "
                        f"sweep failed: {exc}"
                    )

            # ── Test on out-of-sample data ─────
            try:
                test_result = await self.run(
                    strategy_class=strategy_class,
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    from_date=test_start.strftime("%Y-%m-%d"),
                    to_date=test_end.strftime("%Y-%m-%d"),
                    initial_capital=initial_capital,
                    parameters=best_params,
                    broker_name=broker_name,
                )
                oos_results.append(test_result)
                best_params_list.append(best_params)
                windows.append({
                    "window":      window_idx,
                    "train_start": str(train_start.date()),
                    "train_end":   str(train_end.date()),
                    "test_start":  str(test_start.date()),
                    "test_end":    str(test_end.date()),
                    "best_params": best_params,
                    "test_metrics": test_result.metrics,
                })
            except Exception as exc:
                logger.warning(
                    f"WF window {window_idx} test failed: {exc}"
                )

            current = test_start  # Rolling window

        # ── Aggregate OOS metrics ─────────────
        combined_metrics: dict[str, Any] = {}
        if oos_results:
            combined_metrics = {
                "windows_run":           window_idx,
                "successful_windows":    len(oos_results),
                "avg_return_pct":        round(
                    np.mean([
                        r.metrics.get("total_return_pct", 0)
                        for r in oos_results
                    ]), 4
                ),
                "avg_sharpe":            round(
                    np.mean([
                        r.metrics.get("sharpe_ratio", 0)
                        for r in oos_results
                    ]), 4
                ),
                "avg_max_drawdown_pct":  round(
                    np.mean([
                        r.metrics.get("max_drawdown_pct", 0)
                        for r in oos_results
                    ]), 4
                ),
                "avg_win_rate_pct":      round(
                    np.mean([
                        r.metrics.get("win_rate_pct", 0)
                        for r in oos_results
                    ]), 4
                ),
                "total_oos_trades":      sum(
                    r.metrics.get("total_trades", 0)
                    for r in oos_results
                ),
            }

        logger.info(
            f"Walk-forward complete | "
            f"windows={window_idx} | "
            f"avg_sharpe={combined_metrics.get('avg_sharpe', 0):.4f}"
        )

        return {
            "windows":                windows,
            "combined_metrics":       combined_metrics,
            "best_params_per_window": best_params_list,
            "oos_results":            oos_results,
        }

    # ─────────────────────────────────────────
    # Result Comparison
    # ─────────────────────────────────────────

    def compare_results(
        self,
        results: Optional[list[BacktestResult]] = None,
        metrics: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Compares multiple backtest results as a DataFrame.
        Uses stored results if none are provided.

        Args:
            results: List of BacktestResult to compare.
                     Uses self._results if None.
            metrics: List of metric names to include.
                     Includes all metrics if None.

        Returns:
            DataFrame with one row per backtest run.

        Example:
            df = runner.compare_results(sweep_results)
            best = df.sort_values("sharpe_ratio", ascending=False)
            print(best[["fast_period", "slow_period", "sharpe_ratio"]])
        """
        import pandas as pd

        target = results or self._results
        if not target:
            return pd.DataFrame()

        rows = []
        for result in target:
            row = {
                "run_id":        result.run_id,
                "strategy":      result.strategy_name,
                "symbol":        result.symbol,
                "interval":      result.interval,
                "from_date":     result.from_date,
                "to_date":       result.to_date,
            }
            # Add parameters
            for k, v in result.parameters.items():
                row[f"param_{k}"] = v
            # Add metrics
            for k, v in result.metrics.items():
                if metrics is None or k in metrics:
                    row[k] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        if "sharpe_ratio" in df.columns:
            df.sort_values(
                "sharpe_ratio", ascending=False, inplace=True
            )
        df.reset_index(drop=True, inplace=True)
        return df

    def get_best_result(
        self,
        results: Optional[list[BacktestResult]] = None,
        rank_by: str = "sharpe_ratio",
    ) -> Optional[BacktestResult]:
        """
        Returns the best result from a list by the given metric.

        Args:
            results: List to search. Uses self._results if None.
            rank_by: Metric to rank by (higher = better).

        Returns:
            Best BacktestResult or None if list is empty.
        """
        target = results or self._results
        if not target:
            return None
        return max(
            target,
            key=lambda r: r.metrics.get(rank_by, float("-inf")),
        )

    @property
    def stored_results(self) -> list[BacktestResult]:
        """Returns all results stored in this runner session."""
        return list(self._results)

    def clear_results(self) -> None:
        """Clears all stored results from this session."""
        self._results.clear()