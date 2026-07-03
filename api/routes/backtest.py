"""
NexaTrade — Backtesting Routes.

Endpoints:
    POST /api/v1/backtest/run            → run single backtest
    POST /api/v1/backtest/sweep          → parameter sweep
    GET  /api/v1/backtest/{run_id}       → fetch run result
    GET  /api/v1/backtest/runs           → list all runs
    GET  /api/v1/backtest/{run_id}/equity → equity curve
    GET  /api/v1/backtest/{run_id}/fills  → fills list
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import (
    get_backtest_runner,
    get_current_user,
    get_pg,
    get_strategy_engine,
)
from api.schemas import (
    BacktestMetricsResponse,
    BacktestRunRequest,
    ParameterSweepRequest,
    SuccessResponse,
)
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/run",
    response_model=BacktestMetricsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run a single backtest",
)
async def run_backtest(
    body: BacktestRunRequest,
    runner=Depends(get_backtest_runner),
    engine=Depends(get_strategy_engine),
    user: dict = Depends(get_current_user),
) -> BacktestMetricsResponse:
    """
    Runs a complete event-driven backtest for a strategy.
    Fetches data, simulates fills, computes metrics.
    """
    # Resolve strategy class
    if body.strategy_name not in engine._registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Strategy '{body.strategy_name}' not registered. "
                f"Available: {list(engine._registry.keys())}"
            ),
        )
    strategy_class = engine._registry[body.strategy_name]

    try:
        result = await runner.run(
            strategy_class=strategy_class,
            symbol=body.symbol,
            exchange=body.exchange,
            interval=body.interval,
            from_date=body.from_date,
            to_date=body.to_date,
            initial_capital=body.initial_capital,
            slippage_pct=body.slippage_pct,
            commission_pct=body.commission_pct,
            warmup_bars=body.warmup_bars,
            parameters=body.parameters,
            broker_name=body.broker_name,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest failed: {exc}",
        )

    m = result.metrics
    return BacktestMetricsResponse(
        run_id=result.run_id,
        strategy_name=result.strategy_name,
        symbol=result.symbol,
        interval=result.interval,
        from_date=result.from_date,
        to_date=result.to_date,
        initial_capital=result.initial_capital,
        parameters=result.parameters,
        final_capital=m.get("final_capital", 0.0),
        total_pnl=m.get("total_pnl", 0.0),
        total_return_pct=m.get("total_return_pct", 0.0),
        cagr_pct=m.get("cagr_pct", 0.0),
        total_trades=m.get("total_trades", 0),
        winning_trades=m.get("winning_trades", 0),
        losing_trades=m.get("losing_trades", 0),
        win_rate_pct=m.get("win_rate_pct", 0.0),
        avg_win=m.get("avg_win", 0.0),
        avg_loss=m.get("avg_loss", 0.0),
        largest_win=m.get("largest_win", 0.0),
        largest_loss=m.get("largest_loss", 0.0),
        profit_factor=m.get("profit_factor", 0.0),
        expectancy=m.get("expectancy", 0.0),
        max_drawdown_pct=m.get("max_drawdown_pct", 0.0),
        max_drawdown_inr=m.get("max_drawdown_inr", 0.0),
        sharpe_ratio=m.get("sharpe_ratio", 0.0),
        sortino_ratio=m.get("sortino_ratio", 0.0),
        calmar_ratio=m.get("calmar_ratio", 0.0),
        total_commission=m.get("total_commission", 0.0),
        total_slippage=m.get("total_slippage", 0.0),
    )


@router.post(
    "/sweep",
    summary="Parameter sweep (grid search)",
    status_code=status.HTTP_201_CREATED,
)
async def run_parameter_sweep(
    body: ParameterSweepRequest,
    runner=Depends(get_backtest_runner),
    engine=Depends(get_strategy_engine),
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Runs a grid search over all parameter combinations.
    Returns results sorted by the rank_by metric.
    """
    if body.strategy_name not in engine._registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy '{body.strategy_name}' not found.",
        )
    strategy_class = engine._registry[body.strategy_name]

    try:
        results = await runner.parameter_sweep(
            strategy_class=strategy_class,
            symbol=body.symbol,
            exchange=body.exchange,
            interval=body.interval,
            from_date=body.from_date,
            to_date=body.to_date,
            param_grid=body.param_grid,
            initial_capital=body.initial_capital,
            slippage_pct=body.slippage_pct,
            commission_pct=body.commission_pct,
            max_concurrent=body.max_concurrent,
            rank_by=body.rank_by,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Parameter sweep failed: {exc}",
        )

    return {
        "total_combinations": len(results),
        "rank_by":            body.rank_by,
        "results": [
            {
                "run_id":      r.run_id,
                "parameters":  r.parameters,
                "metrics": {
                    "total_return_pct": r.metrics.get("total_return_pct"),
                    "sharpe_ratio":     r.metrics.get("sharpe_ratio"),
                    "max_drawdown_pct": r.metrics.get("max_drawdown_pct"),
                    "win_rate_pct":     r.metrics.get("win_rate_pct"),
                    "total_trades":     r.metrics.get("total_trades"),
                    "profit_factor":    r.metrics.get("profit_factor"),
                },
            }
            for r in results
        ],
    }


@router.get(
    "/{run_id}/equity",
    summary="Equity curve for a backtest run",
)
async def get_equity_curve(
    run_id: str,
    runner=Depends(get_backtest_runner),
    user: dict = Depends(get_current_user),
) -> dict:
    """Returns the equity curve time series for a completed run."""
    result = next(
        (r for r in runner.stored_results if r.run_id == run_id),
        None,
    )
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found in session.",
        )
    return {
        "run_id":       run_id,
        "equity_curve": result.equity_curve,
        "points":       len(result.equity_curve),
    }


@router.get(
    "/{run_id}/fills",
    summary="Fills list for a backtest run",
)
async def get_fills(
    run_id: str,
    runner=Depends(get_backtest_runner),
    user: dict = Depends(get_current_user),
) -> dict:
    """Returns all simulated fills for a completed backtest run."""
    result = next(
        (r for r in runner.stored_results if r.run_id == run_id),
        None,
    )
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found in session.",
        )
    return {
        "run_id": run_id,
        "fills":  [
            {
                "executed_at":      str(f.executed_at),
                "symbol":           f.symbol,
                "transaction_type": str(f.transaction_type),
                "quantity":         f.quantity,
                "price":            f.price,
                "commission":       f.commission,
            }
            for f in result.fills
        ],
        "total_fills": len(result.fills),
    }