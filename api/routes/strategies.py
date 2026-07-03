"""
NexaTrade — Strategy Management Routes.

Endpoints:
    GET    /api/v1/strategies/registered      → all registered strategies
    GET    /api/v1/strategies/active          → all running strategies
    POST   /api/v1/strategies/activate        → start a strategy
    DELETE /api/v1/strategies/{name}          → stop a strategy
    POST   /api/v1/strategies/{name}/restart  → restart a strategy
    PATCH  /api/v1/strategies/{name}/params   → update parameters
    GET    /api/v1/strategies/{name}/stats    → runtime stats
"""

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_current_user, get_strategy_engine
from api.schemas import (
    StrategyActivateRequest,
    StrategyInfoResponse,
    StrategyStatsResponse,
    StrategyUpdateParamsRequest,
    SuccessResponse,
)
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/registered",
    response_model=list[StrategyInfoResponse],
    summary="All registered strategy classes",
)
async def get_registered_strategies(
    engine=Depends(get_strategy_engine),
    user: dict = Depends(get_current_user),
) -> list[StrategyInfoResponse]:
    """Returns metadata for all discovered strategy plugins."""
    return [
        StrategyInfoResponse(**s)
        for s in engine.get_registered_strategies()
    ]


@router.get(
    "/active",
    response_model=list[StrategyStatsResponse],
    summary="All active (running) strategies",
)
async def get_active_strategies(
    engine=Depends(get_strategy_engine),
    user: dict = Depends(get_current_user),
) -> list[StrategyStatsResponse]:
    """Returns runtime stats for all currently active strategies."""
    return [
        StrategyStatsResponse(**s)
        for s in engine.get_active_strategies()
    ]


@router.post(
    "/activate",
    response_model=SuccessResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Activate a strategy",
)
async def activate_strategy(
    body: StrategyActivateRequest,
    engine=Depends(get_strategy_engine),
    user: dict = Depends(get_current_user),
) -> SuccessResponse:
    """
    Activates a registered strategy.
    Calls on_start() and begins routing ticks and candles.
    """
    try:
        success = await engine.activate_strategy(
            strategy_name=body.strategy_name,
            parameters=body.parameters,
            instruments=body.instruments,
            capital=body.capital,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy activation failed: {exc}",
        )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy {body.strategy_name} failed to start.",
        )

    logger.info(
        f"Strategy activated via API | "
        f"name={body.strategy_name} | "
        f"user={user.get('username')}"
    )
    return SuccessResponse(
        message=f"Strategy '{body.strategy_name}' activated."
    )


@router.delete(
    "/{strategy_name}",
    response_model=SuccessResponse,
    summary="Deactivate a running strategy",
)
async def deactivate_strategy(
    strategy_name: str,
    engine=Depends(get_strategy_engine),
    user: dict = Depends(get_current_user),
) -> SuccessResponse:
    """Stops a running strategy and calls its on_stop() handler."""
    success = await engine.deactivate_strategy(
        strategy_name=strategy_name.lower(),
        reason=f"api_request:{user.get('username')}",
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy '{strategy_name}' is not active.",
        )
    logger.info(
        f"Strategy deactivated via API | "
        f"name={strategy_name} | "
        f"user={user.get('username')}"
    )
    return SuccessResponse(
        message=f"Strategy '{strategy_name}' deactivated."
    )


@router.post(
    "/{strategy_name}/restart",
    response_model=SuccessResponse,
    summary="Restart a running strategy",
)
async def restart_strategy(
    strategy_name: str,
    engine=Depends(get_strategy_engine),
    user: dict = Depends(get_current_user),
) -> SuccessResponse:
    """Restarts a strategy, preserving its parameters."""
    success = await engine.restart_strategy(
        strategy_name=strategy_name.lower()
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy '{strategy_name}' restart failed.",
        )
    return SuccessResponse(
        message=f"Strategy '{strategy_name}' restarted."
    )


@router.patch(
    "/{strategy_name}/params",
    response_model=SuccessResponse,
    summary="Update strategy parameters at runtime",
)
async def update_strategy_params(
    strategy_name: str,
    body: StrategyUpdateParamsRequest,
    engine=Depends(get_strategy_engine),
    user: dict = Depends(get_current_user),
) -> SuccessResponse:
    """Updates strategy parameters without restarting it."""
    instance = engine.get_strategy_instance(
        strategy_name.lower()
    )
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy '{strategy_name}' is not active.",
        )
    instance.update_parameters(body.parameters)
    return SuccessResponse(
        message=f"Parameters updated for '{strategy_name}'.",
        data=instance.parameters,
    )


@router.get(
    "/{strategy_name}/stats",
    response_model=StrategyStatsResponse,
    summary="Runtime stats for a specific strategy",
)
async def get_strategy_stats(
    strategy_name: str,
    engine=Depends(get_strategy_engine),
    user: dict = Depends(get_current_user),
) -> StrategyStatsResponse:
    """Returns runtime statistics for a specific active strategy."""
    instance = engine.get_strategy_instance(
        strategy_name.lower()
    )
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy '{strategy_name}' is not active.",
        )
    return StrategyStatsResponse(**instance.get_stats())