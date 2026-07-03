"""
NexaTrade — Risk Management Routes.

Endpoints:
    GET  /api/v1/risk/stats               → risk manager stats
    POST /api/v1/risk/kill-switch/arm      → arm kill switch
    POST /api/v1/risk/kill-switch/disarm   → disarm kill switch
    GET  /api/v1/risk/kill-switch/status   → kill switch status
"""

from fastapi import APIRouter, Depends, status

from api.dependencies import get_current_user, get_redis, get_risk_manager
from api.schemas import KillSwitchRequest, RiskStatsResponse, SuccessResponse
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/stats",
    response_model=RiskStatsResponse,
    summary="Risk manager statistics",
)
async def get_risk_stats(
    risk=Depends(get_risk_manager),
    user: dict = Depends(get_current_user),
) -> RiskStatsResponse:
    """Returns signal approval/rejection statistics."""
    return RiskStatsResponse(**risk.get_stats())


@router.post(
    "/kill-switch/arm",
    response_model=SuccessResponse,
    summary="Arm the kill switch",
)
async def arm_kill_switch(
    body: KillSwitchRequest,
    risk=Depends(get_risk_manager),
    user: dict = Depends(get_current_user),
) -> SuccessResponse:
    """
    Arms the kill switch, blocking all new signals immediately.
    If global_switch=True, blocks ALL brokers.
    """
    if body.global_switch:
        await risk.arm_global_kill_switch(reason=body.reason)
        msg = "Global kill switch armed."
    else:
        broker = body.broker_name or "breeze"
        await risk.arm_kill_switch(
            broker_name=broker, reason=body.reason
        )
        msg = f"Kill switch armed for broker: {broker}"

    logger.warning(
        f"Kill switch armed via API | "
        f"global={body.global_switch} | "
        f"user={user.get('username')} | "
        f"reason={body.reason}"
    )
    return SuccessResponse(message=msg)


@router.post(
    "/kill-switch/disarm",
    response_model=SuccessResponse,
    summary="Disarm the kill switch",
)
async def disarm_kill_switch(
    body: KillSwitchRequest,
    risk=Depends(get_risk_manager),
    user: dict = Depends(get_current_user),
) -> SuccessResponse:
    """Disarms the kill switch, resuming signal processing."""
    if body.global_switch:
        await risk.disarm_global_kill_switch()
        msg = "Global kill switch disarmed."
    else:
        broker = body.broker_name or "breeze"
        await risk.disarm_kill_switch(broker_name=broker)
        msg = f"Kill switch disarmed for broker: {broker}"

    logger.info(
        f"Kill switch disarmed via API | "
        f"user={user.get('username')}"
    )
    return SuccessResponse(message=msg)


@router.get(
    "/kill-switch/status",
    summary="Kill switch status",
)
async def kill_switch_status(
    redis=Depends(get_redis),
    user: dict = Depends(get_current_user),
) -> dict:
    """Returns current kill switch state for all brokers."""
    global_ks = await redis.is_global_kill_switch_active()
    return {
        "global_kill_switch": global_ks,
        "timestamp": str(__import__("datetime").datetime.utcnow()),
    }