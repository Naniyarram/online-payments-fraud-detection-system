"""
Health and readiness probe endpoints.

/health  - Pure liveness. Returns 200 immediately. Used by Render health checks
           and Better Stack uptime monitors. Never checks model, DB, or any
           downstream dependency.

/ready   - Readiness probe. Verifies that the fraud model and feature pipeline
           are loaded and the prediction service can actually handle requests.
           Returns 503 if artifacts are missing.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from src.api import routers

health_router = APIRouter(tags=["Health"])


@health_router.get("/health")
def health_check():
    """
    Liveness probe.
    If this endpoint returns anything other than 200, the process is dead.
    Render and Better Stack hit this on a regular interval.
    """
    return {
        "status": "healthy",
        "service": "Enterprise Fraud Intelligence API",
        "version": "1.0.0",
        "environment": "production",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@health_router.get("/ready")
def readiness_check():
    """
    Readiness probe.
    Checks whether the ML artifacts loaded during startup are actually
    available. If they aren't, the service is alive but can't serve
    predictions, so we return 503.
    """
    model_loaded = routers.MODEL is not None
    pipeline_loaded = routers.PIPELINE is not None

    if not model_loaded or not pipeline_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "model_loaded": model_loaded,
                "pipeline_loaded": pipeline_loaded,
                "service": "Enterprise Fraud Intelligence API",
            },
        )

    return {
        "status": "ready",
        "model_loaded": model_loaded,
        "pipeline_loaded": pipeline_loaded,
        "service": "Enterprise Fraud Intelligence API",
    }
