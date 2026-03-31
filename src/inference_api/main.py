"""
FastAPI Inference Service

Production ML inference API with:
- /health - Service health check
- /model - Model metadata
- /predict - Inference endpoint
"""

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __model_version__, __version__
from .logging_config import get_logger, setup_logging
from .schemas import HealthResponse, ModelResponse, PredictRequest, PredictResponse
from .scorer import score

# Setup structured logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(
        "service_startup",
        extra={
            "version": __version__,
            "model_version": __model_version__,
        },
    )
    yield
    logger.info("service_shutdown")


app = FastAPI(
    title="Props Scorer",
    description="Inference service with structured observability",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Add request_id and track latency for all requests."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    # Attach request_id to request state
    request.state.request_id = request_id

    response = await call_next(request)

    # Calculate latency
    latency_ms = (time.perf_counter() - start_time) * 1000

    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-MS"] = f"{latency_ms:.2f}"

    # Log request
    logger.info(
        "request_complete",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
        },
    )

    return response


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Service health check endpoint.

    Returns service status, version, and uptime information.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        model_version=__model_version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/model", response_model=ModelResponse)
async def model_info():
    """
    Model metadata endpoint.

    Returns information about the deployed model, distinguishing
    between request inputs and internally engineered features.
    """
    return ModelResponse(
        model_version=__model_version__,
        model_type="XGBoost Classifier",
        api_version=__version__,
        supported_sports=["NBA", "NFL", "NHL", "MLB"],
        inputs=[
            "sport",
            "stat_type",
            "line",
            "glicko_mu",
            "glicko_phi",
            "last_5_avg",
            "last_10_avg",
            "is_home",
            "team_elo",
            "opponent_elo",
        ],
        engineered_features=[
            "line_vs_avg",
            "elo_diff",
        ],
        notes="First 3 inputs are required; others have sensible defaults.",
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, payload: PredictRequest):
    """
    Inference endpoint.

    Accepts feature inputs and returns probability scores.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.info(
        "prediction_request",
        extra={
            "request_id": request_id,
            "sport": payload.sport,
            "stat_type": payload.stat_type,
        },
    )

    # Run inference
    result = score(payload)

    logger.info(
        "prediction_complete",
        extra={
            "request_id": request_id,
            "probability": result.probability,
            "confidence": result.confidence,
        },
    )

    return result


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with structured logging."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        "unhandled_exception",
        extra={
            "request_id": request_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
        },
    )
