"""
FastAPI Inference Service

  GET  /health          Service health check
  GET  /model           Model metadata and feature list
  POST /predict         Single player prop inference
  POST /predict/batch   Batch inference (up to 50 requests)
"""
import time, uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from . import __model_version__, __version__
from .logging_config import get_logger, setup_logging
from .schemas import (BatchPredictRequest, BatchPredictResponse, HealthResponse,
                      ModelResponse, PredictRequest, PredictResponse)
from .scorer import preload_models, score

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_models()
    logger.info("service_startup", extra={"version": __version__, "model_version": __model_version__})
    yield
    logger.info("service_shutdown")


app = FastAPI(
    title="Props Scorer",
    description="NBA player prop inference — calibrated XGBoost on real game log data.",
    version=__version__, lifespan=lifespan,
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    request.state.request_id = request_id
    response   = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-MS"] = f"{latency_ms:.2f}"
    logger.info("request_complete", extra={
        "request_id": request_id, "method": request.method,
        "path": request.url.path, "status_code": response.status_code,
        "latency_ms": round(latency_ms, 2),
    })
    return response


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", version=__version__,
                          model_version=__model_version__,
                          timestamp=datetime.now(timezone.utc).isoformat())


@app.get("/model", response_model=ModelResponse)
async def model_info():
    return ModelResponse(
        model_version=__model_version__, model_type="XGBoost + isotonic calibration",
        api_version=__version__, supported_sports=["NBA"],
        inputs=["stat_type","line","rolling_5_avg","rolling_10_avg","rolling_5_std",
                "rest_days","is_home","opp_def_rtg","games_played"],
        engineered_features=["line_vs_avg","avg_trend"],
        notes="rolling_5_avg and rolling_10_avg must be computed prior to game tip-off.",
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, payload: PredictRequest):
    rid = getattr(request.state, "request_id", "unknown")
    logger.info("prediction_request", extra={"request_id": rid, "stat_type": payload.stat_type})
    result = score(payload)
    logger.info("prediction_complete", extra={"request_id": rid, "probability": result.probability,
                                              "source": result.source})
    return result


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: Request, payload: BatchPredictRequest):
    rid = getattr(request.state, "request_id", "unknown")
    logger.info("batch_request", extra={"request_id": rid, "count": len(payload.predictions)})
    results = [score(p) for p in payload.predictions]
    return BatchPredictResponse(results=results, count=len(results))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "unknown")
    logger.error("unhandled_exception", extra={
        "request_id": rid, "error_type": type(exc).__name__, "error_message": str(exc)})
    return JSONResponse(status_code=500,
                        content={"error": "Internal server error", "request_id": rid})
