"""
Scoring logic.

Loads calibrated XGBoost models (per stat) trained on real NBA game logs.
Falls back to a rule-based heuristic when no model file is present.

Generate model files: python scripts/fetch_data.py && python scripts/train_model.py
"""

import logging
from pathlib import Path

from .schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

_MODEL_REGISTRY: dict[str, str] = {"points": "nba_points_v1.joblib"}
_cache: dict[str, object] = {}


def _load_model(stat_type: str):
    if stat_type in _cache:
        return _cache[stat_type]
    filename = _MODEL_REGISTRY.get(stat_type)
    if filename is None:
        _cache[stat_type] = None
        return None
    path = MODELS_DIR / filename
    if not path.exists():
        logger.info("model_not_found", extra={"stat_type": stat_type})
        _cache[stat_type] = None
        return None
    try:
        import joblib
        model = joblib.load(path)
        _cache[stat_type] = model
        logger.info("model_loaded", extra={"stat_type": stat_type})
        return model
    except Exception as exc:
        logger.warning("model_load_failed", extra={"error": str(exc)})
        _cache[stat_type] = None
        return None


def preload_models() -> None:
    for stat_type in _MODEL_REGISTRY:
        _load_model(stat_type)


def _build_features(req: PredictRequest) -> list[float]:
    line_vs_avg = req.line - req.rolling_10_avg
    avg_trend   = req.rolling_5_avg - req.rolling_10_avg
    return [
        req.rolling_5_avg, req.rolling_10_avg, req.rolling_5_std,
        float(req.rest_days), float(req.is_home), req.opp_def_rtg,
        float(min(req.games_played, 60)), line_vs_avg, avg_trend,
    ]


def _heuristic(req: PredictRequest) -> float:
    avg         = req.rolling_10_avg or req.rolling_5_avg or 1.0
    line_vs_avg = req.line - avg
    prob        = 0.50 - (line_vs_avg / avg) * 0.40
    prob       += 0.02 if req.is_home else 0.0
    prob       -= (req.opp_def_rtg - 110.0) / 100.0 * 0.05
    prob       += (req.rest_days - 2)        / 10.0  * 0.02
    return max(0.05, min(0.95, prob))


def score(req: PredictRequest) -> PredictResponse:
    from . import __model_version__

    stat  = req.stat_type if isinstance(req.stat_type, str) else req.stat_type.value
    model = _load_model(stat)

    if model is not None:
        probability = float(model.predict_proba([_build_features(req)])[0][1])
        source      = "model"
    else:
        probability = _heuristic(req)
        source      = "heuristic"

    probability = max(0.05, min(0.95, probability))
    distance    = abs(probability - 0.5)
    confidence  = "high" if distance >= 0.12 else ("medium" if distance >= 0.07 else "low")

    if probability >= 0.55:
        recommendation = "OVER"
    elif probability <= 0.45:
        recommendation = "UNDER"
    else:
        recommendation = "NO_EDGE"

    return PredictResponse(
        probability=round(probability, 4),
        confidence=confidence,
        recommendation=recommendation,
        model_version=__model_version__,
        source=source,
    )
