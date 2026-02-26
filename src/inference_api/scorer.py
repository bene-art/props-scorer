"""
Scoring logic for inference.

Loads a trained XGBoost model and produces calibrated probabilities.
Falls back to a rule-based heuristic if the model file is unavailable.
"""

import logging
from pathlib import Path

from . import __model_version__
from .schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

# Sport and stat_type encoding (must match training script)
_SPORT_ENC = {"NBA": 0, "NFL": 1, "NHL": 2, "MLB": 3}
_STAT_ENC = {
    "points": 0, "rebounds": 1, "assists": 2, "threes": 3,
    "pts_rebs": 4, "pts_asts": 5, "pts_rebs_asts": 6,
    "passing_yards": 7, "rushing_yards": 8, "receiving_yards": 9,
    "touchdowns": 10, "hits": 11, "home_runs": 12, "rbis": 13,
    "strikeouts": 14, "total_bases": 15, "goals": 16, "shots": 17,
    "saves": 18,
}

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "xgb_scorer_v3.joblib"

# Load model once at import time
_model = None
try:
    import joblib
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        logger.info("model_loaded", extra={"path": str(MODEL_PATH)})
except Exception as e:
    logger.warning("model_load_failed", extra={"error": str(e)})


def _build_features(request: PredictRequest) -> list[float]:
    """Build feature vector from request (matches training feature order)."""
    avg = (request.last_5_avg + request.last_10_avg) / 2 if request.last_10_avg else request.last_5_avg
    line_vs_avg = request.line - avg if avg else 0.0
    elo_diff = request.team_elo - request.opponent_elo

    return [
        float(_SPORT_ENC.get(request.sport, 0)),
        float(_STAT_ENC.get(request.stat_type, 0)),
        request.line,
        request.glicko_mu,
        request.glicko_phi,
        request.last_5_avg,
        request.last_10_avg,
        float(request.is_home),
        request.team_elo,
        request.opponent_elo,
        line_vs_avg,
        elo_diff,
    ]


def _score_model(request: PredictRequest) -> float:
    """Score using the trained XGBoost model."""
    features = _build_features(request)
    proba = _model.predict_proba([features])[0][1]
    return float(proba)


def _score_heuristic(request: PredictRequest) -> float:
    """Fallback rule-based scoring when model is unavailable."""
    avg = (request.last_5_avg + request.last_10_avg) / 2 if request.last_10_avg else request.last_5_avg
    elo_diff = request.team_elo - request.opponent_elo

    base_prob = 0.5
    base_prob += ((request.glicko_mu - 1500) / 1000) * 0.1
    if avg > 0:
        base_prob += ((avg - request.line) / request.line) * 0.15
    if request.is_home:
        base_prob += 0.02
    base_prob += (elo_diff / 500) * 0.05
    base_prob -= ((request.glicko_phi - 150) / 500) * 0.05

    return max(0.05, min(0.95, base_prob))


def score(request: PredictRequest) -> PredictResponse:
    """
    Score a prediction request.

    Uses trained XGBoost model if available, otherwise falls back
    to rule-based heuristic.

    Args:
        request: Validated prediction request

    Returns:
        PredictResponse with probability, confidence, and recommendation
    """
    if _model is not None:
        probability = _score_model(request)
    else:
        probability = _score_heuristic(request)

    # Clamp probability
    probability = max(0.05, min(0.95, probability))

    # Confidence based on Glicko uncertainty
    if request.glicko_phi < 120:
        confidence = "high"
    elif request.glicko_phi < 180:
        confidence = "medium"
    else:
        confidence = "low"

    # Recommendation
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
    )
