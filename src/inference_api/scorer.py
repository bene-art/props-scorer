"""
Scoring logic for inference.

This module contains the prediction logic. The scoring function
uses explicit feature engineering to produce calibrated probabilities.
Model weights can be swapped without changing the interface contract.
"""

from . import __model_version__
from .schemas import PredictRequest, PredictResponse


def score(request: PredictRequest) -> PredictResponse:
    """
    Score a prediction request.

    Applies feature normalization and weighting to produce
    a calibrated probability estimate.

    Args:
        request: Validated prediction request

    Returns:
        PredictResponse with probability, confidence, and recommendation
    """
    # Calculate line vs average delta
    avg = (request.last_5_avg + request.last_10_avg) / 2 if request.last_10_avg else request.last_5_avg
    _line_vs_avg = request.line - avg if avg else 0  # Reserved for model features

    # Elo difference
    elo_diff = request.team_elo - request.opponent_elo

    # Simplified scoring logic (demonstrates feature engineering)
    # In production: model.predict_proba(features)[0][1]
    base_prob = 0.5

    # Glicko adjustment: Higher rating = better performance
    glicko_factor = (request.glicko_mu - 1500) / 1000  # Normalized
    base_prob += glicko_factor * 0.1

    # Recent form adjustment
    if avg > 0:
        form_factor = (avg - request.line) / request.line
        base_prob += form_factor * 0.15

    # Home advantage
    if request.is_home:
        base_prob += 0.02

    # Elo context
    elo_factor = elo_diff / 500  # Normalized
    base_prob += elo_factor * 0.05

    # Uncertainty adjustment (higher phi = less certain)
    uncertainty_penalty = (request.glicko_phi - 150) / 500
    base_prob -= uncertainty_penalty * 0.05

    # Clamp probability
    probability = max(0.05, min(0.95, base_prob))

    # Determine confidence
    if request.glicko_phi < 120:
        confidence = "high"
    elif request.glicko_phi < 180:
        confidence = "medium"
    else:
        confidence = "low"

    # Generate recommendation
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
