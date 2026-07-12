"""Pydantic schemas for request/response validation."""
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class StatTypeEnum(str, Enum):
    points   = "points"    # Model-backed (calibrated XGBoost)
    rebounds = "rebounds"  # Heuristic fallback
    assists  = "assists"   # Heuristic fallback
    threes   = "threes"    # Heuristic fallback


class HealthResponse(BaseModel):
    status: str; version: str; model_version: str; timestamp: str


class ModelResponse(BaseModel):
    model_version: str; model_type: str; api_version: str
    supported_sports: list[str]; inputs: list[str]
    engineered_features: list[str]; notes: str = ""


class PredictRequest(BaseModel):
    """All rolling averages must be computed on games *prior* to the current game."""
    stat_type:      StatTypeEnum = Field(...,              description="Stat type")
    line:           float        = Field(..., gt=0,        description="Betting line value")
    rolling_5_avg:  float        = Field(..., ge=0,        description="5-game rolling average")
    rolling_10_avg: float        = Field(..., ge=0,        description="10-game rolling average")
    rolling_5_std:  float        = Field(default=3.0, ge=0)
    rest_days:      int          = Field(default=2, ge=0, le=10)
    is_home:        bool         = Field(default=True)
    opp_def_rtg:    float        = Field(default=110.0,    description="Opponent defensive rating (league avg ~110)")
    games_played:   int          = Field(default=20, ge=0)

    model_config = ConfigDict(
        json_schema_extra={"example": {
            "stat_type": "points", "line": 27.5,
            "rolling_5_avg": 28.2, "rolling_10_avg": 26.8, "rolling_5_std": 4.1,
            "rest_days": 2, "is_home": True, "opp_def_rtg": 112.3, "games_played": 45,
        }},
        use_enum_values=True,
    )


class PredictResponse(BaseModel):
    probability:    float = Field(..., description="P(over line)")
    confidence:     str   = Field(..., description="low / medium / high")
    recommendation: str   = Field(..., description="OVER / UNDER / NO_EDGE")
    model_version:  str
    source:         str   = Field(..., description="model or heuristic")

    model_config = ConfigDict(json_schema_extra={"example": {
        "probability": 0.61, "confidence": "medium", "recommendation": "OVER",
        "model_version": "nba_points_v1", "source": "model",
    }})


class BatchPredictRequest(BaseModel):
    predictions: list[PredictRequest] = Field(..., min_length=1, max_length=50)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    count:   int
