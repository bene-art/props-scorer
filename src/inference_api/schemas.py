"""
Pydantic schemas for request/response validation.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class SportEnum(str, Enum):
    """Supported sports."""

    NBA = "NBA"
    NFL = "NFL"
    NHL = "NHL"
    MLB = "MLB"


class StatTypeEnum(str, Enum):
    """Supported stat types."""

    # Basketball
    points = "points"
    rebounds = "rebounds"
    assists = "assists"
    threes = "threes"
    pts_rebs = "pts_rebs"
    pts_asts = "pts_asts"
    pts_rebs_asts = "pts_rebs_asts"
    # Football
    passing_yards = "passing_yards"
    rushing_yards = "rushing_yards"
    receiving_yards = "receiving_yards"
    touchdowns = "touchdowns"
    # Baseball
    hits = "hits"
    home_runs = "home_runs"
    rbis = "rbis"
    strikeouts = "strikeouts"
    total_bases = "total_bases"
    # Hockey
    goals = "goals"
    shots = "shots"
    saves = "saves"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_version: str = Field(..., description="Deployed model version")
    timestamp: str = Field(..., description="ISO timestamp")


class ModelResponse(BaseModel):
    """Model metadata response."""

    model_version: str = Field(..., description="Model version identifier")
    model_type: str = Field(..., description="Model architecture type")
    api_version: str = Field(..., description="API version")
    supported_sports: list[str] = Field(..., description="Supported sports")
    inputs: list[str] = Field(..., description="Required/optional fields in predict request")
    engineered_features: list[str] = Field(
        ..., description="Features computed internally from inputs"
    )
    notes: str = Field(default="", description="Additional information about the model")


class PredictRequest(BaseModel):
    """Prediction request payload."""

    sport: SportEnum = Field(..., description="Sport code (NBA, NFL, NHL, MLB)")
    stat_type: StatTypeEnum = Field(..., description="Statistic type")
    line: float = Field(..., description="Betting line value")

    # Player features
    glicko_mu: float = Field(default=1500.0, description="Player Glicko rating")
    glicko_phi: float = Field(default=200.0, description="Player Glicko uncertainty")
    last_5_avg: float = Field(default=0.0, description="Last 5 games average")
    last_10_avg: float = Field(default=0.0, description="Last 10 games average")

    # Game context
    is_home: bool = Field(default=True, description="Home game indicator")
    team_elo: float = Field(default=1500.0, description="Team Elo rating")
    opponent_elo: float = Field(default=1500.0, description="Opponent Elo rating")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sport": "NBA",
                "stat_type": "points",
                "line": 25.5,
                "glicko_mu": 1650.0,
                "glicko_phi": 150.0,
                "last_5_avg": 27.2,
                "last_10_avg": 26.8,
                "is_home": True,
                "team_elo": 1580.0,
                "opponent_elo": 1520.0,
            }
        },
        use_enum_values=True,
    )


class PredictResponse(BaseModel):
    """Prediction response."""

    probability: float = Field(..., description="P(over) probability")
    confidence: str = Field(..., description="Confidence level (low/medium/high)")
    recommendation: str = Field(..., description="Betting recommendation")
    model_version: str = Field(..., description="Model version used")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "probability": 0.62,
                "confidence": "medium",
                "recommendation": "OVER",
                "model_version": "xgb-v3",
            }
        }
    )
