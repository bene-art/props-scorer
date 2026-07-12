"""
Props Scorer

End-to-end NBA player prop inference pipeline:
- Real game log data via nba_api (2023-25 regular seasons)
- Per-stat calibrated XGBoost (isotonic calibration)
- FastAPI serving with request tracing and structured logging
- Batch endpoint for multi-player scoring

Run scripts/fetch_data.py then scripts/train_model.py before serving.
"""

__version__ = "1.0.0"
__model_version__ = "nba_points_v1"
