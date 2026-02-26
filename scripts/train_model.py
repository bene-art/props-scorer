#!/usr/bin/env python3
"""
Train the props scorer model on synthetic data.

Generates reproducible synthetic training data and trains an XGBoost
classifier. The synthetic data uses realistic feature distributions
but contains NO real player, game, or odds data.

Usage:
    python scripts/train_model.py

Output:
    models/xgb_scorer_v3.joblib
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Fixed seed for reproducibility
SEED = 42
N_SAMPLES = 5000
MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_scorer_v3.joblib"

# Sport and stat_type encoded as integers (label encoding)
SPORTS = {"NBA": 0, "NFL": 1, "NHL": 2, "MLB": 3}
STAT_TYPES = {
    "points": 0, "rebounds": 1, "assists": 2, "threes": 3,
    "pts_rebs": 4, "pts_asts": 5, "pts_rebs_asts": 6,
    "passing_yards": 7, "rushing_yards": 8, "receiving_yards": 9,
    "touchdowns": 10, "hits": 11, "home_runs": 12, "rbis": 13,
    "strikeouts": 14, "total_bases": 15, "goals": 16, "shots": 17,
    "saves": 18,
}

# Feature order (must match scorer.py feature construction)
FEATURE_NAMES = [
    "sport_enc", "stat_type_enc", "line", "glicko_mu", "glicko_phi",
    "last_5_avg", "last_10_avg", "is_home", "team_elo", "opponent_elo",
    "line_vs_avg", "elo_diff",
]


def generate_synthetic_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data with realistic distributions.

    Returns:
        X: Feature matrix (n, 12)
        y: Binary target (1 = over, 0 = under)
    """
    rng = np.random.RandomState(seed)

    # Categorical features
    sport_enc = rng.randint(0, 4, size=n).astype(float)
    stat_type_enc = rng.randint(0, 19, size=n).astype(float)

    # Player rating features (Glicko-like distributions)
    glicko_mu = rng.normal(1500, 150, size=n).clip(1000, 2200)
    glicko_phi = rng.gamma(3, 50, size=n).clip(50, 350)

    # Recent performance
    baseline = rng.normal(25, 8, size=n).clip(5, 60)
    last_5_avg = baseline + rng.normal(0, 3, size=n)
    last_10_avg = baseline + rng.normal(0, 2, size=n)

    # Betting line (centered near player baseline with market noise)
    line = baseline + rng.normal(0, 4, size=n)

    # Game context
    is_home = rng.binomial(1, 0.5, size=n).astype(float)
    team_elo = rng.normal(1500, 100, size=n).clip(1200, 1800)
    opponent_elo = rng.normal(1500, 100, size=n).clip(1200, 1800)

    # Engineered features
    avg = (last_5_avg + last_10_avg) / 2
    line_vs_avg = line - avg
    elo_diff = team_elo - opponent_elo

    X = np.column_stack([
        sport_enc, stat_type_enc, line, glicko_mu, glicko_phi,
        last_5_avg, last_10_avg, is_home, team_elo, opponent_elo,
        line_vs_avg, elo_diff,
    ])

    # Generate target with realistic signal
    # Higher recent avg vs line → more likely over
    # Higher Glicko → slight edge
    # Home advantage → slight edge
    # Lower uncertainty → stronger signal
    logit = (
        -0.8 * (line_vs_avg / 10)       # Line above avg → less likely over
        + 0.3 * ((glicko_mu - 1500) / 200)  # Better player → more likely over
        + 0.15 * is_home                     # Home boost
        + 0.1 * (elo_diff / 200)             # Team strength
        - 0.2 * ((glicko_phi - 150) / 100)   # Uncertainty penalty
        + rng.normal(0, 0.5, size=n)         # Noise
    )
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob)

    return X, y


def train_and_save():
    """Train model and save to disk."""
    print(f"Generating {N_SAMPLES} synthetic samples (seed={SEED})...")
    X, y = generate_synthetic_data(N_SAMPLES, SEED)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y,
    )

    print(f"Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"Class balance: {y.mean():.1%} over / {1 - y.mean():.1%} under")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)

    print(f"\nTest metrics:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  AUC-ROC:  {auc:.3f}")
    print(f"  Log Loss: {logloss:.3f}")

    # Feature importance
    print(f"\nTop features:")
    importance = dict(zip(FEATURE_NAMES, model.feature_importances_))
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
        print(f"  {feat}: {imp:.3f}")

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"\nModel saved: {MODEL_PATH} ({size_kb:.0f} KB)")

    # Save metadata alongside model
    meta = {
        "model_version": "xgb-v3",
        "training_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features": FEATURE_NAMES,
        "accuracy": round(accuracy, 4),
        "auc_roc": round(auc, 4),
        "log_loss": round(logloss, 4),
        "seed": SEED,
        "data_source": "synthetic",
        "xgb_params": {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
        },
    }
    meta_path = MODEL_PATH.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    try:
        train_and_save()
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Install with: pip install xgboost scikit-learn numpy", file=sys.stderr)
        sys.exit(1)
