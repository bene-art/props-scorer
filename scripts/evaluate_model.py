"""Evaluate calibration on held-out test data.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --stat points --buckets 10
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from train_model import STAT_COLS, load_and_engineer, build_examples  # noqa: E402

MODELS_DIR = Path(__file__).parent.parent / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat",    default="points", choices=list(STAT_COLS))
    parser.add_argument("--buckets", type=int, default=10)
    args = parser.parse_args()

    model_name = f"nba_{args.stat}_v1"
    model_path = MODELS_DIR / f"{model_name}.joblib"
    meta_path  = MODELS_DIR / f"{model_name}.json"
    if not model_path.exists():
        print(f"No model at {model_path}. Run train_model.py first."); return

    model      = joblib.load(model_path)
    split_date = pd.Timestamp(json.loads(meta_path.read_text())["split_date"])

    df_test        = load_and_engineer(STAT_COLS[args.stat], min_games=15)
    df_test        = df_test[df_test["GAME_DATE"] >= split_date]
    X_test, y_test = build_examples(df_test, STAT_COLS[args.stat])
    proba          = model.predict_proba(X_test)[:, 1]

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=args.buckets)
    print(f"Model:       {model_name}")
    print(f"Test games:  {len(df_test):,}   examples: {len(X_test):,}")
    print(f"Brier score: {brier_score_loss(y_test, proba):.4f}")
    print(f"AUC-ROC:     {roc_auc_score(y_test, proba):.4f}")

    print("\nCalibration (predicted -> actual over-rate):")
    for pred, actual in zip(mean_pred, frac_pos):
        bar  = "█" * int(actual * 20)
        diff = actual - pred
        flag = "  <- over" if diff > 0.05 else ("  <- under" if diff < -0.05 else "")
        print(f"  {pred:.2f} -> {actual:.2f}  {bar}{flag}")

    print("\nEdge bucket analysis:")
    ev           = pd.DataFrame({"p": proba, "y": y_test})
    ev["edge"]   = (ev["p"] - 0.5).abs()
    ev["bucket"] = pd.cut(ev["edge"], bins=5)
    for bucket, grp in ev.groupby("bucket", observed=True):
        print(f"  edge {bucket}: win_rate={grp['y'].mean():.3f}  n={len(grp):,}")


if __name__ == "__main__":
    main()
