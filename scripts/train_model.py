"""Train a calibrated XGBoost model on real NBA game log data.

Requires data from scripts/fetch_data.py.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --stat points --min-games 15
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

STAT_COLS = {"points": "PTS", "rebounds": "REB", "assists": "AST", "threes": "FG3M"}

FEATURES = [
    "rolling_5_avg", "rolling_10_avg", "rolling_5_std",
    "rest_days", "is_home", "opp_def_rtg", "games_played",
    "line_vs_avg", "avg_trend",
]

# 7 line offsets ±20%: teaches the model how line position shifts P(over)
LINE_OFFSETS = np.linspace(-0.20, 0.20, 7)


def load_and_engineer(stat_col: str, min_games: int) -> pd.DataFrame:
    logs    = pd.read_csv(DATA_DIR / "game_logs_combined.csv",    parse_dates=["GAME_DATE"])
    defense = pd.read_csv(DATA_DIR / "team_defense_combined.csv")

    logs["is_home"]  = logs["MATCHUP"].str.contains(r"vs\.", regex=True).astype(int)
    logs["opp_abbr"] = logs["MATCHUP"].str.extract(r"(?:vs\.|@)\s+(\w+)")

    def_map = defense.set_index(["TEAM_ABBREVIATION", "SEASON"])["DEF_RATING"].to_dict()
    logs["opp_def_rtg"] = logs.apply(
        lambda r: def_map.get((r["opp_abbr"], r["SEASON"]), 110.0), axis=1)

    logs = logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    grp  = logs.groupby("PLAYER_ID")[stat_col]

    # shift(1) on all rolling features — no lookahead into the current game
    logs["rolling_5_avg"]  = grp.transform(lambda x: x.shift(1).rolling(5,  min_periods=3).mean())
    logs["rolling_10_avg"] = grp.transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    logs["rolling_5_std"]  = grp.transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).std().fillna(3.0))

    logs["prev_date"]    = logs.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    logs["rest_days"]    = (logs["GAME_DATE"] - logs["prev_date"]).dt.days.clip(0, 10).fillna(2)
    logs["games_played"] = logs.groupby(["PLAYER_ID", "SEASON"]).cumcount()

    logs = logs.dropna(subset=["rolling_5_avg", "rolling_10_avg"])
    return logs[logs["games_played"] >= min_games]


def build_examples(df: pd.DataFrame, stat_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    """7 training examples per game via synthetic line variation."""
    rows, targets = [], []
    for _, row in df.iterrows():
        base = row["rolling_10_avg"]
        for offset in LINE_OFFSETS:
            line = base * (1 + offset)
            rows.append({
                "rolling_5_avg":  row["rolling_5_avg"],
                "rolling_10_avg": base,
                "rolling_5_std":  row["rolling_5_std"],
                "rest_days":      row["rest_days"],
                "is_home":        row["is_home"],
                "opp_def_rtg":    row["opp_def_rtg"],
                "games_played":   min(row["games_played"], 60),
                "line_vs_avg":    line - base,
                "avg_trend":      row["rolling_5_avg"] - base,
            })
            targets.append(int(row[stat_col] > line))
    return pd.DataFrame(rows, columns=FEATURES), np.array(targets)


def report(label: str, y: np.ndarray, p: np.ndarray) -> dict:
    m = {"brier": round(float(brier_score_loss(y,p)),4),
         "log_loss": round(float(log_loss(y,p)),4),
         "auc_roc": round(float(roc_auc_score(y,p)),4)}
    print(f"  {label}: brier={m['brier']}  log_loss={m['log_loss']}  auc={m['auc_roc']}")
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat",      default="points", choices=list(STAT_COLS))
    parser.add_argument("--min-games", type=int, default=15)
    args = parser.parse_args()

    stat_col   = STAT_COLS[args.stat]
    model_name = f"nba_{args.stat}_v1"
    print(f"Training {model_name}...")

    df = load_and_engineer(stat_col, args.min_games)
    print(f"Games: {len(df):,}  ({df['PLAYER_ID'].nunique()} players)")

    X, y = build_examples(df, stat_col)
    print(f"Examples: {len(X):,}  (x{len(LINE_OFFSETS)} line offsets)")

    # Time-based split — last 20% of games held out
    split_date = df.iloc[int(len(df)*0.80)]["GAME_DATE"]
    n_aug      = len(LINE_OFFSETS)
    game_train = (df["GAME_DATE"] < split_date).values
    mask       = np.repeat(game_train, n_aug)
    X_tr, X_te = X[mask], X[~mask]
    y_tr, y_te = y[mask], y[~mask]
    print(f"Train: {len(X_tr):,}  |  Test: {len(X_te):,}  (split {split_date.date()})")

    xgb_params = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      eval_metric="logloss", random_state=42, verbosity=0)
    model = CalibratedClassifierCV(XGBClassifier(**xgb_params), method="isotonic", cv=5)
    model.fit(X_tr, y_tr)

    base_raw = XGBClassifier(**xgb_params)
    base_raw.fit(X_tr, y_tr)

    print("\nEvaluation:")
    m_raw = report("XGBoost (uncalibrated)", y_te, base_raw.predict_proba(X_te)[:,1])
    m_cal = report("XGBoost + isotonic    ", y_te, model.predict_proba(X_te)[:,1])

    print("\nFeature importances:")
    for feat, imp in sorted(zip(FEATURES, base_raw.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:<18} {imp:.3f}")

    model_path = MODELS_DIR / f"{model_name}.joblib"
    meta_path  = MODELS_DIR / f"{model_name}.json"
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps({
        "model_version": model_name, "sport": "NBA", "stat": args.stat,
        "features": FEATURES, "calibration": "isotonic",
        "games_train": int(game_train.sum()), "games_test": int((~game_train).sum()),
        "split_date": str(split_date.date()),
        "metrics_uncalibrated": m_raw, "metrics_calibrated": m_cal,
        "data_source": "nba_api (real game logs)",
    }, indent=2))

    delta = m_cal["brier"] - m_raw["brier"]
    print(f"\nSaved {model_path.name}  |  brier {m_raw['brier']} -> {m_cal['brier']}  ({delta:+.4f})")


if __name__ == "__main__":
    main()
