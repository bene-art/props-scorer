"""Fetch NBA player game logs and cache locally.

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --seasons 2023-24 2024-25
    python scripts/fetch_data.py --force
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DEFAULT_SEASONS = ["2023-24", "2024-25"]


def fetch_game_logs(season: str, delay: float = 0.6) -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueGameLog
    print(f"  game logs {season}...", flush=True)
    ep = LeagueGameLog(season=season, season_type_all_star="Regular Season",
                       player_or_team_abbreviation="P")
    time.sleep(delay)
    df = ep.get_data_frames()[0]
    df["SEASON"] = season
    return df


def fetch_team_defense(season: str, delay: float = 0.6) -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueDashTeamStats
    print(f"  team defense {season}...", flush=True)
    ep = LeagueDashTeamStats(season=season, season_type_all_star="Regular Season",
                             measure_type_detailed_defense="Advanced")
    time.sleep(delay)
    df = ep.get_data_frames()[0]
    return df[["TEAM_ID", "TEAM_ABBREVIATION", "DEF_RATING"]].assign(SEASON=season)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    all_logs, all_defense = [], []
    for season in args.seasons:
        slug     = season.replace("-", "_")
        log_path = DATA_DIR / f"game_logs_{slug}.csv"
        def_path = DATA_DIR / f"team_defense_{slug}.csv"
        print(f"Season {season}:")
        if log_path.exists() and not args.force:
            print("  game logs   (cached)")
            logs = pd.read_csv(log_path, parse_dates=["GAME_DATE"])
        else:
            logs = fetch_game_logs(season)
            logs.to_csv(log_path, index=False)
        if def_path.exists() and not args.force:
            print("  team defense (cached)")
            defense = pd.read_csv(def_path)
        else:
            defense = fetch_team_defense(season)
            defense.to_csv(def_path, index=False)
        all_logs.append(logs)
        all_defense.append(defense)
        print(f"  {len(logs):,} player-games")

    pd.concat(all_logs,    ignore_index=True).to_csv(DATA_DIR / "game_logs_combined.csv",    index=False)
    pd.concat(all_defense, ignore_index=True).to_csv(DATA_DIR / "team_defense_combined.csv", index=False)
    print(f"\nTotal: {sum(len(l) for l in all_logs):,} player-games  |  data saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
