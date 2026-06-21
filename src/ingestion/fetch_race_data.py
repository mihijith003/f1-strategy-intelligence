"""FastF1-based ingestion of race/qualifying lap data.

Loads a FastF1 session, extracts the relevant lap-level columns used for
race strategy analysis, and provides a helper to persist the result to CSV.
"""

from pathlib import Path

import fastf1
import pandas as pd

RAW_CACHE_DIR = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")

RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(RAW_CACHE_DIR))

LAP_COLUMNS = [
    "LapNumber",
    "Driver",
    "Team",
    "LapTime",
    "Compound",
    "TyreLife",
    "PitOutTime",
    "PitInTime",
    "IsAccurate",
]


def load_session(year: int, grand_prix: str, session_type: str) -> pd.DataFrame:
    """Load a FastF1 session and return its lap data.

    Args:
        year: Season year, e.g. 2023.
        grand_prix: Grand Prix name or round identifier, e.g. "British Grand Prix".
        session_type: FastF1 session identifier, e.g. "R" (Race) or "Q" (Qualifying).

    Returns:
        A DataFrame of lap data containing LapNumber, Driver, Team, LapTime,
        Compound, TyreLife, PitOutTime, PitInTime, and IsAccurate.
    """
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load()
    laps = session.laps
    return laps[LAP_COLUMNS].copy()


def save_to_csv(df: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame to ./data/processed/ as CSV.

    Args:
        df: DataFrame to save.
        filename: Output filename, e.g. "2023_british_gp_race_laps.csv".

    Returns:
        The full path the file was written to.
    """
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path


def load_full_season(year: int) -> None:
    """Load race lap data for every round of a season and save each to CSV.

    Args:
        year: Season year, e.g. 2023.
    """
    import logging

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rounds = schedule[schedule["EventFormat"] != "testing"]
    total = len(rounds)
    succeeded = 0

    for _, event in rounds.iterrows():
        round_num = int(event["RoundNumber"])
        gp_name: str = event["EventName"]
        slug = gp_name.lower().replace(" ", "_")
        filename = f"{year}_r{round_num:02d}_{slug}_race_laps.csv"
        output_path = PROCESSED_DIR / filename

        if output_path.exists():
            print(f"[{round_num:02d}/{total}] {gp_name} — already saved, skipping")
            succeeded += 1
            continue

        print(f"[{round_num:02d}/{total}] Loading {gp_name} ...", end=" ", flush=True)
        try:
            laps_df = load_session(year, round_num, "R")
            laps_df["Season"] = year
            save_to_csv(laps_df, filename)
            print(f"saved {len(laps_df)} laps -> {filename}")
            succeeded += 1
        except Exception as exc:
            logging.error("Failed to fetch %s round %d: %s", gp_name, round_num, exc)
            print(f"ERROR — {exc}")

    print(f"\nCompleted {succeeded}/{total} races")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch F1 race lap data for a given season.")
    parser.add_argument("--year", type=int, default=2023, help="Season year (default: 2023)")
    args = parser.parse_args()

    load_full_season(args.year)
