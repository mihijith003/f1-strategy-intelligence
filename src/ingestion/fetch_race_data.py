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


if __name__ == "__main__":
    laps_df = load_session(2023, "British Grand Prix", "R")
    print(laps_df.head(5))
    saved_path = save_to_csv(laps_df, "2023_british_gp_race_laps.csv")
    print(f"Saved to {saved_path}")
