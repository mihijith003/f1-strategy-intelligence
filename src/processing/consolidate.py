"""Consolidate per-race CSV files into a single master SQLite table.

Reads all CSVs from data/processed/, enriches each with GrandPrix and Round
columns derived from the filename, cleans the data, and saves to the 'laps'
table in data/f1_strategy.db.
"""

import re
import sys
from pathlib import Path

import pandas as pd

# Allow running from any working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingestion.database import save_to_db  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _parse_filename(path: Path) -> tuple[str, int | None]:
    """Extract GrandPrix label and Round number from a CSV filename.

    Expected pattern: 2023_r<NN>_<name>_race_laps.csv
    Falls back gracefully if no round number is present.

    Args:
        path: Path to the CSV file.

    Returns:
        Tuple of (grand_prix_name, round_number_or_None).
    """
    stem = path.stem  # e.g. 2023_r01_bahrain_grand_prix_race_laps
    match = re.match(r"^\d{4}_r(\d+)_(.+?)_race_laps$", stem)
    if match:
        round_num = int(match.group(1))
        raw_name = match.group(2).replace("_", " ").title()
        return raw_name, round_num

    # Fallback: no round number in filename
    fallback_match = re.match(r"^\d{4}_(.+?)_race_laps$", stem)
    if fallback_match:
        raw_name = fallback_match.group(1).replace("_", " ").title()
        return raw_name, None

    return stem, None


def _laptime_to_seconds(series: pd.Series) -> pd.Series:
    """Convert LapTime strings to total seconds as float.

    Handles two formats:
    - Timedelta string: '0 days 00:01:37.974000'
    - Plain seconds float already stored as string

    Args:
        series: Raw LapTime column.

    Returns:
        Float series of lap time in seconds.
    """
    converted = pd.to_timedelta(series, errors="coerce")
    valid_mask = converted.notna()

    result = pd.Series(index=series.index, dtype=float)
    result[valid_mask] = converted[valid_mask].dt.total_seconds()

    # Rows that failed timedelta parse — try numeric fallback
    fallback_mask = ~valid_mask
    result[fallback_mask] = pd.to_numeric(series[fallback_mask], errors="coerce")

    return result


def load_and_tag(path: Path) -> pd.DataFrame:
    """Load a single race CSV and add GrandPrix and Round columns.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with GrandPrix and Round columns prepended.
    """
    df = pd.read_csv(path)
    grand_prix, round_num = _parse_filename(path)
    df.insert(0, "GrandPrix", grand_prix)
    df.insert(1, "Round", round_num)
    return df


def consolidate(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Read, tag, and combine all race CSVs into one master DataFrame.

    Args:
        processed_dir: Directory containing per-race CSV files.

    Returns:
        Cleaned master DataFrame.

    Raises:
        FileNotFoundError: If no CSV files are found in processed_dir.
    """
    csv_files = sorted(processed_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {processed_dir}")

    frames = [load_and_tag(p) for p in csv_files]
    master = pd.concat(frames, ignore_index=True)

    # --- Cleaning ---
    # 1. Drop rows where IsAccurate is False
    if "IsAccurate" in master.columns:
        master = master[master["IsAccurate"].astype(str).str.lower() != "false"]

    # 2. Convert LapTime to seconds
    if "LapTime" in master.columns:
        master["LapTime"] = _laptime_to_seconds(master["LapTime"])

    # 3. Drop nulls in key columns
    master = master.dropna(subset=["Compound", "TyreLife", "Driver"])

    master = master.reset_index(drop=True)
    return master


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise summary of the consolidated DataFrame.

    Args:
        df: The master laps DataFrame.
    """
    print("=" * 50)
    print("Consolidation summary")
    print("=" * 50)
    print(f"Total rows      : {len(df):,}")
    print(f"Unique drivers  : {df['Driver'].nunique()}")
    print(f"Unique races    : {df['GrandPrix'].nunique()}")

    if "LapTime" in df.columns:
        valid_times = df["LapTime"].dropna()
        if not valid_times.empty:
            min_s = valid_times.min()
            max_s = valid_times.max()
            print(f"LapTime range   : {min_s:.3f}s – {max_s:.3f}s")

    rounds = df["Round"].dropna()
    if not rounds.empty:
        print(f"Round range     : R{int(rounds.min())} – R{int(rounds.max())}")
    print("=" * 50)


def main() -> None:
    """Entry point: consolidate CSVs, save to SQLite, print summary."""
    print(f"Scanning {PROCESSED_DIR} ...")
    master = consolidate()

    print(f"Saving {len(master):,} rows to SQLite table 'laps' ...")
    save_to_db(master, "laps")
    print("Saved.")

    print_summary(master)


if __name__ == "__main__":
    main()
