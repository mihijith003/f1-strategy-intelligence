"""SQLite persistence helpers for processed F1 strategy data.

Uses SQLAlchemy to write and read pandas DataFrames to/from a local
SQLite database file at ./data/f1_strategy.db.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

DB_PATH = Path("./data/f1_strategy.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine: Engine = create_engine(f"sqlite:///{DB_PATH}")


def save_to_db(df: pd.DataFrame, table_name: str) -> None:
    """Write a DataFrame to the SQLite database, replacing any existing table.

    Args:
        df: DataFrame to persist.
        table_name: Name of the destination table.
    """
    df.to_sql(table_name, engine, if_exists="replace", index=False)


def load_from_db(table_name: str) -> pd.DataFrame:
    """Read a table back from the SQLite database.

    Args:
        table_name: Name of the table to read.

    Returns:
        The table contents as a DataFrame.
    """
    return pd.read_sql_table(table_name, engine)
