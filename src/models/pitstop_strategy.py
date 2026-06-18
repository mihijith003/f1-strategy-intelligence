"""Pit stop strategy model using a Random Forest classifier.

Detects pit stop events from lap data, engineers per-lap features, and trains
a binary classifier predicting whether a driver will pit within the next 5 laps.
The fitted model is persisted to src/models/artifacts/pitstop_model.pkl.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.ingestion.database import load_from_db
from src.models.tyre_degradation import add_lap_time_delta

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "pitstop_model.pkl"
PIT_WINDOW = 5


def detect_pit_stops(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean `pitted` column marking laps on which a pit stop occurred.

    A pit stop is inferred when TyreLife resets to 0 or 1 between consecutive
    laps for the same driver in the same race (identified by Driver + Round).

    Args:
        df: Laps DataFrame sorted by Driver, Round, LapNumber.

    Returns:
        DataFrame with a new boolean `pitted` column.
    """
    df = df.copy().sort_values(["Driver", "Round", "LapNumber"])
    prev_tyre_life = df.groupby(["Driver", "Round"])["TyreLife"].shift(1)
    # A pit stop occurred when TyreLife drops relative to the previous lap
    # (out-laps are excluded from the DB so TyreLife resets to ~2, not 0/1)
    df["pitted"] = df["TyreLife"] < prev_tyre_life
    df["pitted"] = df["pitted"].fillna(False)
    return df


def add_will_pit_next_n(df: pd.DataFrame, n: int = PIT_WINDOW) -> pd.DataFrame:
    """Add a binary target `will_pit_next_5_laps` for each lap.

    For each driver+race group, marks 1 if any of the next `n` laps contains
    a pit stop, 0 otherwise.

    Args:
        df: Laps DataFrame with a `pitted` column (see detect_pit_stops).
        n: Look-ahead window in laps.

    Returns:
        DataFrame with a new integer column `will_pit_next_5_laps`.
    """
    df = df.copy()

    def _label_group(group: pd.DataFrame) -> pd.Series:
        pit_flags = group["pitted"].to_numpy().astype(int)
        labels = np.zeros(len(pit_flags), dtype=int)
        for i in range(len(pit_flags)):
            labels[i] = int(pit_flags[i + 1 : i + 1 + n].any())
        return pd.Series(labels, index=group.index)

    df["will_pit_next_5_laps"] = (
        df.groupby(["Driver", "Round"], group_keys=False)
        .apply(_label_group, include_groups=False)
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model feature set from the laps DataFrame.

    Features produced:
        - TyreLife: laps on the current tyre set
        - Compound_encoded: ordinal encoding of SOFT/MEDIUM/HARD/INTERMEDIATE/WET
        - LapNumber: sequential lap number within the race
        - NormalisedLapTime: lap_time_delta from add_lap_time_delta (seconds above
          the driver's median pace for that race)
        - LapsRemaining: total race laps minus LapNumber

    Args:
        df: Raw laps DataFrame loaded from the database.

    Returns:
        DataFrame with feature columns and `will_pit_next_5_laps` target,
        with rows containing NaN in any feature dropped.
    """
    df = detect_pit_stops(df)
    df = add_will_pit_next_n(df, n=PIT_WINDOW)
    df = add_lap_time_delta(df)

    le = LabelEncoder()
    df["Compound_encoded"] = le.fit_transform(df["Compound"].fillna("UNKNOWN"))

    total_laps = df.groupby(["Driver", "Round"])["LapNumber"].transform("max")
    df["LapsRemaining"] = total_laps - df["LapNumber"]

    df = df.rename(columns={"lap_time_delta": "NormalisedLapTime"})

    feature_cols = [
        "TyreLife",
        "Compound_encoded",
        "LapNumber",
        "NormalisedLapTime",
        "LapsRemaining",
    ]
    df = df.dropna(subset=feature_cols + ["will_pit_next_5_laps"])
    return df[feature_cols + ["will_pit_next_5_laps"]]


DECISION_THRESHOLD = 0.35


def train_model(df: pd.DataFrame) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Train a Random Forest classifier to predict pit stop probability.

    Uses class_weight='balanced' to counter the minority-class imbalance and
    applies a lowered decision threshold (DECISION_THRESHOLD) at evaluation
    time to improve recall on pit-window alerts.

    Splits 80/20 stratified by the target, trains a RandomForestClassifier,
    and evaluates on the held-out test set.

    Args:
        df: Feature DataFrame from engineer_features().

    Returns:
        Tuple of (fitted classifier, metrics dict with accuracy/precision/recall/f1).
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c != "will_pit_next_5_laps"]
    X = df[feature_cols].to_numpy()
    y = df["will_pit_next_5_laps"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Apply lowered threshold so the model flags pit windows more aggressively
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    joblib.dump(clf, MODEL_PATH)
    return clf, metrics


def plot_feature_importance(
    clf: RandomForestClassifier,
    feature_names: list[str],
) -> go.Figure:
    """Plot feature importances from the trained Random Forest as a bar chart.

    Args:
        clf: Fitted RandomForestClassifier.
        feature_names: Ordered list of feature column names matching clf input.

    Returns:
        Plotly Figure with one bar per feature, sorted descending by importance.
    """
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_vals = importances[order]

    fig = go.Figure(
        go.Bar(
            x=sorted_names,
            y=sorted_vals,
            marker_color="steelblue",
        )
    )
    fig.update_layout(
        title="Pit Stop Model — Feature Importances",
        xaxis_title="Feature",
        yaxis_title="Importance (mean decrease in impurity)",
        template="plotly_dark",
    )
    return fig


if __name__ == "__main__":
    raw = load_from_db("laps")
    print(f"Loaded {len(raw):,} laps from database.")

    df_features = engineer_features(raw)
    print(f"Feature rows after engineering: {len(df_features):,}")
    print(f"Positive rate (will pit next 5): {df_features['will_pit_next_5_laps'].mean():.2%}")

    clf, metrics = train_model(df_features)

    print(f"\n{'Metric':<12} {'Score':>8}")
    print("-" * 22)
    for name, val in metrics.items():
        print(f"{name:<12} {val:>8.4f}")

    print(f"\nModel saved to: {MODEL_PATH}")

    feature_cols = [c for c in df_features.columns if c != "will_pit_next_5_laps"]
    fig = plot_feature_importance(clf, feature_cols)
    fig.show()
