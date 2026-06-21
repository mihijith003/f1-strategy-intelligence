"""Tyre degradation model using Polynomial Regression per compound.

Trains a degree-2 polynomial regression for SOFT, MEDIUM, and HARD compounds.
The target is lap_time_delta — each lap expressed as seconds above the driver's
median pace for that race — which removes circuit-speed variation and isolates
the degradation signal. Fitted pipelines are persisted to src/models/artifacts/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.ingestion.database import load_from_db

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
CompoundLiteral = Literal["SOFT", "MEDIUM", "HARD"]


def _artifact_path(compound: str) -> Path:
    return ARTIFACTS_DIR / f"{compound.lower()}_model.pkl"


def _build_pipeline(degree: int = 2) -> Pipeline:
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lr", LinearRegression()),
    ])


def add_lap_time_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Add a lap_time_delta column normalised per driver per race.

    For each (Driver, Round) group, the baseline is the driver's median lap
    time across their laps in that race. lap_time_delta is then:
        LapTime - baseline

    This removes the circuit-speed component so the model learns only the
    degradation signal (how much slower the tyre gets as laps accumulate).

    Args:
        df: Laps DataFrame containing LapTime, Driver, and Round columns.

    Returns:
        DataFrame with a new lap_time_delta column added.
    """
    df = df.copy()
    baseline = (
        df.groupby(["Season", "Driver", "Round"])["LapTime"]
        .transform("median")
    )
    df["lap_time_delta"] = df["LapTime"] - baseline
    return df


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and lap_time_delta target from the laps DataFrame.

    Args:
        df: Laps DataFrame filtered to the desired compounds; must already
            contain the lap_time_delta column (see add_lap_time_delta).

    Returns:
        Tuple of (X, y) numpy arrays. X columns: [TyreLife, Round].
        y is lap_time_delta in seconds.
    """
    df = df.dropna(subset=["lap_time_delta", "TyreLife", "Round"]).copy()
    X = df[["TyreLife", "Round"]].to_numpy()
    y = df["lap_time_delta"].to_numpy()
    return X, y


def train_models(df: pd.DataFrame) -> dict[str, Pipeline]:
    """Train a degree-2 polynomial regression per compound on lap_time_delta.

    Args:
        df: Full laps DataFrame; will be filtered to SOFT/MEDIUM/HARD and
            enriched with lap_time_delta before training.

    Returns:
        Dict mapping compound name to fitted sklearn Pipeline.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df_enriched = add_lap_time_delta(df[df["Compound"].isin(COMPOUNDS)])

    models: dict[str, Pipeline] = {}
    print(f"\n{'Compound':<10} {'MAE (s)':>10} {'R²':>10}  Samples")
    print("-" * 45)

    for compound in COMPOUNDS:
        subset = df_enriched[df_enriched["Compound"] == compound]
        X, y = prepare_features(subset)

        pipeline = _build_pipeline(degree=2)
        pipeline.fit(X, y)

        y_pred = pipeline.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"{compound:<10} {mae:>10.4f} {r2:>10.4f}  n={len(y)}")

        joblib.dump(pipeline, _artifact_path(compound))
        models[compound] = pipeline

    return models


def plot_degradation_curves(
    df: pd.DataFrame,
    models: dict[str, Pipeline],
) -> go.Figure:
    """Plot predicted lap_time_delta vs tyre age for each compound.

    The y-axis shows seconds above the driver's baseline pace, isolating the
    degradation curve from circuit-speed differences.

    Args:
        df: Laps DataFrame (used to determine tyre-life range per compound).
        models: Dict of compound -> fitted Pipeline from train_models().

    Returns:
        Plotly Figure with one trace per compound.
    """
    colors = {"SOFT": "red", "MEDIUM": "yellow", "HARD": "white"}
    fig = go.Figure()

    df_filtered = df[df["Compound"].isin(COMPOUNDS)]
    median_round = int(df_filtered["Round"].median())

    for compound in COMPOUNDS:
        subset = df_filtered[df_filtered["Compound"] == compound].dropna(
            subset=["TyreLife"]
        )
        max_life = int(subset["TyreLife"].max())
        tyre_ages = np.arange(1, max_life + 1)

        X_plot = np.column_stack([
            tyre_ages,
            np.full(len(tyre_ages), median_round),
        ])

        y_pred = models[compound].predict(X_plot)

        fig.add_trace(go.Scatter(
            x=tyre_ages,
            y=y_pred,
            mode="lines",
            name=compound.capitalize(),
            line=dict(color=colors[compound], width=2),
        ))

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="grey",
        annotation_text="Baseline pace",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Tyre Degradation: Predicted Delta from Baseline Pace by Compound",
        xaxis_title="Tyre Life (laps)",
        yaxis_title="Seconds slower than baseline pace",
        template="plotly_dark",
        legend_title="Compound",
    )
    return fig


def predict_lap_time(compound: CompoundLiteral, tyre_life: float, round_num: int = 10) -> float:
    """Load saved model and predict the lap-time delta for a given compound and tyre age.

    Args:
        compound: One of 'SOFT', 'MEDIUM', or 'HARD'.
        tyre_life: Number of laps on the current set of tyres.
        round_num: Race round number (defaults to mid-season estimate).

    Returns:
        Predicted seconds above the driver's baseline pace (lap_time_delta).
    """
    path = _artifact_path(compound.upper())
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model for {compound}. Run train_models() first."
        )
    pipeline: Pipeline = joblib.load(path)
    X = np.array([[tyre_life, round_num]])
    return float(pipeline.predict(X)[0])


if __name__ == "__main__":
    df = load_from_db("laps")
    models = train_models(df)

    print("\nArtifacts saved:")
    for compound in COMPOUNDS:
        p = _artifact_path(compound)
        print(f"  {p}  ({p.stat().st_size / 1024:.1f} KB)")

    fig = plot_degradation_curves(df, models)
    fig.show()

    print("\nSample delta predictions (tyre_life=10, round=10):")
    for compound in COMPOUNDS:
        delta = predict_lap_time(compound, tyre_life=10, round_num=10)
        print(f"  {compound}: {delta:+.3f}s vs baseline")
