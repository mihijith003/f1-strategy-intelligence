"""F1 Strategy Intelligence Dashboard.

Multi-page Streamlit app with Race Explorer, Tyre Degradation analysis,
and Strategy Simulator pages backed by SQLite lap data and trained ML models.
"""

from __future__ import annotations

from pathlib import Path

from groq import Groq
import fastf1
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path("data/f1_strategy.db")
ARTIFACTS_DIR = Path("src/models/artifacts")

COMPOUND_COLOURS = {
    "SOFT": "#e8002d",
    "MEDIUM": "#ffd700",
    "HARD": "#ffffff",
    "INTERMEDIATE": "#39b54a",
    "WET": "#0067ff",
}

# Alphabetical LabelEncoder order used during pitstop model training:
# HARD=0, INTERMEDIATE=1, MEDIUM=2, SOFT=3, UNKNOWN=4, WET=5
COMPOUND_ENCODING = {"HARD": 0, "INTERMEDIATE": 1, "MEDIUM": 2, "SOFT": 3, "UNKNOWN": 4, "WET": 5}

DARK_BG = "#15151e"
ACCENT_RED = "#e8002d"
ACCENT_TEAL = "#00d2be"
ACCENT_BLUE = "#3671c6"

DRIVER_NAMES = {
    "VER": "Verstappen", "PER": "Pérez", "ALO": "Alonso",
    "HAM": "Hamilton", "RUS": "Russell", "LEC": "Leclerc",
    "SAI": "Sainz", "NOR": "Norris", "PIA": "Piastri",
    "STR": "Stroll", "OCO": "Ocon", "GAS": "Gasly",
    "ALB": "Albon", "TSU": "Tsunoda", "BOT": "Bottas",
    "ZHO": "Zhou", "HUL": "Hülkenberg", "MAG": "Magnussen",
    "SAR": "Sargeant", "DEV": "De Vries", "LAW": "Lawson",
    "RIC": "Ricciardo", "BEA": "Bearman", "COL": "Colapinto",
}

# Official team colours for the position chart lines.
# FastF1's Team strings vary slightly between seasons, so we map several variants.
TEAM_COLOURS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",
    "AlphaTauri": "#6692FF",
    "RB": "#6692FF",
    "Visa Cash App RB Formula One Team": "#6692FF",
    "Alfa Romeo": "#C92D4B",
    "Kick Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
    "Haas": "#B6BABD",
}

# 20 visually distinct fallbacks for any team name not in the map above
FALLBACK_COLOURS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor="#1e1e2e",
    font=dict(color="#e0e0e0"),
    xaxis=dict(gridcolor="#2a2a3e", zerolinecolor="#2a2a3e"),
    yaxis=dict(gridcolor="#2a2a3e", zerolinecolor="#2a2a3e"),
)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="F1 Strategy Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject dark background CSS
st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {DARK_BG}; }}
        section[data-testid="stSidebar"] {{ background-color: #0d0d16; }}
        .metric-card {{
            background: #1e1e2e;
            border-radius: 8px;
            padding: 16px 20px;
            text-align: center;
            border: 1px solid #2a2a3e;
        }}
        .metric-card .label {{ color: #888; font-size: 0.85rem; margin-bottom: 4px; }}
        .metric-card .value {{ font-size: 1.6rem; font-weight: 700; }}
        .indicator {{
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            font-size: 1.4rem;
            font-weight: 700;
            border: 2px solid;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data helpers (cached)
# ---------------------------------------------------------------------------


@st.cache_data
def load_laps() -> pd.DataFrame:
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM laps", conn)
    conn.close()
    return df


@st.cache_resource
def load_tyre_models() -> dict[str, Pipeline]:
    models = {}
    for compound in ("SOFT", "MEDIUM", "HARD"):
        path = ARTIFACTS_DIR / f"{compound.lower()}_model.pkl"
        if path.exists():
            models[compound] = joblib.load(path)
    return models


@st.cache_resource
def load_pitstop_model() -> RandomForestClassifier:
    path = ARTIFACTS_DIR / "pitstop_model.pkl"
    return joblib.load(path)


def compute_lap_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Add lap_time_delta = LapTime minus per-driver-per-race median.

    Groups by Season + Driver + Round so baselines are never computed across
    season boundaries, even if multi-season data is accidentally passed in.
    """
    df = df.copy()
    group_cols = ["Season", "Driver", "Round"] if "Season" in df.columns else ["Driver", "Round"]
    baseline = df.groupby(group_cols)["LapTime"].transform("median")
    df["lap_time_delta"] = df["LapTime"] - baseline
    return df


def detect_pit_laps(df: pd.DataFrame) -> set[int]:
    """Return the set of LapNumbers where a pit stop occurred (TyreLife reset)."""
    df = df.sort_values("LapNumber").copy()
    prev = df["TyreLife"].shift(1)
    pit_mask = df["TyreLife"] < prev
    return set(df.loc[pit_mask, "LapNumber"].tolist())


@st.cache_data(show_spinner=False)
def load_session_positions(season: int, round_num: int) -> pd.DataFrame:
    """Load lap-by-lap Position data from local FastF1 cache.

    Uses st.cache_data so a race is only parsed once per Streamlit session —
    subsequent filter changes on the same race are instant.
    FastF1 reads from data/raw/ (the local disk cache) and does NOT
    re-download anything.
    """
    fastf1.Cache.enable_cache(str(Path("data/raw")))
    session = fastf1.get_session(int(season), int(round_num), "R")
    # Load only lap data — skipping telemetry/weather keeps startup fast
    session.load(laps=True, telemetry=False, weather=False, messages=False)

    cols = ["LapNumber", "Driver", "Position"]
    if "Team" in session.laps.columns:
        cols.append("Team")

    pos_df = session.laps[cols].copy()
    pos_df = pos_df.dropna(subset=["Position"])
    pos_df["Position"] = pos_df["Position"].astype(int)
    pos_df["LapNumber"] = pos_df["LapNumber"].astype(int)
    return pos_df


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"<p style='color:{ACCENT_RED}; font-size:26px; font-weight:700; margin-bottom:0; line-height:1.2'>🏎 F1 Strategy</p>"
        "<p style='color:#888; font-size:0.8rem; margin-top:2px'>Intelligence Dashboard</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Navigate",
        ["Race Explorer", "Tyre Degradation", "Strategy Simulator", "Position Chart", "AI Race Analyst"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        "<p style='color:#555; font-size:0.75rem'>Data: FastF1 · Models: scikit-learn</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Page 1 — Race Explorer
# ---------------------------------------------------------------------------

if page == "Race Explorer":
    st.title("Race Explorer")
    st.markdown("Explore lap-by-lap pace and tyre strategy for any driver in any race.")

    laps = load_laps()

    # Three-column selector: Season → Grand Prix → Driver
    # Season must be chosen first so the GP list is scoped to that year only.
    # Without this filter, same-named races (e.g. "British Grand Prix") in both
    # 2023 and 2024 would be merged into one chart, doubling pit markers and
    # producing completely nonsensical lap time patterns.
    col0, col1, col2 = st.columns(3)
    with col0:
        available_seasons = sorted(laps["Season"].dropna().unique().astype(int), reverse=True)
        selected_season = st.selectbox("Season", available_seasons, format_func=str)

    season_laps = laps[laps["Season"] == selected_season]

    with col1:
        gp_options = sorted(season_laps["GrandPrix"].dropna().unique())
        selected_gp = st.selectbox("Grand Prix", gp_options)

    race_laps = season_laps[season_laps["GrandPrix"] == selected_gp]

    with col2:
        driver_codes = sorted(race_laps["Driver"].dropna().unique())
        driver_labels = [DRIVER_NAMES.get(code, code) for code in driver_codes]
        selected_label = st.selectbox("Driver", driver_labels)
        selected_driver = driver_codes[driver_labels.index(selected_label)]

    driver_laps = race_laps[race_laps["Driver"] == selected_driver].copy()
    driver_laps = compute_lap_delta(driver_laps)
    driver_laps = driver_laps.sort_values("LapNumber")

    pit_laps = detect_pit_laps(driver_laps)

    # Build chart
    fig = go.Figure()

    for compound in driver_laps["Compound"].dropna().unique():
        sub = driver_laps[driver_laps["Compound"] == compound]
        colour = COMPOUND_COLOURS.get(compound, "#aaaaaa")
        fig.add_trace(go.Scatter(
            x=sub["LapNumber"],
            y=sub["lap_time_delta"],
            mode="lines+markers",
            name=compound.capitalize(),
            line=dict(color=colour, width=2),
            marker=dict(size=5),
            hovertemplate="Lap %{x}<br>Delta: %{y:+.3f}s<extra>" + compound + "</extra>",
        ))

    # Pit stop markers
    for pit_lap in sorted(pit_laps):
        fig.add_vline(
            x=pit_lap,
            line_dash="dash",
            line_color=ACCENT_TEAL,
            line_width=1.5,
            annotation_text="PIT",
            annotation_font_color=ACCENT_TEAL,
            annotation_font_size=10,
        )

    fig.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"{DRIVER_NAMES.get(selected_driver, selected_driver)} — {selected_gp}",
            font=dict(size=18, color="#e0e0e0"),
        ),
        xaxis_title="Lap Number",
        yaxis_title="Lap Time Delta (s vs driver baseline)",
        legend_title="Compound",
        hovermode="x unified",
        height=480,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics row
    st.divider()
    compounds_used = driver_laps["Compound"].dropna().unique()
    metric_cols = st.columns(len(compounds_used))
    for col, compound in zip(metric_cols, sorted(compounds_used)):
        sub = driver_laps[driver_laps["Compound"] == compound]
        avg_delta = sub["lap_time_delta"].mean()
        colour = COMPOUND_COLOURS.get(compound, "#aaa")
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>{compound.capitalize()} avg delta</div>"
            f"<div class='value' style='color:{colour}'>{avg_delta:+.3f}s</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Page 2 — Tyre Degradation
# ---------------------------------------------------------------------------

elif page == "Tyre Degradation":
    st.title("Tyre Degradation")
    st.markdown("Predicted lap-time delta vs tyre age for each compound, using polynomial regression models trained on 2023–2024 race data.")

    laps = load_laps()
    models = load_tyre_models()

    if not models:
        st.error("No tyre model artifacts found. Run `python -m src.models.tyre_degradation` to train them.")
        st.stop()

    df_filtered = laps[laps["Compound"].isin(("SOFT", "MEDIUM", "HARD"))].copy()
    median_round = int(df_filtered["Round"].median())

    fig = go.Figure()
    degradation_rates: dict[str, float] = {}

    for compound in ("SOFT", "MEDIUM", "HARD"):
        if compound not in models:
            continue

        subset = df_filtered[df_filtered["Compound"] == compound].dropna(subset=["TyreLife"])
        max_life = min(int(subset["TyreLife"].max()), 50)
        tyre_ages = np.arange(1, max_life + 1)

        X_plot = np.column_stack([tyre_ages, np.full(len(tyre_ages), median_round)])
        y_pred = models[compound].predict(X_plot)

        colour = COMPOUND_COLOURS[compound]
        fig.add_trace(go.Scatter(
            x=tyre_ages,
            y=y_pred,
            mode="lines",
            name=compound.capitalize(),
            line=dict(color=colour, width=2.5),
            hovertemplate="Lap %{x}<br>Delta: %{y:+.3f}s<extra>" + compound + "</extra>",
        ))

        # Degradation rate: slope between lap 5 and max (seconds per lap)
        if len(tyre_ages) > 5:
            rate = (y_pred[-1] - y_pred[4]) / (tyre_ages[-1] - tyre_ages[4])
        else:
            rate = (y_pred[-1] - y_pred[0]) / max(len(y_pred) - 1, 1)
        degradation_rates[compound] = rate

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#555",
        line_width=1,
        annotation_text="Baseline pace",
        annotation_font_color="#888",
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="Tyre Degradation — Predicted Delta from Baseline Pace",
            font=dict(size=18, color="#e0e0e0"),
        ),
        xaxis_title="Tyre Life (laps)",
        yaxis_title="Seconds slower than baseline pace",
        legend_title="Compound",
        height=460,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Degradation rate cards
    st.subheader("Degradation Rate")
    st.caption("Average seconds lost per lap as the tyre ages (laps 5 onward).")
    rate_cols = st.columns(3)
    for col, compound in zip(rate_cols, ("SOFT", "MEDIUM", "HARD")):
        rate = degradation_rates.get(compound)
        if rate is None:
            continue
        colour = COMPOUND_COLOURS[compound]
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>{compound.capitalize()}</div>"
            f"<div class='value' style='color:{colour}'>{rate:+.4f} s/lap</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Page 3 — Strategy Simulator
# ---------------------------------------------------------------------------

elif page == "Strategy Simulator":
    st.title("Strategy Simulator")
    st.markdown("Predict lap-time delta and pit stop recommendation based on current tyre state.")

    col_inputs, col_outputs = st.columns([1, 1], gap="large")

    with col_inputs:
        st.subheader("Current Conditions")
        compound = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"])
        tyre_life = st.slider("Tyre Life (laps on current set)", min_value=1, max_value=40, value=10)
        laps_remaining = st.slider("Laps Remaining", min_value=1, max_value=60, value=20)

    # Load models
    tyre_models = load_tyre_models()
    pitstop_model = load_pitstop_model()

    # --- Tyre model prediction ---
    lap_delta: float | None = None
    if compound in tyre_models:
        laps_df = load_laps()
        median_round = int(laps_df["Round"].median())
        X_tyre = np.array([[tyre_life, median_round]])
        lap_delta = float(tyre_models[compound].predict(X_tyre)[0])

    # --- Pitstop model prediction ---
    # Features: [TyreLife, Compound_encoded, LapNumber, NormalisedLapTime, LapsRemaining]
    # LapNumber estimated from total_laps (median ~57) - laps_remaining
    ASSUMED_TOTAL_LAPS = 57
    lap_number = max(1, ASSUMED_TOTAL_LAPS - laps_remaining)
    compound_enc = COMPOUND_ENCODING.get(compound, 4)
    norm_lap_time = lap_delta if lap_delta is not None else 0.0

    X_pit = np.array([[tyre_life, compound_enc, lap_number, norm_lap_time, laps_remaining]])
    pit_proba = float(pitstop_model.predict_proba(X_pit)[0, 1])

    # Decision thresholds (mirroring DECISION_THRESHOLD=0.35 from training)
    if pit_proba >= 0.60:
        recommendation = "Pit Now"
        indicator_colour = ACCENT_RED
        border_colour = ACCENT_RED
    elif pit_proba >= 0.35:
        recommendation = "Consider Pit"
        indicator_colour = "#f5a623"
        border_colour = "#f5a623"
    else:
        recommendation = "Stay Out"
        indicator_colour = "#27ae60"
        border_colour = "#27ae60"

    with col_outputs:
        st.subheader("Predictions")

        # Lap time delta card
        if lap_delta is not None:
            delta_str = f"{lap_delta:+.3f}s"
            delta_colour = ACCENT_RED if lap_delta > 1.0 else ("#f5a623" if lap_delta > 0.3 else "#27ae60")
            st.markdown(
                f"<div class='metric-card' style='margin-bottom:16px'>"
                f"<div class='label'>Predicted Lap Time Delta (vs baseline)</div>"
                f"<div class='value' style='color:{delta_colour}'>{delta_str}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Pit recommendation indicator
        st.markdown(
            f"<div class='indicator' style='background:{indicator_colour}22; border-color:{border_colour}; color:{indicator_colour}; margin-bottom:16px'>"
            f"{recommendation}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Confidence bar
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>Pit Probability</div>"
            f"<div class='value' style='color:{indicator_colour}'>{pit_proba:.1%}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(pit_proba, text="")

    # --- Degradation curve for selected compound ---
    st.divider()
    st.subheader(f"{compound.capitalize()} Degradation Curve")

    if compound in tyre_models:
        laps_df = load_laps()
        median_round = int(laps_df["Round"].median())
        ages = np.arange(1, 41)
        X_curve = np.column_stack([ages, np.full(len(ages), median_round)])
        y_curve = tyre_models[compound].predict(X_curve)

        colour = COMPOUND_COLOURS[compound]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ages,
            y=y_curve,
            mode="lines",
            line=dict(color=colour, width=2.5),
            name=compound.capitalize(),
            hovertemplate="Lap %{x}<br>Delta: %{y:+.3f}s<extra></extra>",
        ))
        # Current tyre life marker
        if lap_delta is not None:
            fig.add_trace(go.Scatter(
                x=[tyre_life],
                y=[lap_delta],
                mode="markers",
                marker=dict(size=12, color=ACCENT_TEAL, symbol="circle", line=dict(color="white", width=2)),
                name="Current position",
                hovertemplate=f"Tyre Life: {tyre_life} laps<br>Delta: {lap_delta:+.3f}s<extra></extra>",
            ))
        fig.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)
        fig.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Tyre Life (laps)",
            yaxis_title="Lap Time Delta (s)",
            height=300,
            showlegend=True,
            margin=dict(t=20),
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 4 — Animated Position Chart
# ---------------------------------------------------------------------------

elif page == "Position Chart":
    st.title("🏁 Race Positions")
    st.markdown(
        "Lap-by-lap position changes for every driver. "
        "Hit **▶ Play** to animate the race, or drag the lap slider."
    )

    laps = load_laps()

    col1, col2 = st.columns(2)
    with col1:
        available_seasons = sorted(laps["Season"].dropna().unique().astype(int), reverse=True)
        selected_season = st.selectbox("Season", available_seasons, format_func=str, key="pos_season")

    # Build an ordered race list for the chosen season from SQLite
    # (avoids hardcoding round numbers — they differ between seasons)
    season_laps = laps[laps["Season"] == selected_season]
    race_index = (
        season_laps.groupby(["Round", "GrandPrix"])
        .size()
        .reset_index()[["Round", "GrandPrix"]]
        .sort_values("Round")
        .reset_index(drop=True)
    )

    with col2:
        gp_list = race_index["GrandPrix"].tolist()
        selected_gp = st.selectbox("Grand Prix", gp_list, key="pos_gp")

    selected_round = int(race_index.loc[race_index["GrandPrix"] == selected_gp, "Round"].iloc[0])

    # Load positions from FastF1 cache (cached after first call)
    with st.spinner(f"Loading {selected_gp} {selected_season} from local cache…"):
        pos_df = load_session_positions(selected_season, selected_round)

    if pos_df.empty:
        st.error("No position data found for this race.")
        st.stop()

    # ---- Build driver → colour map using team colours where available ----
    drivers = sorted(pos_df["Driver"].unique())
    driver_team: dict[str, str] = {}
    if "Team" in pos_df.columns:
        driver_team = pos_df.groupby("Driver")["Team"].first().to_dict()

    driver_colours: dict[str, str] = {}
    fallback_idx = 0
    for driver in drivers:
        team = driver_team.get(driver, "")
        if team in TEAM_COLOURS:
            driver_colours[driver] = TEAM_COLOURS[team]
        else:
            driver_colours[driver] = FALLBACK_COLOURS[fallback_idx % len(FALLBACK_COLOURS)]
            fallback_idx += 1

    max_lap = int(pos_df["LapNumber"].max())
    num_drivers = len(drivers)

    # ---- Initial traces: lap 1 only ----
    # Full trace spec lives here; frames only update x/y to keep JSON small.
    initial_traces = []
    for driver in drivers:
        d1 = pos_df[(pos_df["Driver"] == driver) & (pos_df["LapNumber"] == 1)]
        initial_traces.append(go.Scatter(
            x=d1["LapNumber"].tolist(),
            y=d1["Position"].tolist(),
            mode="lines+markers",
            name=DRIVER_NAMES.get(driver, driver),
            line=dict(color=driver_colours[driver], width=2),
            marker=dict(size=5, color=driver_colours[driver]),
            hovertemplate="<b>%{fullData.name}</b><br>Lap %{x} · P%{y}<extra></extra>",
        ))

    # ---- Animation frames: each frame extends every driver line by one lap ----
    frames = []
    for lap in range(1, max_lap + 1):
        frame_traces = []
        for driver in drivers:
            d_laps = (
                pos_df[(pos_df["Driver"] == driver) & (pos_df["LapNumber"] <= lap)]
                .sort_values("LapNumber")
            )
            # Only x/y needed — all other style properties inherit from initial_traces
            frame_traces.append(go.Scatter(
                x=d_laps["LapNumber"].tolist(),
                y=d_laps["Position"].tolist(),
            ))
        frames.append(go.Frame(data=frame_traces, name=str(lap)))

    # ---- Slider steps (one per lap) ----
    slider_steps = [
        dict(
            method="animate",
            args=[
                [str(lap)],
                dict(mode="immediate", frame=dict(duration=150, redraw=True), transition=dict(duration=0)),
            ],
            label=str(lap),
        )
        for lap in range(1, max_lap + 1)
    ]

    fig = go.Figure(
        data=initial_traces,
        frames=frames,
        layout=go.Layout(
            # Inline the three non-conflicting properties from PLOTLY_LAYOUT.
            # Can't use **PLOTLY_LAYOUT here because it already defines xaxis
            # and yaxis, and we need custom versions of both for this chart.
            paper_bgcolor=DARK_BG,
            plot_bgcolor="#1e1e2e",
            font=dict(color="#e0e0e0"),
            title=dict(
                text=f"{selected_gp} {selected_season} — Race Positions",
                font=dict(size=18, color="#e0e0e0"),
            ),
            xaxis=dict(
                title="Lap Number",
                range=[0, max_lap + 1],
                gridcolor="#2a2a3e",
                zerolinecolor="#2a2a3e",
            ),
            yaxis=dict(
                title="Position",
                autorange="reversed",
                tickvals=list(range(1, num_drivers + 1)),
                range=[num_drivers + 0.5, 0.5],
                gridcolor="#2a2a3e",
                zerolinecolor="#2a2a3e",
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0", size=11),
                x=1.01,
                y=1,
            ),
            height=640,
            margin=dict(l=60, r=160, t=80, b=130),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.12,
                    x=0.0,
                    xanchor="left",
                    buttons=[
                        dict(
                            label="▶  Play",
                            method="animate",
                            args=[
                                None,
                                dict(frame=dict(duration=150, redraw=True), fromcurrent=True, transition=dict(duration=0)),
                            ],
                        ),
                        dict(
                            label="⏸  Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0)),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=slider_steps,
                    x=0.0,
                    len=1.0,
                    y=-0.08,
                    currentvalue=dict(
                        prefix="Lap: ",
                        font=dict(color="#e0e0e0", size=14),
                        visible=True,
                        xanchor="center",
                    ),
                    font=dict(color="#888", size=9),
                    bgcolor="#1e1e2e",
                    bordercolor="#2a2a3e",
                    tickcolor="#555",
                )
            ],
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---- Final classification table ----
    st.subheader("Final Classification")
    final_lap_per_driver = pos_df.groupby("Driver")["LapNumber"].max().reset_index()
    final_pos = pos_df.merge(final_lap_per_driver, on=["Driver", "LapNumber"])
    final_pos = final_pos.sort_values("Position")[["Position", "Driver"]].copy()
    final_pos["Driver"] = final_pos["Driver"].map(lambda c: DRIVER_NAMES.get(c, c))
    final_pos = final_pos.reset_index(drop=True)

    # Split into 3 columns so it doesn't stretch to a single tall list
    n = len(final_pos)
    chunk = (n + 2) // 3
    col_a, col_b, col_c = st.columns(3)
    for col, slice_df in zip(
        [col_a, col_b, col_c],
        [final_pos.iloc[:chunk], final_pos.iloc[chunk : 2 * chunk], final_pos.iloc[2 * chunk :]],
    ):
        with col:
            st.dataframe(slice_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 5 — AI Race Analyst
# ---------------------------------------------------------------------------


def load_race_summary(season: int, gp_name: str) -> str:
    """Build a structured text summary of a race for the AI analyst.

    Args:
        season: Championship year.
        gp_name: Grand Prix name matching the GrandPrix column in the DB.

    Returns:
        Formatted multi-line string summarising stints, pit stops, and
        best compound performance.
    """
    laps = load_laps()
    race = laps[(laps["Season"] == season) & (laps["GrandPrix"] == gp_name)].copy()
    if race.empty:
        return f"No data found for {gp_name} {season}."

    race = compute_lap_delta(race)

    total_laps = int(race["LapNumber"].max())
    drivers_in_race = sorted(race["Driver"].unique())
    num_drivers = len(drivers_in_race)

    # Determine final position order if available
    if "Position" in race.columns:
        final_lap = race.groupby("Driver")["LapNumber"].max().reset_index()
        final_pos_df = race.merge(final_lap, on=["Driver", "LapNumber"])
        pos_order = final_pos_df.sort_values("Position")["Driver"].tolist()
        ordered_drivers = [d for d in pos_order if d in drivers_in_race]
        ordered_drivers += [d for d in drivers_in_race if d not in ordered_drivers]
    else:
        ordered_drivers = sorted(drivers_in_race)

    lines: list[str] = [
        f"Race: {gp_name} {season}",
        f"Total laps: {total_laps}  |  Drivers: {num_drivers}",
        "",
        "## Driver Stint Breakdown",
    ]

    for driver in ordered_drivers:
        driver_laps = race[race["Driver"] == driver].sort_values("LapNumber")
        pit_laps = detect_pit_laps(driver_laps)
        display_name = DRIVER_NAMES.get(driver, driver)

        # Split into stints at each pit lap
        stint_starts = [int(driver_laps["LapNumber"].min())]
        for pl in sorted(pit_laps):
            stint_starts.append(int(pl))
        stint_starts = sorted(set(stint_starts))

        stint_summaries: list[str] = []
        for i, start in enumerate(stint_starts):
            end = stint_starts[i + 1] - 1 if i + 1 < len(stint_starts) else total_laps
            stint = driver_laps[
                (driver_laps["LapNumber"] >= start) & (driver_laps["LapNumber"] <= end)
            ]
            if stint.empty:
                continue
            compound = stint["Compound"].mode().iloc[0] if not stint["Compound"].mode().empty else "UNKNOWN"
            n_laps = len(stint)
            avg_delta = stint["lap_time_delta"].mean()
            stint_summaries.append(
                f"    Stint {i + 1}: {compound} laps {start}–{end} ({n_laps} laps, avg delta {avg_delta:+.3f}s)"
            )

        pit_str = ", ".join(str(p) for p in sorted(pit_laps)) if pit_laps else "none"
        lines.append(f"\n{display_name} ({driver})  —  Pit laps: {pit_str}")
        lines.extend(stint_summaries)

    # Best average delta per compound
    lines += ["", "## Best Average Delta Per Compound"]
    for compound in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"):
        subset = race[race["Compound"] == compound]
        if subset.empty:
            continue
        best_driver = (
            subset.groupby("Driver")["lap_time_delta"]
            .mean()
            .idxmin()
        )
        best_delta = subset.groupby("Driver")["lap_time_delta"].mean().min()
        lines.append(
            f"  {compound}: {DRIVER_NAMES.get(best_driver, best_driver)} ({best_driver}) — avg {best_delta:+.3f}s"
        )

    return "\n".join(lines)


def call_race_analyst(summary: str, gp_name: str, season: int) -> str:
    """Send the race summary to the Groq API and return the analysis text.

    Args:
        summary: Structured race summary produced by load_race_summary().
        gp_name: Grand Prix name (used in the user prompt).
        season: Championship year (used in the user prompt).

    Returns:
        The model's analysis as a string.
    """
    load_dotenv()
    import os
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment / .env file.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an F1 race strategist and analyst. "
                    "Analyse race data and provide sharp, insightful commentary. "
                    "Use markdown formatting with headers and bold text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Analyse the {gp_name} {season} Formula 1 race using the data below. "
                    "Please cover:\n"
                    "1. **Overall race narrative** (2–3 sentences)\n"
                    "2. **Strategic battle** — who made the best tyre calls and why\n"
                    "3. **Standout performances** — fastest stints, biggest positions gained\n"
                    "4. **One tactical insight** a race strategist would find interesting\n\n"
                    f"{summary}"
                ),
            },
        ],
    )
    return response.choices[0].message.content


if page == "AI Race Analyst":
    st.title("AI Race Analyst")
    st.markdown(
        "Select a race and let an AI strategist break down the tactics, "
        "tyre calls, and standout performances."
    )

    laps = load_laps()

    col1, col2 = st.columns(2)
    with col1:
        available_seasons = sorted(laps["Season"].dropna().unique().astype(int), reverse=True)
        selected_season = st.selectbox("Season", available_seasons, format_func=str, key="ai_season")

    season_laps = laps[laps["Season"] == selected_season]
    with col2:
        gp_options = sorted(season_laps["GrandPrix"].dropna().unique())
        selected_gp = st.selectbox("Grand Prix", gp_options, key="ai_gp")

    import os
    load_dotenv()
    api_key_present = bool(os.getenv("GROQ_API_KEY"))

    if not api_key_present:
        st.error(
            "GROQ_API_KEY is not set. "
            "Add it to your `.env` file:  `GROQ_API_KEY=gsk_...`"
        )
        st.stop()

    analyse_clicked = st.button("Analyse Race", type="primary")
    regenerate_clicked = False

    if "ai_analysis" in st.session_state and st.session_state.get("ai_race_key") == (selected_season, selected_gp):
        st.markdown(
            f"<div style='background:#1e1e2e; border:1px solid #2a2a3e; border-radius:8px; padding:20px;'>"
            f"{st.session_state['ai_analysis']}"
            f"</div>",
            unsafe_allow_html=True,
        )
        regenerate_clicked = st.button("Regenerate", key="regenerate")

    if analyse_clicked or regenerate_clicked:
        with st.spinner("Analysing race data…"):
            try:
                summary = load_race_summary(selected_season, selected_gp)
                analysis = call_race_analyst(summary, selected_gp, selected_season)
                st.session_state["ai_analysis"] = analysis
                st.session_state["ai_race_key"] = (selected_season, selected_gp)
                st.markdown(
                    f"<div style='background:#1e1e2e; border:1px solid #2a2a3e; border-radius:8px; padding:20px;'>"
                    f"{analysis}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.button("Regenerate", key="regenerate_after")
            except Exception as exc:
                st.error(f"API call failed: {exc}")
