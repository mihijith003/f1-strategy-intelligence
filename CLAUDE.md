# CLAUDE.md

Guidance for Claude Code (and other contributors) working in this repository.

## Project Goal

F1 Race Strategy Intelligence Dashboard — a data science portfolio project that
ingests historical and live Formula 1 timing/telemetry data, processes it into
race-strategy features (pit stops, tyre degradation, stint pace, undercut/overcut
windows), trains models to surface strategic insights, and presents them through
an interactive dashboard.

## Tech Stack

- **FastF1** — F1 timing, telemetry, and session data ingestion
- **Pandas / NumPy** — data wrangling and feature engineering
- **scikit-learn** — predictive models (e.g. tyre degradation, pit-stop/lap-time forecasting)
- **Streamlit** — interactive dashboard front end
- **Plotly** — charts and visualizations embedded in the dashboard
- **SQLite** (via SQLAlchemy) — lightweight persistence for processed data

## Folder Structure

```
src/
  ingestion/    # FastF1/API data pulls, raw data acquisition
  processing/   # Cleaning, transformation, feature engineering
  models/       # Training, evaluation, and inference code
  dashboard/    # Streamlit app and visualization components
data/
  raw/          # Raw/cached data (FastF1 cache) — gitignored
  processed/    # Cleaned, feature-engineered datasets — gitignored
notebooks/      # Exploratory analysis and prototyping
tests/          # pytest test suite
```

## Branching Strategy

- `main` — stable, always deployable
- `develop` — integration branch for completed features
- `feature/<short-description>` — individual feature branches, branched from
  `develop` and merged back via PR

## Coding Conventions

- Use **type hints** on all function signatures
- Use **docstrings** (summary + Args/Returns) for all public modules,
  classes, and functions
- Write tests with **pytest**; place them under `tests/`, mirroring the
  `src/` package structure
- Keep ingestion, processing, modeling, and dashboard concerns separated by
  module boundary — avoid cross-importing dashboard code into processing/models
