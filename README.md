# Tennis Player Archetypes and Style

## Goal
Map out playing-style archetypes on the ATP Tour using public match data. Phase 1 (current repo state) builds a hard-court baseline from 2018–2024 matches, normalizes for player strength, and runs unsupervised clustering to highlight contrasting styles.

## Data
- `data/raw/atp_matches_*.csv`: match-level results from Jeff Sackmann (tennis-atp-tour).
- `data/raw/atp_rankings_*.csv`: weekly rankings joined to matches to separate style from ability. Also from Jeff Sackmann (tennis-atp-tour).
- `data/raw/charting-m-*.csv`: point-by-point charting data (JeffSackmann/tennis_MatchChartingProject). Not yet integrated; earmarked for richer shot/serve patterns in the next phase.
- Processed outputs live in `data/processed/` (see below).

## Phase-1 pipeline
All code lives in `notebooks/archive/` (kept as flat .py exports from notebooks).

Visuals from early clustering experiments are in `visuals/`.

## How to reproduce
1. Install deps: `pip install pandas numpy scikit-learn matplotlib`.
2. From `notebooks/archive/`, run the scripts in numerical order (or open them as notebooks).

## Next steps (planned)
- Fold in the charting datasets (serve/return direction, rally/shot types) via `notebooks/06_data_prep_w_mcp.py` to get pattern-level features.
- Persist clustering outputs (labels + centroids) and add brief archetype summaries.
- Expand beyond hard courts and revisit normalization (per-surface, era-aware).
