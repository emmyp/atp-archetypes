# %% read tables
from pathlib import Path

import pandas as pd

raw_dir = Path("../data/raw")

matches = pd.read_csv(raw_dir / "charting-m-matches.csv")
oview = pd.read_csv(raw_dir / "charting-m-stats-Overview.csv")
serve = pd.read_csv(raw_dir / "charting-m-stats-ServeBasics.csv")
servedir = pd.read_csv(raw_dir / "charting-m-stats-ServeDirection.csv")
ret = pd.read_csv(raw_dir / "charting-m-stats-ReturnOutcomes.csv")
retdep = pd.read_csv(raw_dir / "charting-m-stats-ReturnDepth.csv")
shotdir = pd.read_csv(raw_dir / "charting-m-stats-ShotDirection.csv")
shotty = pd.read_csv(raw_dir / "charting-m-stats-ShotTypes.csv")
rally = pd.read_csv(raw_dir / "charting-m-stats-Rally.csv")
netpts = pd.read_csv(raw_dir / "charting-m-stats-NetPoints.csv")
snv = pd.read_csv(raw_dir / "charting-m-stats-SnV.csv")

# %% ATP - hard-court matches (2018-2024)

matches["Date"] = pd.to_datetime(matches["Date"], format="%Y%m%d", errors="coerce")
matches = matches.dropna(subset=["Date"])

mask = (
    (matches["Surface"] == "Hard")
    & (matches["Date"].dt.year >= 2018)
    & (matches["Date"].dt.year <= 2024)
)
matches_hard = matches.loc[mask].copy()

cols = ["match_id", "Surface", "Date"]
matches_hard = matches_hard[cols].copy()

# %% standardize tables for match x player style df
match_ids = matches_hard["match_id"].unique()

# filter tables
tables = {
    "overview": oview,
    "rally": rally,
    "serve_basics": serve,
    "serve_direction": servedir,
    "return_outcomes": ret,
    "return_depth": retdep,
    "shot_direction": shotdir,
    "shot_types": shotty,
    "net_points": netpts,
    "snv": snv,
}

out_dir = Path("../data/filtered")

for name, df in tables.items():
    filtered = df[df["match_id"].isin(set(match_ids))].copy()

    filtered.to_parquet(out_dir / f"{name}.parquet", index=False)
