# %% read tables
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.core.internals.blocks import IgnoreRaise

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

# %% rally table gets treated differently than the rest

rally_filtered = rally[rally["match_id"].isin(match_ids)].copy()

MAIN_ROWS = ["Total", "1-3", "4-6", "7-9", "10"]
rally_filtered = rally_filtered[rally_filtered["row"].isin(MAIN_ROWS)].copy()

# long format for player 1 (server)
p1 = rally_filtered[["match_id", "row", "pts", "pl1_won"]].copy()
p1["player_slot"] = 1
p1.rename(columns={"pl1_won": "pts_won"}, inplace=True)

# long format for player 2 (returner)
p2 = rally_filtered[["match_id", "row", "pts", "pl2_won"]].copy()
p2["player_slot"] = 2
p2.rename(columns={"pl2_won": "pts_won"}, inplace=True)

rl_long = pd.concat([p1, p2], ignore_index=True)
rl_long = rl_long.merge(matches, on="match_id", how="left")
rl_long["player_name"] = np.where(
    rl_long["player_slot"] == 1,
    rl_long["Player 1"],
    rl_long["Player 2"],
)
rl_long = rl_long[["match_id", "player_name", "row", "pts", "pts_won"]].copy()

# total points per match & player
pts_by_bucket = rl_long[rl_long["row"] != "Total"].copy()
totals = (
    pts_by_bucket.groupby(["match_id", "player_name"])["pts"]
    .sum()
    .rename("total_points")
    .reset_index()
)
pts_by_bucket = pts_by_bucket.merge(totals, on=["match_id", "player_name"], how="left")
pts_by_bucket["rally_share"] = pts_by_bucket["pts"] / pts_by_bucket["total_points"]

dist = pts_by_bucket.pivot_table(
    index=["match_id", "player_name"],
    columns=["row"],
    values="rally_share",
    fill_value=0.0,
)
dist.columns = [
    f"rally_{c.replace('10', '10plus').replace('-', 'to')}_pct" for c in dist.columns
]
dist = dist.reset_index()

# win % in long rallies (7+ shots)
long_rows = rl_long[rl_long["row"].isin(["7-9", "10"])].copy()
long_agg = (
    long_rows.groupby(["match_id", "player_name"])
    .agg(
        long_points=("pts", "sum"),
        long_points_won=("pts_won", "sum"),
    )
    .reset_index()
)
long_agg["long_rally_win_pct"] = long_agg["long_points_won"] / long_agg["long_points"]

# combine into match x player rally stats
match_player_rally = dist.merge(
    long_agg[["match_id", "player_name", "long_rally_win_pct"]],
    on=["match_id", "player_name"],
    how="left",
)

# aggregate to player-level
player_rally = match_player_rally.groupby("player_name").agg(
    rally_1to3_pct_mean=("rally_1to3_pct", "mean"),
    rally_4to6_pct_mean=("rally_4to6_pct", "mean"),
    rally_7to9_pct_mean=("rally_7to9_pct", "mean"),
    rally_10plus_pct_mean=("rally_10plus_pct", "mean"),
    long_rally_win_pct_mean=("long_rally_win_pct", "mean"),
    matches_charted=("match_id", "nunique"),
)

# %% build match x player style table
match_ids = matches_hard["match_id"].unique()

stats_tables = (
    oview,
    serve,
    servedir,
    ret,
    retdep,
    shotdir,
    shotty,
    netpts,
    snv,
)
for df in stats_tables:
    # subset to only matches we care about
    df.query("match_id in @match_ids", inplace=True)


# %%
rally.head(10)
