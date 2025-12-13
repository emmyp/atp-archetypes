# %% load and filter
from pathlib import Path

import numpy as np
import pandas as pd

raw_dir = Path("../data/raw")

years = range(2018, 2025)
df = pd.concat(
    [pd.read_csv(raw_dir / f"atp_matches_{year}.csv") for year in years],
    ignore_index=True,
)

# start analysis for hard surface  only
df = df[df["surface"] == "Hard"]
decades = ["10s", "20s"]
rankings = pd.concat(
    [pd.read_csv(raw_dir / f"atp_rankings_{decade}.csv") for decade in decades],
    ignore_index=True,
)

# %% turn each match into two player-match rows
winner_rows = df.assign(
    match_id=df["tourney_name"] + "_" + df["match_num"].astype(str),
    tourney_date=df["tourney_date"],
    player_name=df["winner_name"],
    player_id=df["winner_id"],
    opponent_name=df["loser_name"],
    opponent_id=df["loser_id"],
    win=1,
    aces=df["w_ace"],
    double_faults=df["w_df"],
    service_points=df["w_svpt"],
    first_serve_in=df["w_1stIn"],
    first_serve_won=df["w_1stWon"],
    second_serve_won=df["w_2ndWon"],
    bp_faced_per_match=df["w_bpFaced"],
    bp_saved=df["w_bpSaved"],
    return_points=df["l_svpt"] - df["l_1stWon"] - df["l_2ndWon"],
)

loser_rows = df.assign(
    match_id=df["tourney_name"] + "_" + df["match_num"].astype(str),
    tourney_date=df["tourney_date"],
    player_name=df["loser_name"],
    player_id=df["loser_id"],
    opponent_name=df["loser_name"],
    opponent_id=df["winner_id"],
    win=0,
    aces=df["l_ace"],
    double_faults=df["l_df"],
    service_points=df["l_svpt"],
    first_serve_in=df["l_1stIn"],
    first_serve_won=df["l_1stWon"],
    second_serve_won=df["l_2ndWon"],
    bp_faced_per_match=df["l_bpFaced"],
    bp_saved=df["l_bpSaved"],
    return_points=df["w_svpt"] - df["w_1stWon"] - df["w_2ndWon"],
)

player_matches = (
    pd.concat([winner_rows, loser_rows], ignore_index=True)
    .assign(
        aces_per_svpt=lambda x: x["aces"] / x["service_points"],
        df_rate=lambda x: x["double_faults"] / x["service_points"],
        first_serve_in_pct=lambda x: x["first_serve_in"] / x["service_points"],
        first_serve_won_pct=lambda x: x["first_serve_won"] / x["first_serve_in"],
        second_serve_won_pct=lambda x: x["second_serve_won"]
        / (x["service_points"] - x["first_serve_in"]),
        bp_saved_pct=lambda x: x["bp_saved"] / x["bp_faced_per_match"],
    )
    .replace([np.inf, -np.inf], np.nan)  # guard against division by zero
    .loc[
        :,
        [
            "match_id",
            "tourney_date",
            "player_name",
            "player_id",
            "opponent_name",
            "opponent_id",
            "win",
            "aces_per_svpt",
            "df_rate",
            "first_serve_won",
            "first_serve_in_pct",
            "first_serve_won_pct",
            "second_serve_won",
            "second_serve_won_pct",
            "bp_faced_per_match",
            "bp_saved",
            "bp_saved_pct",
            "service_points",
            "return_points",
        ],
    ]
)

player_matches = player_matches[player_matches["service_points"] > 0].copy()

# %% join match and rankings data

player_matches["tourney_date"] = pd.to_datetime(
    player_matches["tourney_date"], format="%Y%m%d"
)
player_matches = player_matches.sort_values(["tourney_date", "player_id"]).reset_index(
    drop=True
)

rankings["ranking_date"] = pd.to_datetime(rankings["ranking_date"], format="%Y%m%d")
rankings = rankings.sort_values(["ranking_date", "player"]).reset_index(drop=True)
rankings.rename(columns={"player": "player_id"}, inplace=True)

out = pd.merge_asof(
    player_matches,
    rankings,
    left_on="tourney_date",
    right_on="ranking_date",
    by="player_id",
    direction="backward",
)

# %% add return stats from opponent's serve data

pair_col = ["match_id"]  # unique identifier for each match
opp = out[
    pair_col
    + [
        "player_id",
        "service_points",
        "first_serve_won",
        "second_serve_won",
        "bp_faced_per_match",
        "bp_saved",
    ]
]
opp.columns = pair_col + [
    "opponent_id",
    "opponent_svpt",
    "opponent_first_serve_won",
    "opponent_second_serve_won",
    "opponent_bp_faced_per_match",
    "opponent_bp_saved",
]

out = out.merge(opp, on=pair_col + ["opponent_id"], how="left")

out["return_points"] = out["opponent_svpt"]
out["return_points_won"] = out["opponent_svpt"] - (
    out["opponent_first_serve_won"] + out["opponent_second_serve_won"]
)
out["return_points_won_pct"] = out["return_points_won"] / out["return_points"]

out["bp_chances"] = out["opponent_bp_faced_per_match"]
out["bp_converted"] = out["opponent_bp_faced_per_match"] - out["opponent_bp_saved"]
out["bp_converted_pct"] = out["bp_converted"] / out["bp_chances"]

# %% aggregate to the player level
features = [
    "aces_per_svpt",
    "df_rate",
    "first_serve_in_pct",
    "first_serve_won_pct",
    "second_serve_won_pct",
    "bp_faced_per_match",
    "bp_saved_pct",
    "return_points_won_pct",
    "bp_converted_pct",
    "win",
    "rank",
]

player_stats = out.groupby("player_id")[features].mean()
player_stats["matches_played"] = out.groupby("player_id").size()

# keep players with enough sample size
player_stats = player_stats[player_stats["matches_played"] >= 40].copy()

player_stats.rename(columns={"win": "win_rate", "rank": "avg_rank"}, inplace=True)
player_stats.reset_index(inplace=True)
# %% normalize style within ability bands (we don't want clusters to just be good vs. better players)


def zscore(group):
    return (group - group.mean()) / group.std()


player_stats["rank_band"] = pd.qcut(
    player_stats["avg_rank"], q=4, labels=["top", "high", "mid", "low"]
)

features.remove("rank")
features.remove("win")
for col in features:
    player_stats[col + "_zscore"] = player_stats.groupby("rank_band", observed=True)[
        col
    ].transform(zscore)

# %% persist processed data
processed_dir = Path("../data/processed")
player_stats.to_csv(processed_dir / "player_stats_norm_hard_2018_2024.csv", index=False)
