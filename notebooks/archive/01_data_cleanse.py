# %% load and filter
from pathlib import Path

import numpy as np
import pandas as pd

raw_dir = Path("../../data/raw")

years = range(2018, 2025)
df = pd.concat(
    [pd.read_csv(raw_dir / f"atp_matches_{year}.csv") for year in years],
    ignore_index=True,
)

# start analysis for hard surface  only
df = df[df["surface"] == "Hard"]

# %% turn each match into two player-match rows
winner_rows = df.assign(
    match_id=df["tourney_name"] + "_" + df["match_num"].astype(str),
    player_name=df["winner_name"],
    player_id=df["winner_id"],
    opponent_name=df["loser_name"],
    opponent_id=df["loser_id"],
    is_winner=1,
    aces=df["w_ace"],
    double_faults=df["w_df"],
    total_service_points=df["w_svpt"],
    first_serve_in=df["w_1stIn"],
    first_serve_won=df["w_1stWon"],
    second_serve_in=df["w_svpt"] - df["w_1stIn"],
    second_serve_won=df["w_2ndWon"],
    bp_faced=df["w_bpFaced"],
    bp_saved=df["w_bpSaved"],
    total_return_points=df["l_svpt"] - df["l_1stWon"] - df["l_2ndWon"],
)

loser_rows = df.assign(
    match_id=df["tourney_name"] + "_" + df["match_num"].astype(str),
    player_name=df["loser_name"],
    player_id=df["loser_id"],
    opponent_name=df["loser_name"],
    opponent_id=df["winner_id"],
    is_winner=0,
    aces=df["l_ace"],
    double_faults=df["l_df"],
    total_service_points=df["l_svpt"],
    first_serve_in=df["l_1stIn"],
    first_serve_won=df["l_1stWon"],
    second_serve_in=df["l_svpt"] - df["l_1stIn"],
    second_serve_won=df["l_2ndWon"],
    bp_faced=df["l_bpFaced"],
    bp_saved=df["l_bpSaved"],
    total_return_points=df["w_svpt"] - df["w_1stWon"] - df["w_2ndWon"],
)

player_matches = (
    pd.concat([winner_rows, loser_rows], ignore_index=True)
    .assign(
        first_serve_in_pct=lambda x: x["first_serve_in"] / x["total_service_points"],
        second_serve_in_pct=lambda x: x["second_serve_in"] / x["total_service_points"],
        first_serve_won_pct=lambda x: x["first_serve_won"] / x["first_serve_in"],
        second_serve_won_pct=lambda x: x["second_serve_won"] / x["second_serve_in"],
        bp_saved_pct=lambda x: x["bp_saved"] / x["bp_faced"],
    )
    .replace([np.inf, -np.inf], np.nan)  # guard against division by zero
    .loc[
        :,
        [
            "match_id",
            "player_name",
            "player_id",
            "opponent_name",
            "opponent_id",
            "is_winner",
            "aces",
            "double_faults",
            "first_serve_in_pct",
            "second_serve_in_pct",
            "first_serve_won_pct",
            "second_serve_won_pct",
            "bp_faced",
            "bp_saved_pct",
            "total_service_points",
            "total_return_points",
        ],
    ]
)

# %% persist processed data
processed_dir = Path("../../data/processed")
player_matches.to_csv(processed_dir / "atp_matches_hard_2018_2024.csv", index=False)
