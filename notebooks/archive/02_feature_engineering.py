# %% load and filter
from pathlib import Path

import pandas as pd

processed_dir = Path("../../data/processed")
pm = pd.read_csv(processed_dir / "atp_matches_hard_2018_2024.csv")

# filter to players with at least 40 matches
player_counts = pm["player_id"].value_counts().reset_index()
eligible_players = player_counts[player_counts["count"] >= 40]["player_id"].tolist()

pm = pm[pm["player_id"].isin(eligible_players)]

# %% per-player averages

# serve aggressiveness (additional features)
pm["aces_per_svpt"] = pm["aces"] / pm["total_service_points"]
pm["df_rate"] = pm["double_faults"] / pm["total_service_points"]

agg = (
    pm.groupby("player_name")
    .agg(
        num_of_matches=("match_id", "size"),
        win_rate=("is_winner", "mean"),
        aces_per_svpt=("aces_per_svpt", "mean"),
        df_rate=("df_rate", "mean"),
        first_serve_in_pct=("first_serve_in_pct", "mean"),
        first_serve_won_pct=("first_serve_won_pct", "mean"),
        second_serve_in_pct=("second_serve_in_pct", "mean"),
        second_serve_won_pct=("second_serve_won_pct", "mean"),
        bp_faced_per_match=("bp_faced", "mean"),
        bp_saved_pct=("bp_saved_pct", "mean"),
    )
    .reset_index()
)

# %% persist processed data
agg.to_csv(processed_dir / "player_stats_hard_2018_2024.csv", index=False)
