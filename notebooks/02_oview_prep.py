# %% read necessary table
from pathlib import Path

import pandas as pd
from utils import safe_div

input_dir = Path("../data/filtered")
oview = pd.read_parquet(input_dir / "overview.parquet")
# %% filter
o = oview.copy()

# match totals: one row per match_id x player
o_agg = o[o["set"] == "Total"].copy()

# %% feature engineering

# core counts
o_agg["serve_pts_won"] = o_agg["first_won"] + o_agg["second_won"]
o_agg["total_pts"] = o_agg["serve_pts"] + o_agg["return_pts"]
o_agg["total_pts_won"] = o_agg["serve_pts_won"] + o_agg["return_pts_won"]

# serve stats
o_agg["ace_rate"] = safe_div(o_agg["aces"], o_agg["serve_pts"])
o_agg["df_rate"] = safe_div(o_agg["dfs"], o_agg["serve_pts"])
o_agg["first_in_pct"] = safe_div(o_agg["first_in"], o_agg["serve_pts"])
o_agg["first_win_pct"] = safe_div(o_agg["first_won"], o_agg["first_in"])

o_agg["second_win_pct"] = safe_div(o_agg["second_won"], o_agg["second_in"])

o_agg["serve_pts_won_pct"] = safe_div(o_agg["serve_pts_won"], o_agg["serve_pts"])

# return / overall stats
o_agg["return_pts_won_pct"] = safe_div(o_agg["return_pts_won"], o_agg["return_pts"])
o_agg["total_pts_won_pct"] = safe_div(o_agg["total_pts_won"], o_agg["total_pts"])

# pressure (bk_pts = break points faced)
o_agg["bp_save_pct"] = safe_div(o_agg["bp_saved"], o_agg["bk_pts"])
o_agg["bk_pts_per_return_pt"] = safe_div(
    o_agg["bk_pts"], o_agg["return_pts"]
)  # pressure frequency proxy (how often they face break points)

# aggression / error balance
o_agg["winners_per_100"] = 100 * safe_div(o_agg["winners"], o_agg["total_pts"])
o_agg["unforced_per_100"] = 100 * safe_div(o_agg["unforced"], o_agg["total_pts"])
o_agg["aggression_index"] = safe_div(
    o_agg["winners"], (o_agg["winners"] + o_agg["unforced"])
)
o_agg["winner_to_unforced"] = safe_div(o_agg["winners"], o_agg["unforced"])

# fh/bh mix for winners and unforced errors
o_agg["fh_winner_share"] = safe_div(o_agg["winners_fh"], o_agg["winners"])
o_agg["bh_winner_share"] = safe_div(o_agg["winners_bh"], o_agg["winners"])
o_agg["fh_ue_share"] = safe_div(o_agg["unforced_fh"], o_agg["unforced"])
o_agg["bh_ue_share"] = safe_div(o_agg["unforced_bh"], o_agg["unforced"])

# classic "dominance ratio"
o_agg["serve_pts_lost_pct"] = 1 - o_agg["serve_pts_won_pct"]
o_agg["dominance_ratio"] = safe_div(
    o_agg["return_pts_won_pct"], o_agg["serve_pts_lost_pct"]
)

# aggregate to player-level
player_overview = (
    o_agg.groupby("player")
    .agg(
        ace_rate_mean=("ace_rate", "mean"),
        df_rate_mean=("df_rate", "mean"),
        first_in_pct_mean=("first_in_pct", "mean"),
        first_win_pct_mean=("first_win_pct", "mean"),
        second_win_pct_mean=("second_win_pct", "mean"),
        serve_pts_won_pct_mean=("serve_pts_won_pct", "mean"),
        return_pts_won_pct_mean=("return_pts_won_pct", "mean"),
        total_pts_won_pct_mean=("total_pts_won_pct", "mean"),
        dominance_ratio_mean=("dominance_ratio", "mean"),
        bp_save_pct_mean=("bp_save_pct", "mean"),
        bk_pts_per_return_pt_mean=("bk_pts_per_return_pt", "mean"),
        winners_per_100_mean=("winners_per_100", "mean"),
        unforced_per_100_mean=("unforced_per_100", "mean"),
        aggression_index_mean=("aggression_index", "mean"),
        winner_to_unforced_mean=("winner_to_unforced", "mean"),
        fh_winner_share_mean=("fh_winner_share", "mean"),
        bh_winner_share_mean=("bh_winner_share", "mean"),
        fh_ue_share_mean=("fh_ue_share", "mean"),
        bh_ue_share_mean=("bh_ue_share", "mean"),
        total_pts_sum=("total_pts", "sum"),
    )
    .reset_index()
)
# %% output
out_dir = Path("../data/processed/features")
player_overview.to_parquet(out_dir / "overview.parquet", index=False)
