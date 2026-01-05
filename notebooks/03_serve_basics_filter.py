# %% read necessary table
from pathlib import Path

import numpy as np
import pandas as pd
from utils import norm_entropy, safe_div

input_dir = Path("../data/filtered")
serve = pd.read_parquet(input_dir / "serve_basics.parquet")

# %% filter
s = serve.copy()

wide = s.pivot_table(
    index=["match_id", "player"],
    columns="row",
    values=s.columns[3:],  # only the features
    aggfunc="sum",
)

wide.columns = [
    f"{m}_{r.lower() if r not in ['1', '2'] else r}" for (m, r) in wide.columns
]
wide = wide.reset_index()

# %% feature engineering
for r in ["total", "1", "2"]:
    pts = wide[f"pts_{r}"]
    pts_won = wide[f"pts_won_{r}"]

    # effectiveness
    wide[f"serve_win_pct_{r}"] = safe_div(pts_won, pts)

    # free points
    for m in ["aces", "unret", "forced_err"]:
        wide[f"{m}_rate_{r}"] = safe_div(wide[f"{m}_{r}"], pts)
    wide[f"free_point_rate_{r}"] = safe_div(
        wide[f"aces_{r}"] + wide[f"unret_{r}"] + wide[f"forced_err_{r}"],
        pts,
    )

    # short point aggression
    wide[f"quick_win_rate_{r}"] = safe_div(wide[f"pts_won_lte_3_shots_{r}"], pts)
    wide[f"quick_win_share_of_wins_{r}"] = safe_div(
        wide[f"pts_won_lte_3_shots_{r}"], pts_won
    )

    # serve placement profile
    for dname in ["wide", "body", "t"]:
        wide[f"{dname}_share_{r}"] = safe_div(wide[f"{dname}_{r}"], pts)

    # normalized entropy over direction: 0 (all one spot) -> 1 (evenly mixed)
    ps = np.vstack(
        [wide[f"wide_share_{r}"], wide[f"body_share_{r}"], wide[f"t_share_{r}"]]
    ).T
    ent = norm_entropy(ps)
    ent = np.where(np.isfinite(ps).all(axis=1), ent, np.nan)

    # placement "signature"
    wide[f"serve_dir_entropy_{r}"] = ent
    wide[f"serve_dir_bias_{r}"] = np.where(
        np.isfinite(ps).all(axis=1), np.max(ps, axis=1) - np.min(ps, axis=1), np.nan
    )
    wide[f"serve_dir_maxshare_{r}"] = np.where(
        np.isfinite(ps).all(axis=1), np.max(ps, axis=1), np.nan
    )

# cross-row comparisons (differences 1st vs 2nd placement)
wide["first_second_win_gap"] = wide["serve_win_pct_1"] - wide["serve_win_pct_2"]
wide["first_second_free_gap"] = wide["free_point_rate_1"] - wide["free_point_rate_2"]
wide["second_serve_share"] = safe_div(wide["pts_2"], wide["pts_total"])

for dname in ["wide", "body", "t"]:
    wide[f"{dname}_share_1_minus_2"] = (
        wide[f"{dname}_share_1"] - wide[f"{dname}_share_2"]
    )

keep = ["match_id", "player"]

player_serve_basics = (
    wide.groupby("player")
    .agg(
        serve_win_pct_1_mean=("serve_win_pct_1", "mean"),
        serve_win_pct_2_mean=("serve_win_pct_2", "mean"),
        first_second_win_gap_mean=("first_second_win_gap", "mean"),
        free_point_rate_1_mean=("free_point_rate_1", "mean"),
        quick_win_rate_1_mean=("quick_win_rate_1", "mean"),
        quick_win_rate_2_mean=("quick_win_rate_2", "mean"),
        wide_share_1_mean=("wide_share_1", "mean"),
        body_share_1_mean=("body_share_1", "mean"),
        t_share_1_mean=("t_share_1", "mean"),
        serve_dir_entropy_1_mean=("serve_dir_entropy_1", "mean"),
    )
    .reset_index()
)

# %% output

out_dir = Path("../data/processed/features")
player_serve_basics.to_parquet(out_dir / "serve_basics.parquet", index=False)
