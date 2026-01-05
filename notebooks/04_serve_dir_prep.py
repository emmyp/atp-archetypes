# %% read necessary table
from pathlib import Path

import numpy as np
import pandas as pd

input_dir = Path("../data/filtered")
servedir = pd.read_parquet(input_dir / "serve_direction.parquet")

# %% engineering from features where row == "Total"
sd = servedir.copy()
sd_total = sd[sd["row"] == "Total"].copy()

DIRS_DEUCE = ["deuce_wide", "deuce_middle", "deuce_t"]
DIRS_AD = ["ad_wide", "ad_middle", "ad_t"]
ERRS = ["err_net", "err_wide", "err_deep", "err_wide_deep", "err_foot", "err_unknown"]

deuce_sum = sd_total[DIRS_DEUCE].sum(axis=1)
ad_sum = sd_total[DIRS_AD].sum(axis=1)
p_deuce = sd_total[DIRS_DEUCE].div(deuce_sum.replace(0, np.nan), axis=0).to_numpy()
p_ad = sd_total[DIRS_AD].div(ad_sum.replace(0, np.nan), axis=0).to_numpy()

# t-usage difference by side
sd_total["t_diff"] = p_deuce[:, 2] - p_ad[:, 2]  # index 2 = *_t

# %% engineering from features where row == "1"
sd_1 = sd[sd["row"] == "1"].copy()

deuce_sum = sd_1[DIRS_DEUCE].sum(axis=1)
ad_sum = sd_1[DIRS_AD].sum(axis=1)

p_deuce = sd_1[DIRS_DEUCE].div(deuce_sum.replace(0, np.nan), axis=0).to_numpy()
p_ad = sd_1[DIRS_AD].div(ad_sum.replace(0, np.nan), axis=0).to_numpy()

# "same placement everywhere" vs "very diff patterns on deuce vs ad" (normalized)
sd_1["side_asym_1"] = 0.5 * np.sum(np.abs(p_deuce - p_ad), axis=1)

# %% engineering from features where row == "2"
sd_2 = sd[sd["row"] == "2"].copy()

deuce_sum = sd_2[DIRS_DEUCE].sum(axis=1)
ad_sum = sd_2[DIRS_AD].sum(axis=1)

p_deuce = sd_2[DIRS_DEUCE].div(deuce_sum.replace(0, np.nan), axis=0).to_numpy()
p_ad = sd_2[DIRS_AD].div(ad_sum.replace(0, np.nan), axis=0).to_numpy()

# "same placement everywhere" vs "very diff patterns on deuce vs ad" (normalized)
sd_2["side_asym_2"] = 0.5 * np.sum(np.abs(p_deuce - p_ad), axis=1)

# double fault miss profile
err_sum = sd_2[ERRS].sum(axis=1).replace(0, np.nan)
sd_2["df_net_share"] = sd_2["err_net"] / err_sum
sd_2["df_long_share"] = (sd_2["err_deep"] + sd_2["err_wide_deep"]) / err_sum

# %% aggregate to player level
sd_agg = sd_total.merge(sd_1, on="player", how="left").merge(
    sd_2, on="player", how="left"
)

player_serve_dir = (
    sd_agg.groupby("player")
    .agg(
        side_asym_1_mean=("side_asym_1", "mean"),
        side_asym_2_mean=("side_asym_2", "mean"),
        t_diff_mean=("t_diff", "mean"),
        df_net_share_mean=("df_net_share", "mean"),
        df_long_share_mean=("df_long_share", "mean"),
    )
    .reset_index()
)

# %% output
out_dir = Path("../data/processed/features")
player_serve_dir.to_parquet(out_dir / "serve_direction.parquet", index=False)
