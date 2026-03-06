import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from utils import wavg, winsorize_features

FILE_NAME = "serve_direction.parquet"
KEYS = ["match_id", "player"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build player-level features from serve_direction.parquet"
    )
    p.add_argument("--input-dir", type=Path, default=Path("../data/filtered"))
    p.add_argument("--out-dir", type=Path, default=Path("../data/processed/features"))

    p.add_argument(
        "--winsorize",
        action="store_true",
        help="clip numeric feature columns to percentile bounds",
    )
    p.add_argument(
        "--winsor-p-low",
        type=float,
        default=0.005,
        help="lower percentile bound (e.g., 0.005)",
    )
    p.add_argument(
        "--winsor-p-high",
        type=float,
        default=0.995,
        help="upper percentile bound (e.g., 0.995)",
    )
    p.add_argument(
        "--winsor-min-matches",
        type=int,
        default=8,
        help="cohort threshold used to compute bounds",
    )

    return p.parse_args()

def load_serve_direction(input_dir: Path) -> pd.DataFrame:
    path = input_dir / FILE_NAME
    return pd.read_parquet(path)

def build_player_serve_direction(df: pd.DataFrame) -> pd.DataFrame:
    # how many matches do we have per player?
    match_counts = (
        df[df["row"] == "Total"]
        .groupby("player", as_index=False)["match_id"]
        .nunique()
        .rename(columns={"match_id": "n_matches"})
    )

    DIRS_DEUCE = ["deuce_wide", "deuce_middle", "deuce_t"]
    DIRS_AD = ["ad_wide", "ad_middle", "ad_t"]
    ERRS = ["err_net", "err_wide", "err_deep", "err_wide_deep", "err_foot", "err_unknown"]

    # --- engineering from features where row == "Total" ---
    sd_total = df[df["row"] == "Total"].copy()
    deuce_sum_tot = sd_total[DIRS_DEUCE].sum(axis=1)
    ad_sum_tot = sd_total[DIRS_AD].sum(axis=1)

    sd_total["pts_tot"] = deuce_sum_tot + ad_sum_tot

    # --- Engineering from features where row == "1" ---
    sd_1 = df[df["row"] == "1"].copy()
    deuce_sum_1 = sd_1[DIRS_DEUCE].sum(axis=1)
    ad_sum_1 = sd_1[DIRS_AD].sum(axis=1)

    p_deuce_1 = sd_1[DIRS_DEUCE].div(deuce_sum_1.replace(0, np.nan), axis=0).to_numpy()
    p_ad_1 = sd_1[DIRS_AD].div(ad_sum_1.replace(0, np.nan), axis=0).to_numpy()

    # "same placement everywhere" vs "very diff patterns on deuce vs ad" (normalized)
    sd_1["side_asym_1"] = 0.5 * np.sum(np.abs(p_deuce_1 - p_ad_1), axis=1)
    sd_1["pts_1"] = deuce_sum_1 + ad_sum_1

    # body serve share (1st serve)
    sd_1["body_share_1"] = (sd_1["deuce_middle"] + sd_1["ad_middle"]) / sd_1["pts_1"].replace(0, np.nan)

    # T-usage difference by side (1st serve)
    sd_1["t_diff_1"] = p_deuce_1[:, 2] - p_ad_1[:, 2]

    # --- engineering from features where row == "2" ---
    sd_2 = df[df["row"] == "2"].copy()
    deuce_sum_2 = sd_2[DIRS_DEUCE].sum(axis=1)
    ad_sum_2 = sd_2[DIRS_AD].sum(axis=1)

    p_deuce_2 = sd_2[DIRS_DEUCE].div(deuce_sum_2.replace(0, np.nan), axis=0).to_numpy()
    p_ad_2 = sd_2[DIRS_AD].div(ad_sum_2.replace(0, np.nan), axis=0).to_numpy()

    # "same placement everywhere" vs "very diff patterns on deuce vs ad" (normalized)
    sd_2["side_asym_2"] = 0.5 * np.sum(np.abs(p_deuce_2 - p_ad_2), axis=1)
    sd_2["pts_2"] = deuce_sum_2 + ad_sum_2

    # body serve share (2nd serve)
    sd_2["body_share_2"] = (sd_2["deuce_middle"] + sd_2["ad_middle"]) / sd_2["pts_2"].replace(0, np.nan)

    # T-usage difference by side (2nd serve)
    sd_2["t_diff_2"] = p_deuce_2[:, 2] - p_ad_2[:, 2]

    # double fault miss profile
    err_sum = sd_2[ERRS].sum(axis=1).replace(0, np.nan)
    sd_2["df_net_share"] = sd_2["err_net"] / err_sum
    sd_2["err_sum"] = err_sum

    # --- aggregate to player level ---
    sd_agg = sd_total[["match_id", "player", "pts_tot"]].merge(
        sd_1[["match_id", "player", "side_asym_1", "body_share_1", "t_diff_1", "pts_1"]], on=KEYS, how="left"
    ).merge(
        sd_2[["match_id", "player", "side_asym_2", "body_share_2", "t_diff_2", "df_net_share", "pts_2", "err_sum"]], on=KEYS, how="left"
    )

    def agg_player_features(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "side_asym_1_mean": wavg(g, "side_asym_1", "pts_1"),
            "side_asym_2_mean": wavg(g, "side_asym_2", "pts_2"),
            "t_diff_1_mean": wavg(g, "t_diff_1", "pts_1"),
            "t_diff_2_mean": wavg(g, "t_diff_2", "pts_2"),
            # "body_share_1_mean": wavg(g, "body_share_1", "pts_1"),
            "body_share_2_mean": wavg(g, "body_share_2", "pts_2"),
            "df_net_share_mean": wavg(g, "df_net_share", "err_sum"),
        })

    player_serve_dir = (
        sd_agg.groupby("player")
        .apply(agg_player_features, include_groups=False)
        .reset_index()
    )

    out = player_serve_dir.merge(match_counts, on="player", how="left")
    cols = ["player", "n_matches"] + [c for c in out.columns if c not in ["player", "n_matches"]]

    return out[cols].copy()

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_serve_direction(args.input_dir)
    out = build_player_serve_direction(df)

    if args.winsorize:
        out = winsorize_features(
            out,
            p_low=args.winsor_p_low,
            p_high=args.winsor_p_high,
            match_col="n_matches",
            min_matches=args.winsor_min_matches,
            exclude_cols=(
                "player",
                "n_matches",
            ),
        )

    out.to_parquet(args.out_dir / FILE_NAME, index=False)


if __name__ == "__main__":
    main()
