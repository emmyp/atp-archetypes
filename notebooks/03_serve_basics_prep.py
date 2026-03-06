import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from utils import norm_entropy, safe_div, wavg, winsorize_features

FILE_NAME = "serve_basics.parquet"
KEYS = ["match_id", "player"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build player-level features from serve_basics.parquet"
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


def load_serve_basics(input_dir: Path) -> pd.DataFrame:
    path = input_dir / FILE_NAME
    return pd.read_parquet(path)


def build_player_serve_basics(df: pd.DataFrame) -> pd.DataFrame:
    # how many matches do we have per player?
    match_counts = (
        df.groupby("player", as_index=False)["match_id"]
        .nunique()
        .rename(columns={"match_id": "n_matches"})
    )

    wide = df.pivot_table(
        index=KEYS,
        columns="row",
        values=df.columns[3:],  # only the features
        aggfunc="sum",
    )

    wide.columns = [
        f"{m}_{r.lower() if r not in ['1', '2'] else r}" for (m, r) in wide.columns
    ]
    wide = wide.reset_index()

    # %% feature engineering per match
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

    # aggregate up to the player level using weighted averages
    def agg_player_features(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            # "serve_win_pct_1_mean": wavg(g, "serve_win_pct_1", "pts_1"),
            # "serve_win_pct_2_mean": wavg(g, "serve_win_pct_2", "pts_2"),
            "first_second_win_gap_mean": wavg(g, "first_second_win_gap", "pts_total"),
            "free_point_rate_1_mean": wavg(g, "free_point_rate_1", "pts_1"),
            # "quick_win_rate_1_mean": wavg(g, "quick_win_rate_1", "pts_1"),
            "quick_win_rate_2_mean": wavg(g, "quick_win_rate_2", "pts_2"),
            # "wide_share_1_mean": wavg(g, "wide_share_1", "pts_1"),
            # "body_share_1_mean": wavg(g, "body_share_1", "pts_1"),
            # "t_share_1_mean": wavg(g, "t_share_1", "pts_1"),
            "serve_dir_entropy_1_mean": wavg(g, "serve_dir_entropy_1", "pts_1"),
        })
    player_agg = wide.groupby("player").apply(agg_player_features, include_groups=False).reset_index()

    # attach match_counts and return
    out = player_agg.merge(match_counts, on="player", how="left")
    cols = ["player", "n_matches"] + [c for c in out.columns if c not in ["player", "n_matches"]]

    return out[cols].copy()

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_serve_basics(args.input_dir)
    out = build_player_serve_basics(df)

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
