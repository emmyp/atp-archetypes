import argparse
from pathlib import Path

import pandas as pd
from utils import safe_div, winsorize_features

FILE_NAME = "overview.parquet"
KEYS = ["match_id", "player"]
SET_COL = "set"
TOTAL_SET_VALUE = "Total"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build player-level features from overview.parquet"
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


def load_overview(input_dir: Path) -> pd.DataFrame:
    path = input_dir / FILE_NAME
    return pd.read_parquet(path)


def build_player_overview(df: pd.DataFrame) -> pd.DataFrame:
    # match totals: one row per match_id x player
    o = df[df[SET_COL] == TOTAL_SET_VALUE].copy()

    # how any matches do we have per player?
    match_counts = (
        o.groupby("player", as_index=False)["match_id"]
        .nunique()
        .rename(columns={"match_id": "n_matches"})
    )

    # aggregate core counts first
    sum_cols = [
        # serve counts
        "aces",
        "dfs",
        "serve_pts",
        "first_in",
        "first_won",
        "second_in",
        "second_won",
        # return / overall counts
        "return_pts",
        "return_pts_won",
        # pressure
        "bp_saved",
        "bk_pts",
        # aggression / errors
        "winners",
        "unforced",
        "winners_fh",
        "winners_bh",
        "unforced_fh",
        "unforced_bh",
    ]

    g = o.groupby("player", as_index=False)[sum_cols].sum()

    # derived totals
    g["serve_pts_won"] = g["first_won"] + g["second_won"]
    g["total_pts"] = g["serve_pts"] + g["return_pts"]
    g["total_pts_won"] = g["serve_pts_won"] + g["return_pts_won"]

    # serve rates (ratio-of-sums)
    g["ace_rate"] = safe_div(g["aces"], g["serve_pts"])
    g["df_rate"] = safe_div(g["dfs"], g["serve_pts"])
    g["first_in_pct"] = safe_div(g["first_in"], g["serve_pts"])
    g["first_win_pct"] = safe_div(g["first_won"], g["first_in"])
    g["second_win_pct"] = safe_div(g["second_won"], g["second_in"])
    g["serve_pts_won_pct"] = safe_div(g["serve_pts_won"], g["serve_pts"])

    # return rate
    g["return_pts_won_pct"] = safe_div(g["return_pts_won"], g["return_pts"])

    # pressure (bk_pts = break points faced)
    g["bp_save_pct"] = safe_div(g["bp_saved"], g["bk_pts"])
    g["bk_pts_per_return_pt"] = safe_div(g["bk_pts"], g["return_pts"])

    # aggression / error balance (per 100 total points)
    g["winners_per_100"] = 100 * safe_div(g["winners"], g["total_pts"])
    g["unforced_per_100"] = 100 * safe_div(g["unforced"], g["total_pts"])
    g["aggression_index"] = safe_div(g["winners"], g["winners"] + g["unforced"])

    # fh / bh mix
    g["fh_winner_share"] = safe_div(g["winners_fh"], g["winners"])
    g["bh_winner_share"] = safe_div(g["winners_bh"], g["winners"])
    g["fh_ue_share"] = safe_div(g["unforced_fh"], g["unforced"])
    g["bh_ue_share"] = safe_div(g["unforced_bh"], g["unforced"])

    # attach match_counts and reorder
    out = g.merge(match_counts, on="player", how="left")

    keep = [
        "player",
        "n_matches",  # useful for credibility
        "serve_pts",
        "return_pts",
        "total_pts",
        # serve
        "ace_rate",
        "df_rate",
        "first_in_pct",
        "first_in_pct",
        "second_win_pct",
        "serve_pts_won_pct",
        # return
        "return_pts_won_pct",
        # pressure
        "bp_save_pct",
        "bk_pts_per_return_pt",
        # aggression / error
        "winners_per_100",
        "unforced_per_100",
        "aggression_index",
        "fh_winner_share",
        "bh_winner_share",
        "fh_ue_share",
        "bh_ue_share",
    ]

    return out[keep].copy()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_overview(args.input_dir)
    out = build_player_overview(df)

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
                "serve_pts",
                "return_pts",
                "total_pts",
            ),
        )

    out.to_parquet(args.out_dir / FILE_NAME, index=False)


if __name__ == "__main__":
    main()
