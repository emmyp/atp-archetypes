# %% read necessary table
from pathlib import Path

import pandas as pd
from utils import safe_div, wavg

input_dir = Path("../data/filtered")
ret = pd.read_parquet(input_dir / "return_outcomes.parquet")
# %% feature engineering
r = ret.copy()

# how often do you actually put the return in play when you have a chance?
r["return_make"] = safe_div(r["in_play"], r["returnable"])

# once the return lands, do you win those points?
r["win_after_make"] = safe_div(r["in_play_won"], r["in_play"])

# aggression
r["return_winner_rate"] = safe_div(r["winners"], r["returnable"])


def return_outcomes_agg(df, row_name, prefix):
    g = df[df["row"] == row_name].groupby("player")
    out = pd.DataFrame(
        {
            "player": g.size().index,
            f"{prefix}_return_make": g.apply(
                lambda x: wavg(x, "return_make", "returnable"), include_groups=False
            ),
            f"{prefix}_win_after_make": g.apply(
                lambda x: wavg(x, "win_after_make", "in_play"), include_groups=False
            ),
            f"{prefix}_return_winner_rate": g.apply(
                lambda x: wavg(x, "return_winner_rate", "returnable"),
                include_groups=False,
            ),
        }
    ).reset_index(drop=True)
    return out


tot = return_outcomes_agg(r, "Total", "ret")
v1 = return_outcomes_agg(r, "v1st", "ret_v1")
v2 = return_outcomes_agg(r, "v2nd", "ret_v2")

player_ret = tot.merge(v1, on="player", how="left")
player_ret = player_ret.merge(v2, on="player", how="left")

# second-serve aggression delta (which players hunt 2nd serves)
player_ret["ret_winner_rate_delta_2minus1"] = (
    player_ret["ret_v2_return_winner_rate"] - player_ret["ret_v1_return_winner_rate"]
)

# does player attack second serve but still make them or do they spray the return when attacking?
player_ret["ret_make_delta_2minus1"] = (
    player_ret["ret_v2_return_make"] - player_ret["ret_v1_return_make"]
)

keep = [
    "player",
    "ret_return_make",
    "ret_win_after_make",
    "ret_return_winner_rate",
    "ret_winner_rate_delta_2minus1",
    "ret_make_delta_2minus1",
]
player_ret = player_ret[keep].copy()

# %% output
out_dir = Path("../data/processed/features")
player_ret.to_parquet(out_dir / "return_outcomes.parquet", index=False)
