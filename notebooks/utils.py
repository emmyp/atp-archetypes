import numpy as np
import pandas as pd


def safe_div(a, b):
    a = a.astype(float)
    b = b.astype(float)
    return np.where(b == 0, np.nan, a / b)


def norm_entropy(p3):
    p = np.clip(p3, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1) / np.log(3)


def wavg(group, value_col, weight_col):
    w = group[weight_col].to_numpy()
    v = group[value_col].to_numpy()
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if m.sum() == 0:
        return np.nan
    return np.sum(v[m] * w[m]) / np.sum(w[m])


def winsorize_features(
    df: pd.DataFrame,
    *,
    p_low: float = 0.005,
    p_high: float = 0.995,
    match_col: str = "n_matches",
    min_matches: int = 8,
    feature_cols: list[str] | None = None,
    exclude_cols: tuple[str, ...] = (
        "player",
        "n_matches",
        "serve_pts",
        "return_pts",
        "total_pts",
    ),
) -> pd.DataFrame:
    """
    winsorize numeric feature columns using percentile clipping.

    percentile bounds are computed on a *cohort* defined by `df[match_col] >= min_matches`,
    then applied to all rows
    """
    d = df.copy()

    if feature_cols is None:
        feature_cols = [
            c
            for c in d.columns
            if (c not in exclude_cols) and pd.api.types.is_numeric_dtype(d[c])
        ]

    # compute bounds on the clustering cohort.
    if match_col in d.columns:
        cohort_mask = (
            pd.to_numeric(d[match_col], errors="coerce").fillna(0) >= min_matches
        )
        cohort = d.loc[cohort_mask]
    else:
        cohort = d

    for c in feature_cols:
        s = pd.to_numeric(cohort[c], errors="coerce").dropna()
        if s.shape[0] < 5:
            # too few observations to estimate robust bounds; leave as-is.
            continue
        lo = float(s.quantile(p_low))
        hi = float(s.quantile(p_high))
        d[c] = pd.to_numeric(d[c], errors="coerce").clip(lower=lo, upper=hi)

    return d
