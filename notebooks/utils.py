import np


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
