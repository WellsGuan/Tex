import numpy as np


def sample_autocovariance(series, max_lag=None):

    x = np.asarray(series, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("series must be a nonempty finite one-dimensional array")
    n = len(x)
    if max_lag is None:
        max_lag = n - 1
    if not isinstance(max_lag, (int, np.integer)) or not 0 <= max_lag < n:
        raise ValueError("max_lag must be an integer between 0 and len(series)-1")
    centered = x - x.mean()
    return np.array([
        np.dot(centered[k:], centered[:n - k]) / n
        for k in range(max_lag + 1)
    ])
