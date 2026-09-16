from pathlib import Path

import numpy as np
from autocovariance import sample_autocovariance
import matplotlib.pyplot as plt


def generate_X(n=512, var_zj1=2.0, var_zj2=2.0, seed=None):


    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2 or n % 2:
        raise ValueError("n must be a positive even integer")

    num_frequencies = n // 2 - 1

    def variance_array(value, name):
        value = np.asarray(value, dtype=float)
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} must contain finite, nonnegative variances")
        if value.ndim == 0:
            return np.full(num_frequencies, value.item())
        if value.shape != (num_frequencies,):
            raise ValueError(f"{name} must be a scalar or have shape ({num_frequencies},)")
        return value

    variance_1 = variance_array(var_zj1, "var_zj1")
    variance_2 = variance_array(var_zj2, "var_zj2")
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1)
    X_components = np.zeros((n // 2 + 1, n))

    Z_0, Z_half = rng.normal(0.0, 1.0, size=2)
    X_components[0, :] = Z_0
    X_components[n // 2, :] = Z_half * (-1.0) ** t


    scales = np.sqrt(np.column_stack((variance_1, variance_2)))
    Z = rng.normal(0.0, scales, size=(num_frequencies, 2))
    for j in range(1, n // 2):
        omega_j = 2 * np.pi * j / n
        X_components[j, :] = (
            Z[j - 1, 0] * np.cos(omega_j * t)
            + Z[j - 1, 1] * np.sin(omega_j * t)
        )

    X = X_components.sum(axis=0) / np.sqrt(n)
    return X


def plot_series_and_autocovariance(series, address, fig_l=10, fig_w=7, *,
                                  max_lag=200, title="Simulated series",
                                  ylim=None, series_ylim=None):

    values = sample_autocovariance(series, max_lag)
    lags = np.arange(len(values))
    fig, axes = plt.subplots(2, 1, figsize=(fig_l, fig_w))
    axes[0].plot(np.arange(1, len(series) + 1), series, linewidth=0.7)
    axes[0].set(xlabel="Time step t", ylabel="X_t", title=title)
    axes[1].stem(lags, values, markerfmt=".", basefmt="k-")
    axes[1].set(xlabel="Lag k (time steps)", ylabel="Autocovariance",
                title="Sample autocovariance (denominator n)")
    if series_ylim is not None:
        axes[0].set_ylim(series_ylim)
    if ylim is not None:
        axes[1].set_ylim(ylim)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    output = Path(address)
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / "png" / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    return fig, axes


if __name__ == "__main__":
    n = 512
    seed = 42
    j = np.arange(1, n // 2)
    u = (j - 1) / (len(j) - 1)
    sigma_min, sigma_max = 0.1, 5.0
    cycles = 3
    sigma_linear = sigma_min + (sigma_max - sigma_min) * u
    sigma_oscillating = sigma_min + (sigma_max - sigma_min) * (
        1 - np.cos(2 * np.pi * cycles * u)
    ) / 2
    cases = [
        ("sigma_linear", sigma_linear, "Linearly increasing sigma_j: 0.1 to 5"),
        ("sigma_oscillating", sigma_oscillating, "Oscillating sigma_j: 0.1 to 5, three cycles"),
    ]


    series_by_case = {
        name: generate_X(n, sigma**2, sigma**2, seed=seed)
        for name, sigma, title in cases
    }
    max_lag = 200
    series_limit = 1.1 * max(np.abs(x).max() for x in series_by_case.values())
    limit = 1.1 * max(
        np.abs(sample_autocovariance(x, max_lag)).max()
        for x in series_by_case.values()
    )
    for name, sigma, title in cases:
        fig, axes = plot_series_and_autocovariance(
            series_by_case[name], f"HW1_5_{name}.png", 10, 7,
            title=title,
            max_lag=max_lag, ylim=(-limit, limit),
            series_ylim=(-series_limit, series_limit),
        )
        plt.close(fig)

    default = generate_X(n=n, seed=seed)
    white_noise = np.random.default_rng(seed + 1).normal(size=n)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="row", sharey="row")
    for column, (values, title) in enumerate([
        (default, "Trigonometric simulation, default variance"),
        (white_noise, "Independent N(0,1) white noise"),
    ]):
        axes[0, column].plot(np.arange(1, n + 1), values, linewidth=0.7)
        axes[0, column].set(title=title, xlabel="Time step t", ylabel="Value")
        covariance = sample_autocovariance(values, max_lag)
        axes[1, column].stem(np.arange(max_lag + 1), covariance,
                             markerfmt=".", basefmt="k-")
        axes[1, column].set(xlabel="Lag k", ylabel="Autocovariance")
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parents[1] / "png" / "HW1_5_default.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
