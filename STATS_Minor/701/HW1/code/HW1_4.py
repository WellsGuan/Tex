from pathlib import Path

import numpy as np
from autocovariance import sample_autocovariance
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from math import pi as pi

df = pd.read_csv(
    Path(__file__).resolve().parent / "cet.csv",
    parse_dates=["date"],
    index_col="date"
)

df = df.sort_index()
if df.index.has_duplicates or df["temp"].isna().any():
    raise ValueError("Duplicate dates or missing temperatures")
if not df.index.equals(pd.date_range(df.index[0], df.index[-1], freq="D")):
    raise ValueError("The daily series contains gaps")

def Plot_and_meanvalue_check(
    series, step, address, fig_l=10, fig_w=6, *,
    times=None, title="Time series diagnostics", ylabel="Residual (deg C)",
):


    series = np.asarray(series, dtype=float)
    if series.ndim != 1 or not np.isfinite(series).all():
        raise ValueError("series must be a finite one-dimensional array")
    if not isinstance(step, (int, np.integer)) or not 2 <= step <= len(series):
        raise ValueError("step must be an integer between 2 and len(series)")
    if times is None:
        times = np.arange(1, len(series) + 1)
    times = np.asarray(times)
    if times.ndim != 1 or len(times) != len(series):
        raise ValueError("times and series must have the same length")

    rolling_mean = np.convolve(series, np.ones(step) / step, mode="valid")
    rolling_std = pd.Series(series).rolling(step).std(ddof=1).to_numpy()[step - 1:]
    window_times = times[step - 1:]

    fig, axes = plt.subplots(2, 1, figsize=(fig_l, fig_w), sharex=True)
    axes[0].plot(times, series, linewidth=0.6, alpha=0.6, label="Observations")
    axes[0].plot(window_times, rolling_mean, color="tab:red",
                 linewidth=1.5, label=f"{step}-observation rolling mean")
    axes[0].axhline(series.mean(), color="0.4", linestyle="--", linewidth=0.8,
                   label="Overall mean")
    axes[0].set_title(title)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    axes[1].plot(window_times, rolling_std, color="tab:orange",
                 label=f"{step}-observation rolling sample SD")
    axes[1].set_ylabel("Standard deviation (deg C)")
    axes[1].legend(fontsize=8)
    if np.issubdtype(times.dtype, np.datetime64):
        locator = mdates.AutoDateLocator()
        axes[1].xaxis.set_major_locator(locator)
        axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axes[1].set_xlabel("Date")
    else:
        axes[1].set_xlabel("Day number")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    output = Path(address)
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / "png" / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    return fig, axes, rolling_mean, rolling_std


window_full = 365
fig, axes, mean_full, std_full = Plot_and_meanvalue_check(
    df["temp"].to_numpy(), window_full, "cet_temp.png", 12, 6,
    times=df.index,
    title=f"Daily CET and {window_full}-day rolling statistics, 1772-2025",
    ylabel="Daily mean temperature (deg C)",
)


x = df["temp"].iloc[:365].to_numpy()
x = x - np.mean(x)

n = len(x)
max_lag = 200


acf_values = sample_autocovariance(x, max_lag)

fig_2, ax_2 = plt.subplots(figsize=(10, 4))
ax_2.stem(np.arange(max_lag + 1), acf_values)
ax_2.set_xlabel("Lag (days)")
ax_2.set_ylabel("Autocovariance (deg C squared)")
ax_2.set_title("Sample autocovariance of daily CET for the first 365 days")
fig_2.tight_layout()
fig_2.savefig(
    Path(__file__).resolve().parents[1] / "png" / "Lags.png",
    dpi=300, bbox_inches="tight",
)


n = 365
x = df["temp"].iloc[:n].to_numpy()
t = np.arange(1, n + 1, dtype=float)
A = np.column_stack((np.ones(n), t))

beta, residuals, rank, singular_values = np.linalg.lstsq(A, x, rcond=None)

x_hat = A @ beta
resid = x - x_hat

resid_linear = resid.copy()
fig_3, axes_3, mean_linear, std_linear = Plot_and_meanvalue_check(
    resid_linear, 30, "Residues.png", times=df.index[:n],
    title="Linear-model residuals and 30-day rolling statistics",
)


n = 3*365

x = df["temp"].iloc[:n].to_numpy()

t = np.arange(1, n + 1, dtype=float)
angle_t = t*2*pi/365
A = np.column_stack((np.ones(n), t,np.cos(angle_t),np.sin(angle_t)))

beta, residuals, rank, singular_values = np.linalg.lstsq(A, x, rcond=None)

x_hat = A @ beta
resid = x - x_hat

resid_seasonal = resid.copy()
fig_4, axes_4, mean_seasonal, std_seasonal = Plot_and_meanvalue_check(
    resid_seasonal, 30, "Tri_Residues.png", times=df.index[:n],
    title="Seasonal-model residuals and 30-day rolling statistics",
)

for values, filename, title in [
    (resid_linear, "Residual_ACov.png", "Linear-model residual autocovariance"),
    (resid_seasonal, "Seasonal_ACov.png", "Seasonal-model residual autocovariance"),
]:
    covariance = sample_autocovariance(values, 200)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(np.arange(201), covariance, markerfmt=".", basefmt="k-")
    ax.set(xlabel="Lag (days)", ylabel="Autocovariance (deg C squared)", title=title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parents[1] / "png" / filename,
                dpi=300, bbox_inches="tight")
    plt.close(fig)
plt.close("all")
