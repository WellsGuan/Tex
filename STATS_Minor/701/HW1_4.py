from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv(
    Path(__file__).resolve().parent / "cet.csv",
    parse_dates=["date"],
    index_col="date"
)

df = df.sort_index()

### Part 1 ###

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    df.index, df["temp"],
    linewidth=0.4, alpha=0.6,
)

ax.set_title("Central England Daily Mean Temperature, 1772-2025")
ax.set_xlabel("Year")
ax.set_ylabel("Daily mean temperature (°C)")

ax.xaxis.set_major_locator(mdates.YearLocator(25))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(df.index.min(), df.index.max())
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    Path(__file__).resolve().parent / "cet_temp.png",
    dpi=300, bbox_inches="tight",
)
plt.show()

### Part 2 ###
x = df["temp"].iloc[:365].to_numpy()
x = x - np.mean(x)

n = len(x)
max_lag = 200

corr = np.correlate(x, x, mode="full")[n - 1:]

acf_values = corr[:max_lag + 1] / corr[0]

fig_2, ax_2 = plt.subplots(figsize=(10, 4))
ax_2.stem(np.arange(max_lag + 1), acf_values)
ax_2.set_xlabel("Lag (days)")
ax_2.set_ylabel("Autocorrelation")
ax_2.set_title("Sample ACF of daily CET  for the first 365 days")
fig_2.tight_layout()
fig_2.savefig(
    Path(__file__).resolve().parent / "Lags.png",
    dpi=300, bbox_inches="tight",
)
plt.show()

### Part 3 ###

n = 365
t = np.arange(1, n + 1, dtype=float)
A = np.column_stack((np.ones(n), t, t**2))

beta, residuals, rank, singular_values = np.linalg.lstsq(A, x, rcond=None)

x_hat = A @ beta
resid = x - x_hat

fig_3, ax_3 = plt.subplots(figsize=(10, 4))
ax_3.plot(t,resid,
    linewidth=0.4, alpha=0.6,)
ax_3.set_xlabel("Days")
ax_3.set_ylabel("Redisual")
ax_3.set_title("Sample residues of daily CET for the first 365 days")
fig_3.tight_layout()
fig_3.savefig(
    Path(__file__).resolve().parent / "Residues.png",
    dpi=300, bbox_inches="tight",
)
plt.show()