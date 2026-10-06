from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA


ROOT = Path(__file__).resolve().parents[1]
PNG = ROOT / "png"
PNG.mkdir(exist_ok=True)


def reflection_to_ar(values):
    kappa = np.tanh(np.asarray(values, dtype=float))
    if len(kappa) == 0:
        return np.array([])
    coefficients = np.array([kappa[0]])
    for m in range(1, len(kappa)):
        previous = coefficients.copy()
        coefficients = np.empty(m + 1)
        coefficients[:m] = previous - kappa[m] * previous[::-1]
        coefficients[m] = kappa[m]
    return coefficients


def simulate_arma_state(ar, ma, n, noise_generator, seed=701, burn=1000):
    ar = np.asarray(ar, dtype=float)
    ma = np.asarray(ma, dtype=float)
    p = len(ar)
    q = len(ma)
    dimension = p + q
    transition = np.zeros((dimension, dimension))
    transition[0, :p] = ar
    transition[0, p:] = ma
    if p > 1:
        transition[1:p, :p - 1] = np.eye(p - 1)
    if q > 1:
        transition[p + 1:, p:p + q - 1] = np.eye(q - 1)
    loading = np.zeros(dimension)
    loading[0] = 1.0
    if q > 0:
        loading[p] = 1.0
    rng = np.random.default_rng(seed)
    innovations = noise_generator(rng, n + burn)
    state = np.zeros(dimension)
    output = np.empty(n + burn)
    for t, innovation in enumerate(innovations):
        state = transition @ state + loading * innovation
        output[t] = state[0]
    return output[burn:]


def spectral_ratio(lam, ar, ma):
    ar_power = np.arange(1, len(ar) + 1)
    ma_power = np.arange(1, len(ma) + 1)
    ar_poly = np.ones(len(lam), dtype=complex)
    ma_poly = np.ones(len(lam), dtype=complex)
    if len(ar):
        ar_poly -= np.exp(-1j * np.outer(lam, ar_power)) @ ar
    if len(ma):
        ma_poly += np.exp(-1j * np.outer(lam, ma_power)) @ ma
    return np.abs(ma_poly) ** 2 / np.abs(ar_poly) ** 2


def fit_whittle(x, p, q, seed=0, starts=4):
    centered = np.asarray(x, dtype=float) - np.mean(x)
    n = len(centered)
    transform = np.fft.rfft(centered)
    lam = 2 * np.pi * np.fft.rfftfreq(n)[1:]
    periodogram = np.abs(transform[1:]) ** 2 / (2 * np.pi * n)

    def decode(parameters):
        ar = reflection_to_ar(parameters[:p])
        ma = -reflection_to_ar(parameters[p:p + q])
        return ar, ma

    def objective(parameters):
        ar, ma = decode(parameters)
        ratio = spectral_ratio(lam, ar, ma)
        scale = np.mean(periodogram / ratio)
        model = np.maximum(scale * ratio, 1e-12)
        return np.mean(np.log(model) + periodogram / model)

    rng = np.random.default_rng(seed)
    initial_values = [np.zeros(p + q)]
    initial_values += [rng.normal(0, 0.2, p + q) for _ in range(starts - 1)]
    fits = [minimize(objective, value, method="L-BFGS-B", options={"maxiter": 700})
            for value in initial_values]
    result = min(fits, key=lambda item: item.fun)
    ar, ma = decode(result.x)
    ratio = spectral_ratio(lam, ar, ma)
    scale = np.mean(periodogram / ratio)
    return ar, ma, 2 * np.pi * scale, result.fun


def autocorrelation(x, nlags):
    centered = np.asarray(x, dtype=float) - np.mean(x)
    denominator = centered @ centered
    return np.array([centered[:len(centered) - lag] @ centered[lag:] / denominator
                     for lag in range(nlags + 1)])


def partial_autocorrelation(x, nlags):
    centered = np.asarray(x, dtype=float) - np.mean(x)
    gamma = np.array([centered[:len(centered) - lag] @ centered[lag:] / len(centered)
                      for lag in range(nlags + 1)])
    output = np.ones(nlags + 1)
    for lag in range(1, nlags + 1):
        output[lag] = np.linalg.solve(toeplitz(gamma[:lag]), gamma[1:lag + 1])[-1]
    return output


def innovations(x, ar, ma):
    x = np.asarray(x, dtype=float) - np.mean(x)
    residuals = np.zeros(len(x))
    for t in range(len(x)):
        ar_part = sum(ar[j] * x[t - j - 1] for j in range(len(ar)) if t > j)
        ma_part = sum(ma[j] * residuals[t - j - 1] for j in range(len(ma)) if t > j)
        residuals[t] = x[t] - ar_part - ma_part
    return residuals


def gaussian_noise(rng, n):
    return rng.normal(size=n)


def heavy_noise(rng, n):
    return rng.standard_t(2.1, size=n)


def fit_package(x, p, q):
    result = ARIMA(
        x,
        order=(p, 0, q),
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit(method_kwargs={"maxiter": 1000})
    return result.arparams, result.maparams, result.params[-1]


true_ar = np.array([0.8, -0.3, 0.1, -0.05])
true_ma = np.array([-0.6, 0.2, 0.1, 0.05])
gaussian = simulate_arma_state(true_ar, true_ma, 1000, gaussian_noise, 801)
heavy = simulate_arma_state(true_ar, true_ma, 1000, heavy_noise, 802)
gaussian_ar, gaussian_ma, gaussian_variance = fit_package(gaussian, 4, 4)
heavy_ar, heavy_ma, heavy_variance = fit_package(heavy, 4, 4)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
axes[0, 0].plot(gaussian[:250], linewidth=0.9)
axes[0, 0].set_title("Gaussian-noise ARMA(4,4)")
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Value")
axes[0, 1].plot(heavy[:250], linewidth=0.9, color="tab:orange")
axes[0, 1].set_title(r"$t_{2.1}$-noise ARMA(4,4)")
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Value")
locations = np.arange(8)
truth = np.r_[true_ar, true_ma]
axes[1, 0].plot(locations, truth, "o-", label="True")
axes[1, 0].plot(locations, np.r_[gaussian_ar, gaussian_ma], "s--", label="Estimated")
axes[1, 0].set_xticks(locations, [r"$\phi_1$", r"$\phi_2$", r"$\phi_3$", r"$\phi_4$",
                                  r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"])
axes[1, 0].set_title("Gaussian fit")
axes[1, 0].legend()
axes[1, 1].plot(locations, truth, "o-", label="True")
axes[1, 1].plot(locations, np.r_[heavy_ar, heavy_ma], "s--", label="Estimated")
axes[1, 1].set_xticks(locations, [r"$\phi_1$", r"$\phi_2$", r"$\phi_3$", r"$\phi_4$",
                                  r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"])
axes[1, 1].set_title("Heavy-tail fit")
axes[1, 1].legend()
fig.tight_layout()
fig.savefig(PNG / "problem2_arma_fits.png", dpi=180)
plt.close(fig)

data = pd.read_csv(ROOT / "code" / "nao.csv")
value_column = [column for column in data.columns if column not in {"year", "month", "day"}][0]
dates = pd.to_datetime(data[["year", "month", "day"]]).to_numpy()
values = data[value_column].to_numpy(dtype=float)
missing_index = np.flatnonzero(np.isnan(values))
valid = np.flatnonzero(~np.isnan(values))
values[missing_index] = np.interp(missing_index, valid, values[valid])
season = 365
seasonal_difference = values[season:] - values[:-season]
nao_ar, nao_ma, nao_variance, _ = fit_whittle(seasonal_difference, 4, 2, 705, 5)
nao_residuals = innovations(seasonal_difference, nao_ar, nao_ma)
simulated_difference = simulate_arma_state(
    nao_ar,
    nao_ma,
    len(seasonal_difference),
    lambda rng, n: rng.normal(scale=np.sqrt(nao_variance), size=n),
    706,
)
simulated = np.empty(len(values))
simulated[:season] = values[:season]
for t in range(season, len(values)):
    simulated[t] = simulated[t - season] + simulated_difference[t - season]

raw_acf = autocorrelation(values, 800)
diff_acf = autocorrelation(seasonal_difference, 800)
diff_pacf = partial_autocorrelation(seasonal_difference, 40)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
axes[0, 0].plot(dates, values, linewidth=0.45)
axes[0, 0].set_title(f"Supplied daily index ({value_column})")
axes[0, 0].set_xlabel("Date")
axes[0, 0].set_ylabel("Index")
axes[0, 1].plot(raw_acf, linewidth=0.9)
axes[0, 1].axhline(0, color="black", linewidth=0.6)
axes[0, 1].axvline(365, color="tab:red", linestyle="--", linewidth=0.8)
axes[0, 1].set_title("Raw empirical ACF")
axes[0, 1].set_xlabel("Lag (days)")
axes[1, 0].plot(diff_acf, linewidth=0.9)
axes[1, 0].axhline(0, color="black", linewidth=0.6)
axes[1, 0].set_title("ACF after seasonal differencing")
axes[1, 0].set_xlabel("Lag (days)")
axes[1, 1].stem(np.arange(1, 41), diff_pacf[1:], basefmt=" ")
axes[1, 1].axhline(0, color="black", linewidth=0.6)
axes[1, 1].set_title("pACF after seasonal differencing")
axes[1, 1].set_xlabel("Lag (days)")
fig.tight_layout()
fig.savefig(PNG / "problem2_nao_diagnostics.png", dpi=180)
plt.close(fig)

observed_acf = autocorrelation(seasonal_difference, 100)
simulated_acf = autocorrelation(simulated_difference, 100)
residual_acf = autocorrelation(nao_residuals[50:], 100)

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
axes[0].plot(observed_acf, label="Observed seasonal difference")
axes[0].plot(simulated_acf, label="Simulated seasonal difference", linestyle="--")
axes[0].axhline(0, color="black", linewidth=0.6)
axes[0].set_title("Observed and fitted-model ACFs")
axes[0].set_xlabel("Lag (days)")
axes[0].legend()
axes[1].plot(residual_acf)
axes[1].axhline(1.96 / np.sqrt(len(nao_residuals)), color="tab:red", linestyle="--")
axes[1].axhline(-1.96 / np.sqrt(len(nao_residuals)), color="tab:red", linestyle="--")
axes[1].set_title("Fitted residual ACF")
axes[1].set_xlabel("Lag (days)")
fig.tight_layout()
fig.savefig(PNG / "problem2_nao_validation.png", dpi=180)
plt.close(fig)

print("GAUSSIAN_AR", np.round(gaussian_ar, 4))
print("GAUSSIAN_MA", np.round(gaussian_ma, 4))
print("GAUSSIAN_VARIANCE", round(gaussian_variance, 4))
print("HEAVY_AR", np.round(heavy_ar, 4))
print("HEAVY_MA", np.round(heavy_ma, 4))
print("HEAVY_VARIANCE", round(heavy_variance, 4))
print("VALUE_COLUMN", value_column)
print("MISSING_ROWS", missing_index)
print("NAO_AR", np.round(nao_ar, 4))
print("NAO_MA", np.round(nao_ma, 4))
print("NAO_VARIANCE", round(nao_variance, 4))
print("RAW_ACF_1_365", np.round(raw_acf[[1, 2, 7, 30, 365]], 4))
print("DIFF_ACF_1_10", np.round(diff_acf[1:11], 4))
print("DIFF_PACF_1_10", np.round(diff_pacf[1:11], 4))
print("RESIDUAL_ACF_MAX", round(np.max(np.abs(residual_acf[1:])), 4))
