from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.optimize import minimize, least_squares


ROOT = Path(__file__).resolve().parents[1]
PNG = ROOT / "png"
PNG.mkdir(exist_ok=True)


def covariance(h, theta):
    h = np.asarray(h, dtype=float)
    numerator = 2 * np.exp(-theta / 2) * (
        -theta * np.cos(np.pi * h)
        + np.exp(theta / 2) * theta
        + 2 * np.pi * h * np.sin(np.pi * h)
    )
    return numerator / (theta ** 2 + (2 * np.pi * h) ** 2)


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


def fit_whittle(x, p, q, seed=0, starts=8):
    centered = np.asarray(x) - np.mean(x)
    n = len(centered)
    transform = np.fft.rfft(centered)
    lam = 2 * np.pi * np.fft.rfftfreq(n)[1:]
    periodogram = np.abs(transform[1:]) ** 2 / (2 * np.pi * n)

    def decode(parameters):
        return reflection_to_ar(parameters[:p]), -reflection_to_ar(parameters[p:p + q])

    def objective(parameters):
        ar, ma = decode(parameters)
        ratio = spectral_ratio(lam, ar, ma)
        scale = np.mean(periodogram / ratio)
        model = np.maximum(scale * ratio, 1e-12)
        return np.mean(np.log(model) + periodogram / model)

    rng = np.random.default_rng(seed)
    starts_list = [np.zeros(p + q)] + [rng.normal(0, 0.25, p + q) for _ in range(starts - 1)]
    fits = [minimize(objective, start, method="L-BFGS-B", options={"maxiter": 900})
            for start in starts_list]
    result = min(fits, key=lambda item: item.fun)
    ar, ma = decode(result.x)
    ratio = spectral_ratio(lam, ar, ma)
    variance = 2 * np.pi * np.mean(periodogram / ratio)
    return ar, ma, variance


def empirical_acf(x, nlags):
    centered = np.asarray(x) - np.mean(x)
    denominator = centered @ centered
    return np.array([centered[:len(centered) - lag] @ centered[lag:] / denominator
                     for lag in range(nlags + 1)])


def spectral_acf(ar, ma, nlags, grid=32768):
    lam = 2 * np.pi * np.arange(grid) / grid
    spectrum = spectral_ratio(lam, ar, ma)
    gamma = np.fft.ifft(spectrum).real[:nlags + 1]
    return gamma / gamma[0]


def rational_fit(theta, p, q, seed=0, starts=8):
    frequency = np.linspace(-0.5, 0.5, 2001)
    target = np.exp(-theta * np.abs(frequency))
    lam = 2 * np.pi * frequency

    def decode(parameters):
        ar = reflection_to_ar(parameters[:p])
        ma = -reflection_to_ar(parameters[p:p + q])
        scale = np.exp(parameters[-1])
        return ar, ma, scale

    def residual(parameters):
        ar, ma, scale = decode(parameters)
        model = np.maximum(scale * spectral_ratio(lam, ar, ma), 1e-12)
        return np.log(model) - np.log(target)

    rng = np.random.default_rng(seed)
    initial_values = [np.zeros(p + q + 1)]
    initial_values += [rng.normal(0, 0.2, p + q + 1) for _ in range(starts - 1)]
    fits = [least_squares(residual, value, max_nfev=3000) for value in initial_values]
    result = min(fits, key=lambda item: np.mean(item.fun ** 2))
    return (*decode(result.x), np.sqrt(np.mean(result.fun ** 2)))


theta = 6.0
n = 1000
lags = np.arange(n)
matrix = toeplitz(covariance(lags, theta))
rng = np.random.default_rng(709)
sample = np.linalg.cholesky(matrix + 1e-12 * np.eye(n)) @ rng.normal(size=n)
p = 4
q = 4
time_ar, time_ma, time_variance = fit_whittle(sample, p, q, 710)
spectral_ar, spectral_ma, spectral_scale, spectral_rmse = rational_fit(theta, p, q, 711)

display_lags = 60
true_acf = covariance(np.arange(display_lags + 1), theta) / covariance(0, theta)
sample_acf = empirical_acf(sample, display_lags)
fitted_acf = spectral_acf(time_ar, time_ma, display_lags)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(true_acf, label="True ACF", linewidth=2)
ax.plot(sample_acf, label="Empirical ACF", alpha=0.8)
ax.plot(fitted_acf, label="Fitted ARMA(4,4) ACF", linestyle="--")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xlabel("Lag")
ax.set_ylabel("Correlation")
ax.set_title("True, empirical, and ARMA-implied ACFs")
ax.legend()
fig.tight_layout()
fig.savefig(PNG / "problem3_acf_comparison.png", dpi=180)
plt.close(fig)

frequency = np.linspace(-0.5, 0.5, 2001)
lam = 2 * np.pi * frequency
true_spectrum = np.exp(-theta * np.abs(frequency))
time_spectrum = time_variance * spectral_ratio(lam, time_ar, time_ma)
direct_spectrum = spectral_scale * spectral_ratio(lam, spectral_ar, spectral_ma)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(frequency, true_spectrum, label="True spectrum", linewidth=2)
ax.plot(frequency, time_spectrum, label="Time-series ARMA fit", linestyle="--")
ax.plot(frequency, direct_spectrum, label="Direct rational fit", linestyle=":", linewidth=2)
ax.set_xlabel("Frequency (cycles per observation)")
ax.set_ylabel("Spectral density")
ax.set_title("True and fitted spectral densities")
ax.legend()
fig.tight_layout()
fig.savefig(PNG / "problem3_spectrum_comparison.png", dpi=180)
plt.close(fig)

time_log_rmse = np.sqrt(np.mean((np.log(np.maximum(time_spectrum, 1e-12))
                                 - np.log(true_spectrum)) ** 2))
print("THETA", theta)
print("TIME_AR", np.round(time_ar, 5))
print("TIME_MA", np.round(time_ma, 5))
print("TIME_VARIANCE", round(time_variance, 5))
print("SPECTRAL_AR", np.round(spectral_ar, 5))
print("SPECTRAL_MA", np.round(spectral_ma, 5))
print("SPECTRAL_SCALE", round(spectral_scale, 5))
print("TIME_LOG_RMSE", round(time_log_rmse, 5))
print("SPECTRAL_LOG_RMSE", round(spectral_rmse, 5))
