from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.signal.windows import dpss
from statsmodels.tsa.arima_process import arma_acovf
from problem2 import periodogram, multitaper_spectrum

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "png"
SEED = 70103
NAO_DATA_PATH = DATA_DIR / "nao.csv"
N = 1024


def load_nao(path=NAO_DATA_PATH, n=N):
    table = pd.read_csv(path).iloc[:n]
    columns = [column for column in table if column not in {"year", "month", "day"}]
    if len(columns) != 1 or len(table) != n:
        raise ValueError("Expected n rows and one index column")
    values = pd.to_numeric(table[columns[0]], errors="raise").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Missing values in the selected segment")
    dates = pd.to_datetime(table[["year", "month", "day"]])
    if not (dates.diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("The selected data are not sampled daily without gaps")
    return values


def spectral_density(frequencies, parameters):
    amplitude, decay, floor = parameters
    return amplitude * np.exp(-decay * np.abs(frequencies)) + floor


def whittle_likelihood(dft, parameters):
    dft = np.asarray(dft)
    n = len(dft)
    frequencies = np.fft.fftfreq(n)
    if np.any(np.asarray(parameters) <= 0):
        return np.inf
    selected = np.arange(1, (n + 1) // 2)
    model = spectral_density(frequencies[selected], parameters)
    objective = np.sum(np.log(model) + np.abs(dft[selected]) ** 2 / model)
    if n % 2 == 0:
        nyquist = spectral_density(np.array([0.5]), parameters)[0]
        objective += 0.5 * (np.log(nyquist) + np.abs(dft[n // 2]) ** 2 / nyquist)
    return float(objective)


def fit_spectrum(dft):
    n = len(dft)
    variance = float(np.sum(np.abs(dft) ** 2) / n)
    fits = []
    bounds = [(-16, 16), (np.log(0.1), np.log(1000)), (-20, 10)]
    for decay in (10.0, 25.0, 60.0, 120.0):
        initial = np.log([max(variance * decay / 2, 1e-6), decay, max(variance * 0.001, 1e-8)])
        fit = minimize(lambda z: whittle_likelihood(dft, np.exp(z)), initial,
                       method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000, "ftol": 1e-12})
        if fit.success and np.isfinite(fit.fun):
            fits.append(fit)
    if not fits:
        raise RuntimeError("Whittle optimization did not converge")
    result = min(fits, key=lambda fit: fit.fun)
    return np.exp(result.x), result


def spectrum_to_acf(parameters, n=N):
    amplitude, decay, floor = parameters
    h = np.arange(n)
    acf = 2 * amplitude * decay * (1 - (-1.0) ** h * np.exp(-decay / 2)) / (decay ** 2 + (2 * np.pi * h) ** 2)
    acf[0] += floor
    return acf


def gaussian_log_likelihood(series, acf):
    series = np.asarray(series, dtype=float)
    factor = cholesky(toeplitz(acf[:len(series)]), lower=True)
    standardized = solve_triangular(factor, series, lower=True)
    return float(-0.5 * (len(series) * np.log(2 * np.pi) + 2 * np.log(np.diag(factor)).sum() + standardized @ standardized))


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    values = load_nao()
    mean = float(values.mean())
    series = values - mean
    dft = np.fft.fft(series) / np.sqrt(N)
    parameters, fit = fit_spectrum(dft)
    acf = spectrum_to_acf(parameters)
    f, raw = periodogram(series)
    _, mt = multitaper_spectrum(series, dpss(N, 4, Kmax=7, norm=2))
    model = spectral_density(f, parameters)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.arange(N), values, linewidth=0.8)
    axes[0].set_xlabel("Day from 1979-01-01")
    axes[0].set_ylabel("Supplied index")
    axes[1].semilogy(f[1:], raw[1:], alpha=0.35, linewidth=0.7, label="Periodogram")
    axes[1].semilogy(f[1:], mt[1:], label="DPSS NW=4, K=7")
    axes[1].semilogy(f[1:], model[1:], "k--", label="Whittle fit")
    axes[1].set_xlabel("Frequency (cycles/day)")
    axes[1].set_ylabel("Two-sided spectral density")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "problem3_whittle_fit.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    empirical = np.array([series[:N - h] @ series[h:] / N for h in range(81)])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(81), empirical / empirical[0], label="Empirical")
    ax.plot(np.arange(81), acf[:81] / acf[0], label="Spectral model")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "problem3_acf.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    seasonal_lag = 365
    differences = series[seasonal_lag:] - series[:-seasonal_lag]
    h = np.arange(len(differences))
    extended = spectrum_to_acf(parameters, N + seasonal_lag)
    difference_acf = 2 * extended[h] - extended[h + seasonal_lag] - extended[np.abs(h - seasonal_lag)]
    old_ar = np.array([0.6642, 0.3518, -0.2324, 0.0837])
    old_ma = np.array([1.1556, 0.3247])
    old_acf = arma_acovf(np.r_[1, -old_ar], np.r_[1, old_ma], nobs=len(differences), sigma2=0.1956)
    exact_full = gaussian_log_likelihood(series, acf)
    exact_differences = gaussian_log_likelihood(differences, difference_acf)
    exact_old = gaussian_log_likelihood(differences, old_acf)
    quadrature = {}
    for lag in (0, 1, 10, 100, 1023):
        numeric = 2 * quad(lambda frequency: spectral_density(frequency, parameters) * np.cos(2 * np.pi * lag * frequency),
                           0, 0.5, epsabs=1e-9, limit=1000)[0]
        quadrature[str(lag)] = {"analytic": float(acf[lag]), "numeric": float(numeric)}
        if not np.isclose(numeric, acf[lag], atol=1e-7, rtol=1e-6):
            raise AssertionError("Spectral inversion check failed")
    results = {"n": N, "column": pd.read_csv(NAO_DATA_PATH, nrows=1).columns[-1],
               "sample_mean": mean, "parameters": parameters.tolist(), "whittle_objective": float(fit.fun),
               "optimizer_success": bool(fit.success), "variance_model": float(acf[0]),
               "variance_empirical": float(np.mean(series ** 2)),
               "exact_raw_log_likelihood": exact_full,
               "common_difference_n": len(differences),
               "stationary_difference_log_likelihood": exact_differences,
               "old_sarima_difference_log_likelihood": exact_old,
               "spectral_inversion": quadrature}
    (ROOT / "problem3_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
