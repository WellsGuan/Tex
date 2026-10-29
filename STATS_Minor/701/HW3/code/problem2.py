from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal.windows import dpss
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "png"
SEED = 70103
TIDE_DATA_PATH = DATA_DIR / "tidegauge.csv"


def circulant_embedding(acf, rng=None):
    acf = np.asarray(acf, dtype=float)
    if acf.ndim != 1 or len(acf) == 0 or not np.isfinite(acf).all():
        raise ValueError("acf must be a finite nonempty vector")
    rng = np.random.default_rng() if rng is None else rng
    if len(acf) == 1:
        if acf[0] < 0:
            raise ValueError("negative variance")
        return rng.normal(scale=np.sqrt(acf[0]), size=1)
    row = np.r_[acf, acf[-2:0:-1]]
    eigenvalues = np.fft.fft(row).real
    tolerance = 1e-11 * max(1.0, np.max(np.abs(eigenvalues)))
    if eigenvalues.min() < -tolerance:
        raise ValueError("Embedding is not positive semidefinite; provide a valid longer covariance extension")
    eigenvalues = np.maximum(eigenvalues, 0)
    white_noise = rng.standard_normal(len(row))
    sample = np.fft.ifft(np.sqrt(eigenvalues) * np.fft.fft(white_noise)).real
    return sample[:len(acf)]


def periodogram(series):
    series = np.asarray(series, dtype=float)
    return np.fft.rfftfreq(len(series)), np.abs(np.fft.rfft(series)) ** 2 / len(series)


def multitaper_spectrum(series, tapers):
    series = np.asarray(series, dtype=float)
    tapers = np.atleast_2d(tapers)
    if tapers.shape[1] != len(series):
        raise ValueError("tapers must have the same length as the data")
    estimates = np.abs(np.fft.rfft(tapers * series, axis=1)) ** 2
    estimates /= np.sum(tapers ** 2, axis=1)[:, None]
    return np.fft.rfftfreq(len(series)), estimates.mean(axis=0)


def sine_tapers(n, k):
    t = np.arange(1, n + 1)
    orders = np.arange(1, k + 1)
    return np.sqrt(2 / (n + 1)) * np.sin(np.pi * orders[:, None] * t / (n + 1))


def normal_peak_acf(n, variance, center, width):
    h = np.arange(n)
    return variance * np.exp(-2 * np.pi ** 2 * width ** 2 * h ** 2) * np.cos(2 * np.pi * center * h)


def normal_peak_spectrum(frequencies, variance, center, width):
    output = np.zeros_like(frequencies, dtype=float)
    for shift in (-1, 0, 1):
        for sign in (-1, 1):
            output += variance / (2 * np.sqrt(2 * np.pi) * width) * np.exp(-0.5 * ((frequencies - sign * center + shift) / width) ** 2)
    return output


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def compare_peak_resolution(rng):
    n = 4096
    h = np.arange(n)
    centers = (0.16, 0.164)
    r = 0.994
    acf = sum(r ** h * np.cos(2 * np.pi * f * h) for f in centers)
    acf[0] += 0.05
    sample = circulant_embedding(acf, rng)
    f, raw = periodogram(sample)
    tapers = dpss(n, NW=24, Kmax=47, norm=2)
    _, mt = multitaper_spectrum(sample, tapers)
    truth = np.full_like(f, 0.05)
    for center in centers:
        for sign in (-1, 1):
            truth += 0.5 * (1 - r ** 2) / (1 + r ** 2 - 2 * r * np.cos(2 * np.pi * (f - sign * center)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, estimate, label in zip(axes, (raw, mt), ("Periodogram", "DPSS multitaper, NW=24, K=47")):
        ax.plot(f, estimate, linewidth=0.8, label=label)
        ax.plot(f, truth, "k--", linewidth=1, label="True spectrum")
        ax.set_xlim(0.145, 0.18)
        ax.set_xlabel("Frequency (cycles/sample)")
        ax.set_ylabel("Two-sided spectral density")
        ax.legend(fontsize=8)
    save_figure(fig, "problem2_resolution.png")
    return {"n": n, "centers": centers, "r": r, "NW": 24, "K": 47, "bandwidth": 48 / n}


def compare_faint_peak_detection(rng):
    n = 4096
    components = ((2.0, 0.10, 0.004), (0.02, 0.27, 0.006))
    acf = sum(normal_peak_acf(n, *component) for component in components)
    acf[0] += 1.0
    sample = circulant_embedding(acf, rng)
    f, raw = periodogram(sample)
    tapers = dpss(n, NW=8, Kmax=15, norm=2)
    _, mt = multitaper_spectrum(sample, tapers)
    truth = np.ones_like(f)
    for component in components:
        truth += normal_peak_spectrum(f, *component)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, estimate, label in zip(axes, (raw, mt), ("Periodogram", "DPSS multitaper, NW=8, K=15")):
        ax.plot(f, estimate, linewidth=0.7, label=label)
        ax.plot(f, truth, "k--", linewidth=1.2, label="True spectrum")
        ax.set_xlim(0.22, 0.32)
        ax.set_ylim(0, 6)
        ax.set_xlabel("Frequency (cycles/sample)")
        ax.set_ylabel("Two-sided spectral density")
        ax.legend(fontsize=8)
    save_figure(fig, "problem2_faint_peak.png")
    center_index = np.argmin(np.abs(f - 0.27))
    raw_values, mt_values = [], []
    for _ in range(128):
        realization = circulant_embedding(acf, rng)
        raw_values.append(periodogram(realization)[1][center_index])
        mt_values.append(multitaper_spectrum(realization, tapers)[1][center_index])
    return {"n": n, "components": components, "noise_variance": 1.0, "NW": 8, "K": 15,
            "true_peak": float(truth[center_index]), "raw_mean": float(np.mean(raw_values)),
            "raw_sd": float(np.std(raw_values, ddof=1)), "mt_mean": float(np.mean(mt_values)),
            "mt_sd": float(np.std(mt_values, ddof=1)), "replicates": 128}


def explore_harmonics(rng):
    n = 2048
    t = np.arange(n)
    f0 = 0.05
    phase = rng.uniform(0, 2 * np.pi)
    wave = np.cos(2 * np.pi * f0 * t + phase)
    noise = rng.normal(scale=0.2, size=n)
    transformations = [("tanh, alpha=0.5", np.tanh(0.5 * wave)),
                       ("tanh, alpha=2", np.tanh(2 * wave)),
                       ("tanh, alpha=8", np.tanh(8 * wave)),
                       ("Half-wave rectification", np.maximum(wave, 0))]
    tapers = dpss(n, NW=3, Kmax=5, norm=2)
    fig, axes = plt.subplots(4, 2, figsize=(11, 10))
    results = {}
    for row, (label, transformed) in enumerate(transformations):
        series = transformed + noise
        f, estimate = multitaper_spectrum(series - series.mean(), tapers)
        axes[row, 0].plot(t[:250], series[:250], linewidth=0.8)
        axes[row, 0].set_title(label)
        axes[row, 0].set_xlabel("Time")
        axes[row, 1].semilogy(f, estimate)
        for multiple in range(1, 10):
            axes[row, 1].axvline(multiple * f0, color="gray", alpha=0.3, linewidth=0.6)
        axes[row, 1].set_xlabel("Frequency (cycles/sample)")
        axes[row, 1].set_ylabel("Centered-data spectrum")
        results[label] = [float(estimate[np.argmin(np.abs(f - multiple * f0))]) for multiple in range(1, 6)]
    save_figure(fig, "problem2_harmonics.png")
    return results


def analyze_tide_data(path=TIDE_DATA_PATH):
    table = pd.read_csv(path)
    if {"year", "month", "day", "hour", "actual"}.issubset(table.columns):
        time = pd.to_datetime(table[["year", "month", "day", "hour"]], errors="raise")
        values = pd.to_numeric(table["actual"], errors="coerce").to_numpy()
    elif table.shape[1] >= 2:
        time = pd.to_datetime(table.iloc[:, 0], errors="raise")
        values = pd.to_numeric(table.iloc[:, 1], errors="coerce").to_numpy()
    else:
        raise ValueError("Provide date columns and actual, or timestamp and observation columns")
    delta = np.diff(time.astype("int64").to_numpy())
    positive = delta[delta > 0]
    if len(positive) == 0:
        raise ValueError("No positive sampling intervals")
    unique, counts = np.unique(positive, return_counts=True)
    spacing = unique[np.argmax(counts)]
    best_start, best_end, start = 0, 0, 0
    for end in range(len(values)):
        if not np.isfinite(values[end]):
            start = end + 1
        elif end > start and delta[end - 1] != spacing:
            start = end
        if end + 1 - start > best_end - best_start:
            best_start, best_end = start, end + 1
    if best_end - best_start < 3000:
        raise ValueError("No regularly sampled gap-free segment of at least 3000 points")
    values = values[best_start:min(best_end, best_start + 8192)]
    series = values - values.mean()
    n = len(series)
    estimators = {"Periodogram": periodogram(series),
                  "Sine multitaper, K=7": multitaper_spectrum(series, sine_tapers(n, 7)),
                  "DPSS multitaper, NW=4, K=7": multitaper_spectrum(series, dpss(n, 4, Kmax=7, norm=2))}
    samples_per_day = 86400e9 / spacing
    constituents = {"Q1": 13.3986609 / 15, "O1": 13.9430356 / 15,
                    "P1": 14.9589314 / 15, "K1": 15.0410686 / 15,
                    "N2": 28.4397295 / 15, "M2": 28.9841042 / 15,
                    "S2": 2.0, "K2": 30.0821373 / 15,
                    "M4": 57.9682084 / 15, "MS4": 58.9841042 / 15,
                    "M6": 86.9523126 / 15}
    days = np.arange(n) / samples_per_day
    columns = [np.ones(n), days - days.mean()]
    for frequency in constituents.values():
        columns.extend([np.cos(2 * np.pi * frequency * days), np.sin(2 * np.pi * frequency * days)])
    design = np.column_stack(columns)
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    fitted = design @ coefficients
    residual = values - fitted
    explained_fraction = 1 - np.sum(residual ** 2) / np.sum(series ** 2)
    amplitudes = {name: float(np.hypot(coefficients[2 + 2 * j], coefficients[3 + 2 * j]))
                  for j, name in enumerate(constituents)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    show = min(n, int(samples_per_day * 14))
    axes[0, 0].plot(days[:show], values[:show], label="Observed", linewidth=0.9)
    axes[0, 0].plot(days[:show], fitted[:show], label="Harmonic regression", linewidth=0.8)
    axes[0, 0].set_xlabel("Days from segment start")
    axes[0, 0].set_ylabel("Supplied tide value")
    axes[0, 0].legend(fontsize=8)
    for label, (f, estimate) in estimators.items():
        for ax in (axes[0, 1], axes[1, 0]):
            ax.semilogy(f[1:] * samples_per_day, estimate[1:] / samples_per_day, label=label, linewidth=0.75)
    axes[0, 1].set_xlim(0, 6.2)
    axes[1, 0].set_xlim(0.8, 2.15)
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_xlabel("Frequency (cycles/day)")
        ax.set_ylabel("Two-sided density per cycle/day")
    axes[0, 1].legend(fontsize=7)
    for name in ("O1", "K1", "N2", "M2", "S2"):
        axes[1, 0].axvline(constituents[name], color="gray", alpha=0.5, linewidth=0.7)
        axes[1, 0].text(constituents[name], 0.98, name, rotation=90,
                        transform=axes[1, 0].get_xaxis_transform(), ha="right", va="top", fontsize=8)
    f, residual_spectrum = multitaper_spectrum(residual, dpss(n, 4, Kmax=7, norm=2))
    tide_spectrum = estimators["DPSS multitaper, NW=4, K=7"][1]
    axes[1, 1].semilogy(f[1:] * samples_per_day, tide_spectrum[1:] / samples_per_day, label="Before regression")
    axes[1, 1].semilogy(f[1:] * samples_per_day, residual_spectrum[1:] / samples_per_day, label="Residual")
    axes[1, 1].set_xlim(0, 6.2)
    axes[1, 1].set_xlabel("Frequency (cycles/day)")
    axes[1, 1].set_ylabel("DPSS spectral density")
    axes[1, 1].legend(fontsize=8)
    save_figure(fig, "problem2_tide.png")
    peaks = find_peaks(tide_spectrum[1:], distance=8)[0] + 1
    peaks = peaks[np.argsort(tide_spectrum[peaks])[-15:][::-1]]
    raw = estimators["Periodogram"][1]
    peak_matches = {}
    for name, frequency in constituents.items():
        indices = np.flatnonzero(np.abs(f * samples_per_day - frequency) <= 2 * samples_per_day / n)
        index = indices[np.argmax(raw[indices])]
        peak_matches[name] = {"reference_frequency": frequency, "nearest_local_raw_peak": float(f[index] * samples_per_day),
                              "raw_peak_density": float(raw[index] / samples_per_day), "regression_amplitude": amplitudes[name]}
    return {"n": n, "total_rows": len(table), "start": str(time.iloc[best_start]),
            "end": str(time.iloc[best_start + n - 1]), "gap_free_length": best_end - best_start,
            "samples_per_day": float(samples_per_day), "grid_spacing_cycles_per_day": float(samples_per_day / n),
            "multitaper_half_bandwidth_cycles_per_day": float(4 * samples_per_day / n),
            "peaks_cycles_per_day": (f[peaks] * samples_per_day).tolist(),
            "peaks_density": (tide_spectrum[peaks] / samples_per_day).tolist(),
            "explained_fraction": float(explained_fraction), "residual_variance": float(np.mean(residual ** 2)),
            "constituents": peak_matches}


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = {"resolution": compare_peak_resolution(rng),
               "faint_peak": compare_faint_peak_detection(rng),
               "harmonics": explore_harmonics(rng)}
    if TIDE_DATA_PATH.exists():
        results["tide"] = analyze_tide_data()
    else:
        results["tide"] = {"status": "Canvas tide data not supplied"}
    (ROOT / "problem2_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
