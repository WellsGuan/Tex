from pathlib import Path
from statsmodels.tsa.arima_process import arma_generate_sample
from scipy.signal.windows import dpss

import json
from scipy.linalg import toeplitz

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "png"
SEED = 70103

N = 512
M = 10 * N
NW = 10


def simulated_paths(times_series, alpha = 10, rng=None):
    n = times_series.shape[0]
    weights = np.ones(n)/(n ** 0.5)
    frequencies = np.arange(-np.floor((n-1)/2),np.floor(n/2)+1)/n
    dft = np.stack([weighted_dft(times_series, weights, frequency) for frequency in frequencies])
    return dft

def simulated_paths_2(times_series, alpha = 10, rng=None):
    n = times_series.shape[0]
    weights = dpss(n, NW=10, Kmax=1, norm=2)[0]
    frequencies = np.arange(-np.floor((n-1)/2),np.floor(n/2)+1)/n
    dft = np.stack([weighted_dft(times_series, weights, frequency) for frequency in frequencies])
    return dft

def draw_paths(paths, x=None, component="real", ax=None):
    paths = np.asarray(paths)
    if paths.ndim != 2 or 0 in paths.shape:
        raise ValueError("paths must be a nonempty (m, n) array")
    if component == "real":
        values = paths.real
    elif component == "imag":
        values = paths.imag
    elif component == "abs":
        values = np.abs(paths)
    else:
        raise ValueError("component must be 'real', 'imag', or 'abs'")

    m, n = paths.shape
    x = np.arange(1, n + 1) if x is None else np.asarray(x)
    if x.ndim != 1 or x.size != n:
        raise ValueError("x must be a one-dimensional array of length n")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, m))
    lines = ax.plot(x, values.T, linewidth=0.9, alpha=0.8 if m <= 20 else 0.25)
    for index, (line, color) in enumerate(zip(lines, colors), start=1):
        line.set_color(color)
        line.set_label(f"Series {index}")
    ax.set_xlabel("Index")
    ax.set_ylabel({"real": "Real part", "imag": "Imaginary part", "abs": "Magnitude"}[component])
    ax.grid(alpha=0.2)
    if m <= 10:
        ax.legend()
    return ax



def draw_matrix(matrix, component="abs", ax=None, cmap=None, colorbar=True, gamma=0.4):
    from matplotlib.colors import PowerNorm

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a positive finite number")
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("matrix must be a nonempty two-dimensional array")
    if component == "abs":
        values = np.abs(matrix)
    elif component == "real":
        values = matrix.real
    elif component == "imag":
        values = matrix.imag
    else:
        raise ValueError("component must be 'abs', 'real', or 'imag'")
    if not np.isfinite(values).all():
        raise ValueError("matrix values must be finite")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    options = {}
    if component == "abs":
        cmap = "viridis" if cmap is None else cmap
        limit = float(np.max(values)) or 1.0
        options["norm"] = PowerNorm(gamma=gamma, vmin=0, vmax=limit)
    else:
        cmap = "RdBu_r" if cmap is None else cmap
        limit = float(np.max(np.abs(values))) or 1.0
        options.update(vmin=-limit, vmax=limit)
    image = ax.imshow(values, origin="upper", interpolation="nearest",
                      aspect="auto", cmap=cmap, **options)
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax)
        bar.set_label({"abs": "Magnitude", "real": "Real part", "imag": "Imaginary part"}[component])
    return ax


def weighted_dft(paths, weights, frequency):
    time = np.arange(1,paths.shape[0]+1)
    exp_term = np.exp( - 1j * 2 * np.pi * frequency * time)
    return np.sum(weights * paths * exp_term)


def empirical_dft_covariance(coefficients):
    dft_covaiance = np.outer(coefficients[0], coefficients[0].conj())
    for coefficient in coefficients[1:]:
        dft_covaiance += np.outer(coefficient, coefficient.conj())
    return dft_covaiance


def batch_dft(samples, weights):
    n = samples.shape[1]
    k = np.arange(-((n - 1) // 2), n // 2 + 1)
    frequencies = k / n
    coefficients = np.fft.fft(samples * weights, axis=1)[:, k % n]
    return frequencies, coefficients * np.exp(-2j * np.pi * frequencies)


def covariance_correlation(matrix):
    standard_deviations = np.sqrt(np.diag(matrix).real)
    return matrix / np.outer(standard_deviations, standard_deviations)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    samples = np.stack([arma_generate_sample(ar=[1, -0.6], ma=[1, 0.3], nsample=N,
                        burnin=500, distrvs=rng.standard_normal) for _ in range(M)])
    ax = draw_paths(samples[:10])
    ax.set_title("ARMA(1,1) sample paths")
    ax.set_xlabel("Time")
    ax.figure.savefig(FIGURE_DIR / "sample_paths.png", dpi=170, bbox_inches="tight")
    plt.close(ax.figure)
    weights = {"flat": np.ones(N) / np.sqrt(N), "dpss": dpss(N, NW=NW, Kmax=1, norm=2)[0]}
    h = np.arange(N)
    acf = np.empty(N)
    acf[0] = (1 + 0.3 ** 2 + 2 * 0.6 * 0.3) / (1 - 0.6 ** 2)
    acf[1:] = (0.6 + 0.3) * (1 + 0.6 * 0.3) / (1 - 0.6 ** 2) * 0.6 ** (h[1:] - 1)
    results = {"n": N, "m": M, "ar": 0.6, "ma": 0.3, "gamma0": float(acf[0]), "gamma1": float(acf[1])}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for row, (name, taper) in enumerate(weights.items()):
        frequencies, coefficients = batch_dft(samples, taper)
        direct = np.array([weighted_dft(samples[0], taper, frequency) for frequency in frequencies])
        np.testing.assert_allclose(coefficients[0], direct, atol=1e-11)
        covariance = empirical_dft_covariance(coefficients) / M
        kernel = taper[None, :] * np.exp(-2j * np.pi * frequencies[:, None] * np.arange(1, N + 1))
        expected = kernel @ toeplitz(acf) @ kernel.conj().T
        empirical_correlation = covariance_correlation(covariance)
        theoretical_correlation = covariance_correlation(expected)
        ax = draw_matrix(covariance)
        ax.set_title(f"ARMA(1,1) DFT covariance: {name}")
        filename = "empirical_covariance_matrix.png" if name == "flat" else "empirical_covariance_matrix_dpss.png"
        ax.figure.savefig(FIGURE_DIR / filename, dpi=170, bbox_inches="tight")
        plt.close(ax.figure)
        for column, (matrix, label) in enumerate(((empirical_correlation, "Empirical"), (theoretical_correlation, "Theoretical"))):
            image = axes[row, column].imshow(np.abs(matrix), origin="upper", vmin=0, vmax=1, cmap="viridis")
            axes[row, column].set_title(f"{name}: {label} correlation magnitude")
            axes[row, column].set_xlabel("Frequency index")
            axes[row, column].set_ylabel("Frequency index")
            fig.colorbar(image, ax=axes[row, column])
        off_diagonal = expected - np.diag(np.diag(expected))
        results[name] = {"relative_diagonal_approximation_error": float(np.linalg.norm(off_diagonal) / np.linalg.norm(expected)),
                         "mean_adjacent_correlation": float(np.mean(np.abs(np.diag(theoretical_correlation, 1)))),
                         "empirical_mean_adjacent_correlation": float(np.mean(np.abs(np.diag(empirical_correlation, 1)))),
                         "relative_simulation_error": float(np.linalg.norm(covariance - expected) / np.linalg.norm(expected))}
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "problem1_correlation_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    (ROOT / "problem1_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
