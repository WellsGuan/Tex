from pathlib import Path
from statsmodels.tsa.arima_process import arma_generate_sample

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


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    samples = np.stack([arma_generate_sample(
    ar=[1, -0.6],
    ma=[1, 0.3],
    nsample=N,
    scale=1,
    burnin=500,
    distrvs=rng.standard_normal,
    ) for i in range(M)])

    #fig, ax = plt.subplots()

    #ax = draw_paths(sample[:10], ax = ax)

    #ax.set_title("ARMA sample paths")
    #ax.set_xlabel("Time")
    #ax.set_ylabel("Value")
    #ax.figure.savefig(FIGURE_DIR / "sample_paths.png", bbox_inches="tight")

    #plt.show()

    dft = np.stack([simulated_paths(sample) for sample in samples])

    empirical_covariance = empirical_dft_covariance(dft)

    fig, ax = plt.subplots()

    ax = draw_matrix(empirical_covariance, ax = ax)

    ax.set_title("ARMA DFT Empirical Covaiance Matrix")
    ax.figure.savefig(FIGURE_DIR / "empirical_covariance_matrix.png", bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    main()
