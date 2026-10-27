from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "png"
SEED = 70103

N = 512
M = 10 * N
NW = 10


def simulate_paths(n=N, m=M, rng=None):
    raise NotImplementedError


def weighted_dft(paths, weights):
    raise NotImplementedError


def empirical_dft_covariance(coefficients):
    raise NotImplementedError


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    raise NotImplementedError


if __name__ == "__main__":
    main()
