from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "png"
SEED = 70103

NAO_DATA_PATH = DATA_DIR / "nao.csv"
N = 1024


def load_nao(path=NAO_DATA_PATH, n=N):
    raise NotImplementedError


def spectral_density(frequencies, parameters):
    raise NotImplementedError


def whittle_likelihood(dft, parameters):
    raise NotImplementedError


def fit_spectrum(dft):
    raise NotImplementedError


def spectrum_to_acf(parameters, n=N):
    raise NotImplementedError


def gaussian_log_likelihood(series, acf):
    raise NotImplementedError


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError


if __name__ == "__main__":
    main()
