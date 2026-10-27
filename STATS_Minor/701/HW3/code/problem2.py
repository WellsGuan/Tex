from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "png"
SEED = 70103

TIDE_DATA_PATH = DATA_DIR / "tide.csv"


def circulant_embedding(acf, rng=None):
    raise NotImplementedError


def periodogram(series):
    raise NotImplementedError


def multitaper_spectrum(series, tapers):
    raise NotImplementedError


def compare_peak_resolution(rng):
    raise NotImplementedError


def compare_faint_peak_detection(rng):
    raise NotImplementedError


def explore_harmonics(rng):
    raise NotImplementedError


def analyze_tide_data(path=TIDE_DATA_PATH):
    raise NotImplementedError


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    raise NotImplementedError


if __name__ == "__main__":
    main()
