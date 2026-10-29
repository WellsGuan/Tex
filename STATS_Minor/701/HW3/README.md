# STAT 701 HW3

`HW3.tex` contains the original questions, worked solutions, figures, and all three Python scripts in its appendix. Problem 2(4) now uses the supplied hourly tide gauge observations. No synthetic observations substitute for that dataset.

## Reproduce results

Run from the HW3 directory in a Python environment with NumPy, SciPy, pandas, Matplotlib, and statsmodels:

```bash
python3 code/problem1.py
python3 code/problem2.py
python3 code/problem3.py
```

For noninteractive execution, set `MPLBACKEND=Agg`. The scripts use reproducible random seeds, write figures to `png/`, and save numerical summaries in `problem1_results.json`, `problem2_results.json`, and `problem3_results.json`. The first script uses a batched FFT equivalent to the direct DFT shown in the answer and verifies this equivalence numerically. Flat and DPSS weights use identical ARMA sample paths.

`data/nao.csv` is the unchanged file from HW2; its value column is named `aao_index_cdas`, and this is disclosed in the answer. Problem 3 uses its first 1024 daily measurements.

## Tide observations

The supplied `data/tidegauge.csv` has columns `year`, `month`, `day`, `hour`, and `actual`. All 119266 rows have consecutive hourly timestamps and finite values. The analysis uses the first 8192 points, from 1980-10-01 13:00 through 1981-09-07 20:00, with no interpolation. The loader also accepts a two-column timestamp/observation CSV. It requires a regularly sampled gap-free segment of at least 3000 points and uses at most 8192.

Run `code/problem2.py` to reproduce `png/problem2_tide.png`, the harmonic regression and its residual spectrum, and the numerical peak summaries in `problem2_results.json`. Reference constituent speeds come from NOAA's harmonic-constituent table (https://tidesandcurrents.noaa.gov/harcon.html?id=8721120); only the universal frequencies, not station-specific amplitudes, are used.

The newly supplied NAO file is byte-identical to `data/nao.csv`, so the existing Problem 3 results are unchanged.

## Numerical conventions

All spectra are two-sided densities on frequencies measured in cycles per observation, even when only the nonnegative half is displayed. For the NAO file an observation is one day. A unit-norm taper is used for each estimate.

The circulant simulator checks that its symmetric embedding is positive semidefinite. Valid Toeplitz input does not ensure that this particular circulant embedding is positive semidefinite; substantive negative eigenvalues cause an error instead of covariance distortion. Only numerical roundoff is clipped. The examples included here have valid embeddings.

The previous SARIMA model has a seasonal unit root. The likelihood comparison is made on the same 659 seasonal differences for both models, rather than comparing likelihoods of different observations. Rounded HW2 coefficients are reused without refitting.

## LaTeX

The existing ElegantPaper layout is retained. The local class adds a `nobib=true` option because this document has no bibliography or citations; this avoids an unnecessary external Biber dependency. The document compiles with the built-in editor and also supports the usual `xelatex HW3.tex` workflow. The external Python listings update when the source scripts change.

No Git commit or push is performed by these scripts.
