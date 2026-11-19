# STAT 701 HW4

`HW4.tex` contains the original questions, completed answers, numerical tables, figures and all Python source in its appendix. It follows the English style and ElegantPaper format of HW3.

## Reproduce the homework

From the HW4 directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/project_setup.py
python code/problem2.py
xelatex HW4.tex
xelatex HW4.tex
```

Open `HW4.code-workspace` in VS Code and select `.venv/bin/python`. Its LaTeX Workshop recipe compiles `HW4.tex` with XeLaTeX. The built-in document editor also supports this project.

For faster numerical execution when BLAS would otherwise start too many threads:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python code/problem2.py
```

## Input and output

- `HW4_assignment.pdf`: unchanged homework handout.
- `data/mystery_data.csv`: unchanged supplied Canvas data, 10000 finite values without a header. The default loader preserves its first observation.
- `code/problem2.py`: the complete model comparison and forecasting pipeline, seed 70104.
- `png/`: observation plots, innovation diagnostics, fitted-model simulations and forecast comparisons.
- `results/`: fitted parameters, model-selection tables, forecast CSVs, holdout metrics and generated TeX summaries.
- `code/project_setup.py`: initial Lorenz/EnKF setup, seed 70141.
- `project_setup/`: original project brief, initial setup configuration and synthetic setup arrays.

The actual data use daily timesteps as required by the homework. Harmonic frequency is in cycles per day; angular frequency equals 2*pi times that value. The loader defaults to a headerless CSV. Use `--header --column value` only for a different CSV that actually has a header; `--data PATH` selects another file. `--check-data` verifies the shape and finite values without fitting. The default AR search is 0 through 12 (`--max-ar` changes it).

## Models and computation

Both model families use stationary Gaussian AR residuals. The harmonic mean compares up to three separated periodogram-peak sets; frequencies are locally refined by profile least squares. The exact stationary Gaussian likelihood is maximized conditional on the selected frequencies. The first p observations use their stationary covariance and the rest use AR innovations. The mean and innovation variance are profiled out, and the computed likelihood is checked against the state-space likelihood. A stationary parameterization is used with Yule-Walker and nested lower-order initializations. No global joint frequency MLE is claimed. BIC counts each selected frequency as a fitted parameter.

Model selection is repeated on the first 9635 observations for the holdout comparison; the last 365 are never used to fit or select that training-only model. Forecasts are checked against state-space predictions. Conditional paths use fresh future innovations. Independent simulations initialize the AR state from its stationary covariance, without an approximate burn-in.

The constant-mean model selects the upper AR order limit and retains significant innovation correlations. This limitation is explicitly discussed; the harmonic preference is for this candidate set and holdout year, not a claim of a uniquely true model. Both empirical interval coverages are below 95%. Intervals are pointwise and condition on fitted parameters; they omit parameter and model-selection uncertainty.

The handout's holdout indices include 366 points and overlap the training endpoint. The original question is preserved. The solution and code explicitly use the final 365 distinct observations instead.

## Project progress

Option 1 is the selected project. The initial setup chooses its Lorenz physical branch: RK4 integration, partial observation H=(1,0,0), Gaussian state/measurement noise, ensemble propagation and one stochastic EnKF update. Those setup operations have actually been run and are saved in `project_setup/`. The project-progress answer does not claim completion of sequential filtering, rolling forecasts, RMSE/CRPS benchmarking or inflation experiments. They are the stated next steps.

The project simulations are separate from the supplied mystery data and are not used as substitute homework observations. Both Python files contain no code comments. Source listings in the appendix update on recompilation.
