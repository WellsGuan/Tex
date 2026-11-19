from pathlib import Path
import argparse
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov, toeplitz, cholesky, solve_triangular
from scipy.optimize import least_squares, minimize
from scipy.signal import find_peaks, periodogram, lfilter
from scipy.stats import probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.tools import constrain_stationary_univariate, unconstrain_stationary_univariate
from statsmodels.tsa.arima_process import arma_acovf

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "mystery_data.csv"
FIGURE_DIR = ROOT / "png"
RESULT_DIR = ROOT / "results"
HORIZON = 365
SEED = 70104


def load_data(path=DATA_PATH, column=None, header=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Place the Canvas data at {path}")
    table = pd.read_csv(path, header=header)
    if column is None:
        if table.shape[1] != 1:
            raise ValueError("Specify --column when the CSV has multiple columns")
        series = table.iloc[:, 0]
    else:
        key = int(column) if header is None else column
        series = table[key]
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    if len(values) <= HORIZON + 60 or not np.isfinite(values).all():
        raise ValueError("Expected more than 425 finite daily observations")
    return values


def design(time, frequencies):
    time = np.asarray(time)
    columns = [np.ones(len(time))]
    for frequency in frequencies:
        columns.extend([np.cos(2 * np.pi * frequency * time),
                        np.sin(2 * np.pi * frequency * time)])
    return np.column_stack(columns)


def frequency_candidates(values):
    n = len(values)
    frequencies, power = periodogram(values, detrend="constant")
    peaks, _ = find_peaks(power)
    peaks = [j for j in peaks if 1 / n <= frequencies[j] < 0.5 - 1 / n]
    chosen = []
    for index in sorted(peaks, key=lambda j: power[j], reverse=True):
        if all(abs(frequencies[index] - f) >= 2 / n for f in chosen):
            chosen.append(float(frequencies[index]))
        if len(chosen) == 3:
            break
    if not chosen:
        raise ValueError("No interior spectral peaks were found")
    return chosen


def refine_frequencies(values, initial):
    time = np.arange(1, len(values) + 1)
    initial = np.asarray(initial)
    radius = 0.49 / len(values)
    def residual(frequencies):
        matrix = design(time, frequencies)
        beta = np.linalg.lstsq(matrix, values, rcond=None)[0]
        return values - matrix @ beta
    fit = least_squares(residual, initial,
                        bounds=(np.maximum(initial - radius, 1e-8),
                                np.minimum(initial + radius, 0.5 - 1e-8)))
    return np.sort(fit.x)


def fit_candidate(values, frequencies, order, previous=None):
    matrix = design(np.arange(1, len(values) + 1), frequencies)
    n = len(values)
    def profile(z, return_fit=False):
        coefficients = constrain_stationary_univariate(z) if order else np.array([])
        try:
            if order:
                covariance = toeplitz(arma_acovf(np.r_[1, -coefficients], [1], nobs=order))
                factor = cholesky(covariance, lower=True)
                initial_y = solve_triangular(factor, values[:order], lower=True)
                initial_x = solve_triangular(factor, matrix[:order], lower=True)
                tail_y = lfilter(np.r_[1, -coefficients], [1], values)[order:]
                tail_x = lfilter(np.r_[1, -coefficients], [1], matrix, axis=0)[order:]
                whitened_y = np.r_[initial_y, tail_y]
                whitened_x = np.vstack([initial_x, tail_x])
                logdet = 2 * np.log(np.diag(factor)).sum()
            else:
                whitened_y, whitened_x, logdet = values, matrix, 0.0
            beta = np.linalg.lstsq(whitened_x, whitened_y, rcond=None)[0]
            residual = whitened_y - whitened_x @ beta
            sigma2 = float(residual @ residual / n)
            objective = 0.5 * (n * (np.log(2 * np.pi * sigma2) + 1) + logdet)
            return (objective, beta, coefficients, sigma2) if return_fit else objective
        except (ValueError, np.linalg.LinAlgError):
            return 1e100
    converged = True
    if order:
        residual = values - matrix @ np.linalg.lstsq(matrix, values, rcond=None)[0]
        initial, _ = yule_walker(residual, order=order, method="mle")
        starts = [unconstrain_stationary_univariate(initial)]
        if previous is not None:
            nested = np.r_[previous["result"].arparams, 0.0]
            starts.append(unconstrain_stationary_univariate(nested))
        attempts = []
        for start in starts:
            fit = minimize(profile, start, method="L-BFGS-B", bounds=[(-100, 100)] * order,
                           options={"maxiter": 350, "ftol": 1e-11, "gtol": 1e-5})
            if not fit.success:
                fit = minimize(profile, fit.x, method="Powell", bounds=[(-100, 100)] * order,
                               options={"maxiter": 300, "ftol": 1e-9})
            if fit.success and np.isfinite(fit.fun):
                attempts.append(fit)
        if not attempts:
            raise RuntimeError("Likelihood optimization did not converge")
        fit = min(attempts, key=lambda result: result.fun)
        converged = bool(fit.success)
        z = fit.x
    else:
        z = np.array([])
    if not converged:
        raise RuntimeError("Likelihood optimization did not converge")
    objective, beta, coefficients, sigma2 = profile(z, True)
    model = SARIMAX(values, exog=matrix, order=(order, 0, 0), trend="n",
                    enforce_stationarity=True)
    result = model.filter(np.r_[beta, coefficients, sigma2], cov_type="none")
    if not np.isfinite(result.llf) or not np.isclose(result.llf, -objective, atol=0.02, rtol=1e-6):
        raise RuntimeError("Profile and state-space likelihoods disagree")
    count = len(result.params) + len(frequencies)
    return {"result": result, "frequencies": np.asarray(frequencies), "order": order,
            "bic": float(-2 * result.llf + count * np.log(n)),
            "aic": float(-2 * result.llf + 2 * count), "parameters": count,
            "optimizer_converged": converged}


def select_models(values, max_ar=12):
    best = {}
    candidates = []
    peaks = frequency_candidates(values)
    frequency_sets = [refine_frequencies(values, peaks[:k])
                      for k in range(1, len(peaks) + 1)]
    for name in ["harmonic", "stationary"]:
        for frequencies in frequency_sets if name == "harmonic" else [np.array([])]:
            previous = None
            for order in range(min(max_ar, max(0, len(values) // 20)) + 1):
                try:
                    fitted = fit_candidate(values, frequencies, order, previous)
                    previous = fitted
                    candidates.append({"model": name, "p": order,
                                       "K": len(frequencies), "bic": fitted["bic"],
                                       "aic": fitted["aic"],
                                       "negative_log_likelihood": -float(fitted["result"].llf)})
                    if name not in best or fitted["bic"] < best[name]["bic"]:
                        best[name] = fitted
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    continue
        if name not in best:
            raise RuntimeError(f"No converged stationary fit for {name}")
        print(f"Selected {name}: K={len(best[name]['frequencies'])}, p={best[name]['order']}", flush=True)
    return best, candidates


def forecast(fitted, values, horizon=HORIZON, rng=None, repetitions=20):
    rng = np.random.default_rng(SEED) if rng is None else rng
    result = fitted["result"]
    frequencies = fitted["frequencies"]
    p = fitted["order"]
    n = len(values)
    beta = np.asarray(result.params[:1 + 2 * len(frequencies)])
    past_mean = design(np.arange(1, n + 1), frequencies) @ beta
    future_mean = design(np.arange(n + 1, n + horizon + 1), frequencies) @ beta
    sigma2 = float(dict(zip(result.param_names, result.params))["sigma2"])
    coefficients = np.asarray(result.arparams)
    residual = values - past_mean
    state = residual[-p:][::-1].copy() if p else np.array([])
    mean = future_mean.copy()
    states = np.tile(state, (repetitions, 1))
    paths = np.tile(future_mean, (repetitions, 1))
    for j in range(horizon):
        conditional_mean = float(coefficients @ state) if p else 0.0
        mean[j] += conditional_mean
        draw = states @ coefficients if p else np.zeros(repetitions)
        draw += rng.normal(scale=np.sqrt(sigma2), size=repetitions)
        paths[:, j] += draw
        if p:
            state = np.r_[conditional_mean, state[:-1]]
            states = np.column_stack([draw, states[:, :-1]])
    psi = np.zeros(horizon)
    psi[0] = 1
    for j in range(1, horizon):
        psi[j] = sum(coefficients[i - 1] * psi[j - i]
                     for i in range(1, min(p, j) + 1))
    variance = sigma2 * np.cumsum(psi ** 2)
    reference = result.get_forecast(horizon, exog=design(np.arange(n + 1, n + horizon + 1), frequencies))
    if not np.allclose(mean, reference.predicted_mean, rtol=1e-4, atol=1e-5):
        raise RuntimeError("Direct and state-space forecast means disagree")
    if not np.allclose(variance, reference.var_pred_mean, rtol=1e-4, atol=1e-5):
        raise RuntimeError("Direct and state-space forecast variances disagree")
    return {"mean": mean, "variance": variance, "lower": mean - 1.96 * np.sqrt(variance),
            "upper": mean + 1.96 * np.sqrt(variance), "paths": paths}


def stationary_path(fitted, n, rng):
    result = fitted["result"]
    p = fitted["order"]
    coefficients = np.asarray(result.arparams)
    sigma2 = float(dict(zip(result.param_names, result.params))["sigma2"])
    if p:
        transition = np.zeros((p, p))
        transition[0] = coefficients
        transition[1:, :-1] = np.eye(p - 1)
        noise = np.zeros((p, p))
        noise[0, 0] = sigma2
        covariance = solve_discrete_lyapunov(transition, noise)
        state = rng.multivariate_normal(np.zeros(p), covariance)
    path = np.empty(n)
    for j in range(n):
        path[j] = (coefficients @ state if p else 0) + rng.normal(scale=np.sqrt(sigma2))
        if p:
            state = np.r_[path[j], state[:-1]]
    beta = np.asarray(result.params[:1 + 2 * len(fitted["frequencies"])])
    return path + design(np.arange(1, n + 1), fitted["frequencies"]) @ beta


def summary(fitted, values):
    result = fitted["result"]
    errors = np.asarray(result.filter_results.standardized_forecasts_error[0])
    errors = errors[max(fitted["order"], 1):]
    lags = sorted(set(max(fitted["order"] + 1, lag) for lag in [10, 20, 40]))
    lags = [lag for lag in lags if lag < len(errors) // 4]
    tests = acorr_ljungbox(errors, lags=lags, model_df=fitted["order"], return_df=True)
    return {"p": fitted["order"], "K": len(fitted["frequencies"]),
            "frequencies_cycles_per_day": fitted["frequencies"].tolist(),
            "periods_days": (1 / fitted["frequencies"]).tolist(),
            "parameters": dict(zip(result.param_names, map(float, result.params))),
            "negative_log_likelihood": -float(result.llf), "bic": fitted["bic"],
            "aic": fitted["aic"], "converged": fitted["optimizer_converged"],
            "ljung_box": {str(lag): float(row.lb_pvalue) for lag, row in tests.iterrows()}}


def diagnostics(models, values, prefix, rng):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, (name, fitted) in enumerate(models.items()):
        errors = np.asarray(fitted["result"].filter_results.standardized_forecasts_error[0])
        errors = errors[max(1, fitted["order"]):]
        axes[row, 0].plot(errors, lw=0.6)
        axes[row, 0].set_title(f"{name}: standardized innovations")
        correlation = acf(errors, nlags=min(60, len(errors) // 4), fft=True)
        axes[row, 1].stem(np.arange(len(correlation)), correlation, markerfmt=".")
        axes[row, 1].axhline(1.96 / np.sqrt(len(errors)), color="gray", ls="--")
        axes[row, 1].axhline(-1.96 / np.sqrt(len(errors)), color="gray", ls="--")
        axes[row, 1].set_title("Innovation ACF")
        probplot(errors, dist="norm", plot=axes[row, 2])
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{prefix}_diagnostics.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    for ax, (name, fitted) in zip(axes, models.items()):
        simulated = stationary_path(fitted, len(values), rng)
        ax.plot(values, lw=0.7, label="Observed")
        ax.plot(simulated, alpha=0.65, lw=0.7, label="Independent fitted-model draw")
        ax.set_title(name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{prefix}_simulations.png", dpi=170)
    plt.close(fig)


def forecast_plot(models, values, prefix, rng, holdout=None):
    predictions = {}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, (name, fitted) in zip(axes, models.items()):
        pred = forecast(fitted, values, rng=rng)
        predictions[name] = pred
        n = len(values)
        start = max(0, n - 730)
        time = np.arange(n + 1, n + HORIZON + 1)
        ax.plot(np.arange(start + 1, n + 1), values[start:], color="black", lw=0.7, label="History")
        for path in pred["paths"][:10]:
            ax.plot(time, path, color="tab:blue", alpha=0.12, lw=0.6)
        ax.fill_between(time, pred["lower"], pred["upper"], alpha=0.25, label="95% pointwise prediction interval")
        ax.plot(time, pred["mean"], label="Conditional mean")
        if holdout is not None:
            ax.plot(time, holdout, color="tab:red", lw=0.8, label="Holdout")
        ax.axvline(n, color="gray", ls="--")
        ax.set_title(name)
        ax.legend(fontsize=8)
        pd.DataFrame({"time": time, "mean": pred["mean"], "variance": pred["variance"],
                      "lower": pred["lower"], "upper": pred["upper"]}).to_csv(
                          RESULT_DIR / f"{prefix}_{name}_forecast.csv", index=False)
    axes[-1].set_xlabel("Day")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{prefix}_forecasts.png", dpi=170)
    plt.close(fig)
    return predictions


def metrics(actual, prediction):
    error = actual - prediction["mean"]
    lower, upper = prediction["lower"], prediction["upper"]
    score = upper - lower + 40 * (lower - actual) * (actual < lower) + 40 * (actual - upper) * (actual > upper)
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(abs(error))),
            "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
            "mean_interval_width": float(np.mean(upper - lower)), "interval_score": float(np.mean(score))}


def write_tex(report):
    for prefix in ["full", "holdout"]:
        records = report[prefix]
        lines = [r"\begin{center}\begin{tabular}{lrrrr}",
                 r"Model & $K$ & $p$ & $-\ell$ & BIC\\\hline"]
        for name, record in records.items():
            lines.append(f"{name} & {record['K']} & {record['p']} & {record['negative_log_likelihood']:.3f} & {record['bic']:.3f}" + r"\\")
        lines.append(r"\end{tabular}\end{center}")
        for name, record in records.items():
            lines.append(r"\noindent\textbf{" + name + r".} ")
            if record["K"]:
                lines.append("The fitted periods in days are $" + ", ".join(f"{x:.5g}" for x in record["periods_days"]) + "$. ")
            lines.append(r"The fitted parameters are listed in the order")
            lines.append(r"\[\mu,A_1,B_1,\ldots,\phi_1,\ldots,\sigma_Z^2.\]")
            lines.append("Their numerical values are")
            lines.append(r"\[\begin{gathered}" + r"\\".join(
                ",\ ".join(f"{x:.5g}" for x in list(record["parameters"].values())[i:i+6])
                for i in range(0, len(record["parameters"]), 6)) + r"\end{gathered}\]")
            lines.append("Innovation Ljung--Box $p$-values (lag: value) are " +
                         ", ".join(f"{lag}: {value:.4g}" for lag, value in record["ljung_box"].items()) + ".\\par")
        preferred = min(records, key=lambda name: records[name]["bic"])
        lines.append("For this fitted candidate set, BIC favors the " + preferred + " model. This is an in-sample comparison.\\par")
        (RESULT_DIR / f"{prefix}_summary.tex").write_text("\n".join(lines))
    lines = [r"\begin{center}\begin{tabular}{lrrrrr}",
             r"Model & RMSE & MAE & Coverage & Width & Interval score\\\hline"]
    for name, record in report["metrics"].items():
        lines.append(name + " & " + " & ".join(f"{record[key]:.4f}" for key in
                     ["rmse", "mae", "coverage", "mean_interval_width", "interval_score"]) + r"\\")
    lines.append(r"\end{tabular}\end{center}")
    best_error = min(report["metrics"], key=lambda name: report["metrics"][name]["rmse"])
    best_interval = min(report["metrics"], key=lambda name: report["metrics"][name]["interval_score"])
    lines.append(f"The {best_error} model has the smaller holdout RMSE, and the {best_interval} model has the smaller interval score. ")
    lines.append("I use the holdout results to assess forecasting; a single holdout year cannot establish that either model is the true mechanism.")
    (RESULT_DIR / "holdout_metrics.tex").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="STAT 701 HW4 model comparison")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--column")
    parser.add_argument("--header", action="store_true")
    parser.add_argument("--max-ar", type=int, default=12)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    values = load_data(args.data, args.column, None if not args.header else "infer")
    if args.check_data:
        print(f"{len(values)} daily observations; {len(values) - HORIZON} training and {HORIZON} holdout points")
        return
    if args.max_ar < 0:
        parser.error("--max-ar must be nonnegative")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    report = {"n": len(values), "seed": SEED, "horizon": HORIZON, "max_ar": args.max_ar}
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    axes[0].plot(np.arange(1, len(values) + 1), values, lw=0.7)
    axes[0].set_xlabel("Day")
    frequency, power = periodogram(values, detrend="constant")
    axes[1].semilogy(frequency[1:], np.maximum(power[1:], np.finfo(float).tiny))
    axes[1].set_xlabel("Frequency (cycles per day)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "data_overview.png", dpi=170)
    plt.close(fig)
    for prefix, series in [("full", values), ("holdout", values[:-HORIZON])]:
        models, candidates = select_models(series, args.max_ar)
        report[prefix] = {name: summary(fitted, series) for name, fitted in models.items()}
        pd.DataFrame(candidates).to_csv(RESULT_DIR / f"{prefix}_candidates.csv", index=False)
        diagnostics(models, series, prefix, rng)
        predictions = forecast_plot(models, series, prefix, rng,
                                    values[-HORIZON:] if prefix == "holdout" else None)
        if prefix == "holdout":
            report["metrics"] = {name: metrics(values[-HORIZON:], pred) for name, pred in predictions.items()}
    (RESULT_DIR / "problem2_results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    write_tex(report)
    print("Figures, forecasts, summaries and TeX tables written.")


if __name__ == "__main__":
    main()
