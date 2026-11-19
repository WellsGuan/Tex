from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 70141
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
DT = 0.01
SUBSTEPS = 5
N = 400
ENSEMBLE_SIZE = 40
Q = 0.01 * np.eye(3)
R = np.array([[1.0]])
H = np.array([[1.0, 0.0, 0.0]])


def lorenz(state):
    x, y, z = np.moveaxis(np.asarray(state), -1, 0)
    return np.stack([SIGMA * (y - x), x * (RHO - z) - y,
                     x * y - BETA * z], axis=-1)


def rk4(state):
    k1 = lorenz(state)
    k2 = lorenz(state + DT * k1 / 2)
    k3 = lorenz(state + DT * k2 / 2)
    k4 = lorenz(state + DT * k3)
    return state + DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def propagate(ensemble, rng):
    result = np.asarray(ensemble).copy()
    for _ in range(SUBSTEPS):
        result = rk4(result)
    return result + rng.multivariate_normal(np.zeros(3), Q, size=len(result))


def stochastic_update(forecast, observation, rng):
    projected = forecast @ H.T
    anomalies = forecast - forecast.mean(axis=0)
    projected_anomalies = projected - projected.mean(axis=0)
    cross = anomalies.T @ projected_anomalies / (len(forecast) - 1)
    innovation_covariance = projected_anomalies.T @ projected_anomalies / (len(forecast) - 1) + R
    gain = np.linalg.solve(innovation_covariance, cross.T).T
    perturbed = observation + rng.multivariate_normal(np.zeros(1), R, size=len(forecast))
    return forecast + (perturbed - projected) @ gain.T, gain


def main():
    output = ROOT / "project_setup"
    output.mkdir(parents=True, exist_ok=True)
    (ROOT / "png").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    state = np.array([1.0, 1.0, 1.0])
    for _ in range(500):
        state = rk4(state)
    states = np.empty((N, 3))
    for t in range(N):
        state = propagate(state[None, :], rng)[0]
        states[t] = state
    observations = states @ H.T + rng.multivariate_normal(np.zeros(1), R, size=N)
    initial = rng.normal(size=(ENSEMBLE_SIZE, 3)) * 3 + np.array([0.0, 1.0, 20.0])
    forecast = propagate(initial, rng)
    analysis, gain = stochastic_update(forecast, observations[0], rng)
    if not np.isfinite(states).all() or not np.isfinite(analysis).all():
        raise RuntimeError("Nonfinite values in the setup check")
    covariance = np.cov(forecast, rowvar=False)
    if not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise RuntimeError("Invalid sample covariance")
    config = {"seed": SEED, "sigma": SIGMA, "rho": RHO, "beta": BETA,
              "rk4_step": DT, "observation_interval": DT * SUBSTEPS,
              "observations": N, "observed_coordinate": "x", "H": H.tolist(),
              "Q": Q.tolist(), "R": R.tolist(), "setup_ensemble_size": ENSEMBLE_SIZE,
              "planned_ensemble_sizes": [20, 40, 80, 160],
              "completed": ["Lorenz simulator", "noisy partial observations",
                            "ensemble propagation", "sample covariance", "one stochastic update"],
              "pending": ["full assimilation experiment", "rolling forecast evaluation",
                          "RMSE and CRPS", "ensemble size comparison", "inflation experiments"]}
    (output / "setup_summary.json").write_text(json.dumps(config, indent=2))
    np.savez_compressed(output / "setup_data.npz", states=states, observations=observations,
                        forecast_ensemble=forecast, analysis_ensemble=analysis, gain=gain)
    time = DT * SUBSTEPS * np.arange(1, N + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(time, states[:, j], lw=0.8, label="Simulated latent state")
        ax.set_ylabel(["x", "y", "z"][j])
        if j == 0:
            ax.scatter(time[::5], observations[::5, 0], s=7, color="tab:red", alpha=0.6,
                       label="Noisy partial observations")
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("Lorenz model time")
    fig.tight_layout()
    fig.savefig(ROOT / "png" / "project_setup.png", dpi=170)
    plt.close(fig)
    print("Basic Lorenz/EnKF setup completed; full experiments are still pending.")


if __name__ == "__main__":
    main()
