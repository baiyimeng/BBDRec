"""Forward Markov paths for trajectory visualizations (no model training).

Each step draws a new independent innovation. Schedules must come from the
model/run configuration; this module does not select visualization-only scales.
"""
import numpy as np


def trajectory_rng(seed, dataset_index, model_index, sample_index):
    """Reproducible, distinct random streams for every model/sample pair."""
    return np.random.default_rng(np.random.SeedSequence(
        [seed, dataset_index, model_index, sample_index]
    ))


def _state(value):
    value = np.asarray(value, dtype=np.float64)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError("States must be finite and nonempty")
    return value.copy()


def sample_ddpm_path(x0, betas, rng):
    """x_t = sqrt(1-beta_t) x_(t-1) + sqrt(beta_t) epsilon_t."""
    state = _state(x0)
    betas = np.asarray(betas, dtype=np.float64)
    if betas.ndim != 1 or not len(betas) or not np.isfinite(betas).all():
        raise ValueError("DDPM betas must be a finite, nonempty 1-D schedule")
    if np.any((betas <= 0) | (betas > 1)):
        raise ValueError("DDPM betas must lie in (0, 1]")
    path = [state.copy()]
    for beta in betas:
        state = np.sqrt(1-beta)*state + np.sqrt(beta)*rng.standard_normal(state.shape)
        path.append(state.copy())
    return np.stack(path)


def sample_bridge_path(x0, endpoint, beta, variance, rng):
    """Sample the bridge's one-step conditional transitions.

    gamma_t = (1-beta_t)/(1-beta_(t-1))
    innovation variance = delta_t - gamma_t**2 * delta_(t-1)
    Both exact and truncated endpoint grids are supported without changing them.
    """
    state, endpoint = _state(x0), _state(endpoint)
    if state.shape != endpoint.shape:
        raise ValueError("Source and endpoint shapes must match")
    beta, variance = np.asarray(beta, dtype=np.float64), np.asarray(variance, dtype=np.float64)
    if beta.ndim != 1 or beta.shape != variance.shape or not len(beta):
        raise ValueError("Bridge schedules must be nonempty matching 1-D arrays")
    if not np.isfinite(beta).all() or not np.isfinite(variance).all():
        raise ValueError("Bridge schedules must be finite")
    if np.any(np.diff(beta) <= 0) or np.any((beta <= 0) | (beta > 1)):
        raise ValueError("Bridge beta must increase strictly within (0, 1]")
    if np.any(variance < 0) or (beta[-1] == 1 and variance[-1] != 0):
        raise ValueError("Invalid bridge marginal variance")
    path = [state.copy()]
    previous_beta = previous_variance = 0.0
    for b, delta in zip(beta, variance):
        gamma = (1-b)/(1-previous_beta)
        conditional_variance = delta - gamma**2 * previous_variance
        tolerance = 1e-12 * max(1.0, delta, previous_variance)
        if conditional_variance < -tolerance:
            raise ValueError("Bridge schedule gives a negative conditional variance")
        mean = gamma*state + (b-gamma*previous_beta)*endpoint
        state = mean + np.sqrt(max(0.0, conditional_variance))*rng.standard_normal(state.shape)
        path.append(state.copy())
        previous_beta, previous_variance = b, delta
    return np.stack(path)
