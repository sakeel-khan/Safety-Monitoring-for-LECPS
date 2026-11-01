"""
synthetic.py -- synthetic data to test STL + ACP independently of any simulator/NN.

Two kinds of data:
 (1) robustness_series: directly synthesize (rho_hat_t, rho_t) with a chosen *error process*
     R_t = rho_hat_t - rho_t. This isolates ACP (Lemma 1 only concerns R_t vs C_t).
 (2) trajectory_series: synthetic 2-D trajectories + noisy "predicted" windows whose noise level
     shifts over time; robustness is computed with the STL module. This exercises STL -> ACP.
Neither claims to reproduce the paper's F1Tenth / race-car numbers.
"""
import numpy as np
from stl import window_collision_robustness

SCENARIOS = ["iid_id", "abrupt_shift", "gradual_drift", "ar1_dependent",
             "adversarial_monotone", "heavy_tail_switch", "trajectory_noise_shift"]


def ar1(rng, T, phi, mean, sd):
    """Stationary AR(1) with marginal mean/sd."""
    x = np.empty(T)
    x[0] = rng.normal(mean, sd)
    inn = sd * np.sqrt(1 - phi ** 2)
    for t in range(1, T):
        x[t] = mean + phi * (x[t - 1] - mean) + rng.normal(0, inn)
    return x


def error_sequence(rng, name, T):
    """Error process R_t = rho_hat_t - rho_t (positive = predictor over-estimates safety)."""
    t = np.arange(T)
    if name == "iid_id":
        return rng.normal(0, 0.2, T)                                  # exchangeable, no shift
    if name == "abrupt_shift":
        R = rng.normal(0, 0.2, T); R[T // 3:] = rng.normal(0.6, 0.5, T - T // 3); return R
    if name == "gradual_drift":
        return rng.normal(1.5 * t / T, 0.2 + 0.4 * t / T)
    if name == "ar1_dependent":
        return ar1(rng, T, 0.95, 0.0, 0.3)                            # strong serial dependence
    if name == "adversarial_monotone":
        # Increasing trend with small noise; not guaranteed to exceed every previous score.
        return 0.002 * t + rng.normal(0, 0.01, T)
    if name == "heavy_tail_switch":
        R = 0.2 * rng.standard_t(2, T); R[T // 2:] = 3 * R[T // 2:] + 0.5; return R
    raise ValueError(name)


def calibration_scores(rng, n=100):
    """Offline calibration errors from the in-distribution regime (N(0, 0.2))."""
    return rng.normal(0, 0.2, n)


def trajectory_series(rng, T, H=5, n_obs=30, c=0.5, sigma_id=0.05, sigma_ood=0.5, ood=True):
    """Unicycle path in a box with random point obstacles. 'Predicted' window = true window + noise
    whose std ramps from sigma_id to sigma_ood over the middle third (distribution shift) and grows
    with lookahead. Returns (rho_hat, rho, obstacles)."""
    obstacles = rng.uniform(-10, 10, (n_obs, 2))
    x = y = 0.0; th = rng.uniform(-np.pi, np.pi); om = 0.0; pos = []
    for _ in range(T + H + 1):
        om = 0.9 * om + rng.normal(0, 0.08); th += om
        x += 0.5 * np.cos(th); y += 0.5 * np.sin(th)
        if np.hypot(x, y) > 12:
            th += 0.3 * np.sign(np.sin(np.arctan2(-y, -x) - th))
        pos.append((x, y))
    pos = np.array(pos)
    end = sigma_ood if ood else sigma_id
    sigma = np.interp(np.arange(T), [0, T / 3, 2 * T / 3, T], [sigma_id, sigma_id, end, end])
    look = np.sqrt(np.arange(1, H + 1))[:, None]
    rho, rho_hat = np.empty(T), np.empty(T)
    for t in range(T):
        true = pos[t + 1:t + H + 1]
        pred = true + rng.normal(0, 1, (H, 2)) * sigma[t] * look
        rho[t] = window_collision_robustness(true, obstacles, c)
        rho_hat[t] = window_collision_robustness(pred, obstacles, c)
    return rho_hat, rho, obstacles


def robustness_series(rng, name, T):
    """(rho_hat, rho) for any scenario name."""
    if name == "trajectory_noise_shift":
        rho_hat, rho, _ = trajectory_series(rng, T)
        return rho_hat, rho
    rho = ar1(rng, T, 0.9, 2.0, 0.5)
    return rho + error_sequence(rng, name, T), rho


def trajectory_calibration(rng, n=300):
    """ID calibration errors for the trajectory scenario (no noise shift)."""
    rho_hat, rho, _ = trajectory_series(rng, n, ood=False)
    return rho_hat - rho