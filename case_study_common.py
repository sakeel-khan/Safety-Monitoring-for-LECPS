"""Shared implementation for the reduced-LiDAR-range and vehicle-count case studies.

This is a controlled synthetic STL/ACP extension. It is not a reproduction of the
original physical F1Tenth simulator. All ACP comparison helpers are defined in this
file, so the case-study scripts do not depend on experiment_bc_common.py.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stl import Always, Predicate, window_collision_robustness
from acp import AdaptiveConformal, run_lagged
from acp_lag_aware import run_lag_aware, deployed_view

DELTA = 0.1
GAMMA = 0.005
DEFAULT_LAGS = (0, 5)
DEFAULT_T0 = 15
DEFAULT_BASELINE_SIGMA = 0.05


def make_base(seed: int, T: int, H: int = 5, n_obs: int = 30, c: float = 0.5):
    """Create one fixed trajectory/environment and standard-normal sensor-noise tensor.

    The environment and standard-normal tensor are shared across conditions, making
    noise-level and pre/post-shift comparisons paired. Noise tensor shape is (T,H,2).
    """
    if T <= H or H < 1:
        raise ValueError("T must be greater than H and H must be at least 1")
    ss = np.random.SeedSequence(int(seed))
    env_seed, noise_seed = ss.spawn(2)
    env_rng = np.random.default_rng(env_seed)
    noise_rng = np.random.default_rng(noise_seed)
    obstacles = env_rng.uniform(-10, 10, (n_obs, 2))
    x = y = 0.0
    th = env_rng.uniform(-np.pi, np.pi)
    om = 0.0
    positions = []
    for _ in range(T + H + 1):
        om = 0.9 * om + env_rng.normal(0, 0.08)
        th += om
        x += 0.5 * np.cos(th)
        y += 0.5 * np.sin(th)
        if np.hypot(x, y) > 12:
            th += 0.3 * np.sign(np.sin(np.arctan2(-y, -x) - th))
        positions.append((x, y))
    positions = np.asarray(positions, dtype=float)
    z = noise_rng.normal(0.0, 1.0, size=(T, H, 2))
    return {"positions": positions, "obstacles": obstacles, "z": z, "T": T, "H": H, "c": c}


def series_from_base(base, sigma, bias=(0.0, 0.0), change_point=None):
    """Compute (rho_hat, rho) for controlled measurement noise.

    sigma can be a scalar or length-T vector. bias is a 2-D additive measurement bias,
    or a (T,2) array. If change_point is provided, bias is applied only at t >= change_point.
    """
    T, H = base["T"], base["H"]
    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.ndim == 0:
        sigma_arr = np.full(T, float(sigma_arr))
    if sigma_arr.shape != (T,):
        raise ValueError("sigma must be scalar or a length-T array")
    if not np.all(np.isfinite(sigma_arr)) or np.any(sigma_arr < 0):
        raise ValueError("sigma values must be finite and nonnegative")
    bias_arr = np.asarray(bias, dtype=float)
    if bias_arr.shape == (2,):
        bias_arr = np.tile(bias_arr, (T, 1))
    if bias_arr.shape != (T, 2):
        raise ValueError("bias must have shape (2,) or (T,2)")
    if change_point is not None:
        if not (0 <= int(change_point) < T):
            raise ValueError("change_point must be in [0, T)")
        bias_arr = bias_arr.copy()
        bias_arr[:int(change_point)] = 0.0

    pos, obstacles, z, c = base["positions"], base["obstacles"], base["z"], base["c"]
    look = np.sqrt(np.arange(1, H + 1, dtype=float))[:, None]
    rho = np.empty(T, dtype=float)
    rho_hat = np.empty(T, dtype=float)
    for t in range(T):
        true_window = pos[t + 1:t + H + 1]
        pred_window = true_window + bias_arr[t] + z[t] * sigma_arr[t] * look
        rho[t] = window_collision_robustness(true_window, obstacles, c)
        rho_hat[t] = window_collision_robustness(pred_window, obstacles, c)
    return rho_hat, rho


def calibration_scores(seed: int, n: int = 300, H: int = 5, sigma: float = DEFAULT_BASELINE_SIGMA):
    """Independent baseline-regime calibration scores, shared by all paired conditions."""
    base = make_base(seed + 1_000_003, n, H=H)
    rh, rho = series_from_base(base, sigma=sigma)
    scores = rh - rho
    if not np.all(np.isfinite(scores)):
        raise ValueError("non-finite calibration scores generated")
    return scores


def forward_window_stats(e_deployed, onset, end, delta=DELTA, window=50, tolerance=0.05):
    """First forward window at/after onset reaching target coverage minus tolerance."""
    values = 1.0 - np.asarray(e_deployed[onset:end + 1], dtype=float)
    if np.isnan(values).any() or len(values) < window:
        return np.nan, True, np.nan, float(np.nanmean(values)) if len(values) else np.nan
    cs = np.concatenate([[0.0], np.cumsum(values)])
    fwd = (cs[window:] - cs[:-window]) / window
    target = 1.0 - delta - tolerance
    hit = np.flatnonzero(fwd >= target)
    return (float(hit[0]), False, float(np.min(fwd)), float(np.mean(values))) if len(hit) else (np.nan, True, float(np.min(fwd)), float(np.mean(values)))


def compare_methods(rho_hat, rho, cal, lags=DEFAULT_LAGS, t0=DEFAULT_T0, delta=DELTA,
                    gamma=GAMMA, onset=None, response_window=50):
    """Run original and issuance-error lag-aware ACP on identical series."""
    T = len(rho)
    if len(rho_hat) != T:
        raise ValueError("rho_hat and rho must have equal length")
    lags = tuple(int(x) for x in lags)
    max_lag = max(lags)
    start_eval, end_eval = max(t0, max_lag), T - 1 - max_lag
    if end_eval < start_eval:
        raise ValueError("T too small for the requested lag/evaluation window")
    eval_mask = (np.arange(T) >= start_eval) & (np.arange(T) <= end_eval)
    rows, traces, checks = [], {}, []
    for H in lags:
        original_ctl = AdaptiveConformal(delta, gamma, init_scores=cal, include_current=False)
        original = run_lagged(rho_hat, rho, original_ctl, lag=H, t0=t0)
        original_view = deployed_view(rho_hat, rho, original["C"], H, original["start"])
        lagged = run_lag_aware(rho_hat, rho, delta, gamma, H, t0, cal,
                               include_current=False, update_error="issuance")
        lagged_view = deployed_view(rho_hat, rho, lagged["C_issue"], H, lagged["start"])
        checks.append({
            "H": H,
            "H0_thresholds_match": bool(H != 0 or np.allclose(original["C"], lagged["C_issue"], equal_nan=True)),
            "H0_delta_match": bool(H != 0 or np.allclose(original["delta_t"], lagged["delta_issue"], equal_nan=True)),
            "original_arrival_metric_defined": bool(np.isfinite(original_view["e_arrival"][eval_mask]).all()),
            "lagged_deployed_metric_defined": bool(np.isfinite(lagged_view["e_deployed"][eval_mask]).all()),
        })
        for method, out, view, ckey, dkey in [
            ("original_acp", original, original_view, "C", "delta_t"),
            ("lag_aware_acp", lagged, lagged_view, "C_issue", "delta_issue"),
        ]:
            ed = np.asarray(view["e_deployed"], float)
            ea = np.asarray(view["e_arrival"], float)
            cov_dep = 1.0 - float(np.mean(ed[eval_mask]))
            cov_arr = 1.0 - float(np.mean(ea[eval_mask]))
            alarms = np.asarray(view["alarm"], bool)[eval_mask]
            row = {
                "method": method, "H": H, "n_eval": int(eval_mask.sum()),
                "deployed_cov": cov_dep, "arrival_scored_cov": cov_arr,
                "absdev_deployed": abs(cov_dep - (1 - delta)),
                "absdev_arrival_scored": abs(cov_arr - (1 - delta)),
                "alarm_rate": float(alarms.mean()),
                "true_violation_rate": float(np.mean(np.asarray(rho)[eval_mask] <= 0)),
                "mean_threshold_finite": float(np.mean(np.asarray(out[ckey])[eval_mask][np.isfinite(np.asarray(out[ckey])[eval_mask])])) if np.isfinite(np.asarray(out[ckey])[eval_mask]).any() else np.nan,
                "frac_threshold_nonfinite": float(np.mean(~np.isfinite(np.asarray(out[ckey])[eval_mask]))),
                "delta_t_min": float(np.nanmin(out[dkey])),
                "delta_t_max": float(np.nanmax(out[dkey])),
            }
            if onset is not None:
                onset_eval = max(int(onset), start_eval)
                delay, censored, min_forward, post_cov = forward_window_stats(
                    ed, onset_eval, end_eval, delta=delta, window=min(response_window, max(1, end_eval-onset_eval+1))
                )
                row.update(response_delay=delay, response_censored=censored,
                           min_forward_cov_post_shift=min_forward, post_shift_deployed_cov=post_cov,
                           shift_onset=int(onset))
            else:
                row.update(response_delay=np.nan, response_censored=np.nan,
                           min_forward_cov_post_shift=np.nan, post_shift_deployed_cov=np.nan,
                           shift_onset=np.nan)
            rows.append(row)
            traces[(method, H)] = {
                "C": np.asarray(out[ckey], float),
                "delta_t": np.asarray(out[dkey], float),
                "e_deployed": ed,
                "e_arrival": ea,
                "alarm": np.asarray(view["alarm"], bool),
            }
    return pd.DataFrame(rows), traces, pd.DataFrame(checks), (start_eval, end_eval)


def rolling_coverage(e, window=50):
    e = np.asarray(e, dtype=float)
    out = np.full(len(e), np.nan)
    for t in range(window - 1, len(e)):
        chunk = e[t-window+1:t+1]
        if np.isfinite(chunk).all():
            out[t] = 1.0 - np.mean(chunk)
    return out


def save_outputs(rows, traces, checks, metadata, out_dir, scenario_label, onset=None, rolling_window=50):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "per_seed_metrics.csv", index=False)
    checks.to_csv(out / "consistency_checks.csv", index=False)
    numeric = [c for c in rows.columns if pd.api.types.is_numeric_dtype(rows[c]) and c not in {"seed", "H", "n_eval", "shift_onset"}]
    grouped = rows.groupby(["condition", "method", "H"], dropna=False)[numeric].mean(numeric_only=True).reset_index()
    grouped.to_csv(out / "summary_by_condition.csv", index=False)
    paired = rows.pivot_table(index=["seed", "condition", "H"], columns="method",
                              values=["deployed_cov", "arrival_scored_cov", "absdev_deployed", "alarm_rate"], aggfunc="first")
    paired.to_csv(out / "paired_differences_wide.csv")
    if ("deployed_cov", "lag_aware_acp") in paired.columns and ("deployed_cov", "original_acp") in paired.columns:
        diff = pd.DataFrame(index=paired.index).reset_index()
        diff["deployed_cov_lag_aware_minus_original"] = (paired[("deployed_cov", "lag_aware_acp")] - paired[("deployed_cov", "original_acp")]).to_numpy()
        diff["absdev_lag_aware_minus_original"] = (paired[("absdev_deployed", "lag_aware_acp")] - paired[("absdev_deployed", "original_acp")]).to_numpy()
        diff["alarm_rate_lag_aware_minus_original"] = (paired[("alarm_rate", "lag_aware_acp")] - paired[("alarm_rate", "original_acp")]).to_numpy()
        diff.to_csv(out / "paired_differences.csv", index=False)
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    # Mean traces over seeds by condition/method/H. Traces keys include seed in the caller's storage.
    if traces:
        trace_rows = []
        groups = {}
        for key, trace in traces.items():
            condition, method, H, seed = key
            groups.setdefault((condition, method, H), []).append((seed, trace))
        for (condition, method, H), items in groups.items():
            for metric in ("C", "delta_t", "e_deployed", "e_arrival"):
                arr = np.stack([v[metric] for _, v in items])
                with np.errstate(invalid="ignore"):
                    counts = np.sum(np.isfinite(arr), axis=0)
                    mean = np.divide(np.nansum(arr, axis=0), counts,
                                     out=np.full(arr.shape[1], np.nan), where=counts > 0)
                for t, val in enumerate(mean):
                    trace_rows.append({"condition": condition, "method": method, "H": H, "metric": metric, "t": t, "mean": val})
        pd.DataFrame(trace_rows).to_csv(out / "mean_traces.csv", index=False)

        conditions = sorted(set(k[0] for k in traces))
        for condition in conditions:
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            for method in ("original_acp", "lag_aware_acp"):
                for H in sorted(set(k[2] for k in traces)):
                    items = [v for (co, me, h, _), v in traces.items() if co == condition and me == method and h == H]
                    if not items:
                        continue
                    cstack = np.stack([x["C"] for x in items])
                    estack = np.stack([x["e_deployed"] for x in items])
                    cc = np.sum(np.isfinite(cstack), axis=0)
                    ec = np.sum(np.isfinite(estack), axis=0)
                    Cmean = np.divide(np.nansum(cstack, axis=0), cc, out=np.full(cstack.shape[1], np.nan), where=cc > 0)
                    emean = np.divide(np.nansum(estack, axis=0), ec, out=np.full(estack.shape[1], np.nan), where=ec > 0)
                    covmean = rolling_coverage(emean, rolling_window)
                    label = f"{method}, H={H}"
                    axes[0].plot(Cmean, label=label)
                    axes[1].plot(covmean, label=label)
            if onset is not None:
                axes[0].axvline(onset, linestyle="--")
                axes[1].axvline(onset, linestyle="--")
            axes[0].set_ylabel("Mean threshold C")
            axes[1].set_ylabel(f"Rolling deployed coverage ({rolling_window})")
            axes[1].set_xlabel("Issue time t")
            axes[0].set_title(f"{scenario_label}: {condition}")
            axes[0].legend(fontsize=7, ncol=2)
            axes[1].axhline(1-DELTA, linestyle=":")
            axes[1].set_ylim(-0.05, 1.05)
            axes[1].legend(fontsize=7, ncol=2)
            fig.tight_layout()
            safe = str(condition).replace("/", "_").replace(" ", "_")
            fig.savefig(out / f"{safe}_threshold_and_coverage.png", dpi=140)
            plt.close(fig)

    # Response delay plot only if values are meaningful.
    if "response_delay" in rows and rows["response_delay"].notna().any():
        plot = rows.dropna(subset=["response_delay"])
        if not plot.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            plot.groupby(["condition", "method", "H"])["response_delay"].mean().unstack(["method", "H"]).plot(kind="bar", ax=ax)
            ax.set_ylabel("Response delay (steps after shift)")
            ax.set_title(f"{scenario_label}: response delay")
            fig.tight_layout()
            fig.savefig(out / "response_delay.png", dpi=140)
            plt.close(fig)


def lidar_range_series(base, lidar_range: float, sigma: float = DEFAULT_BASELINE_SIGMA):
    """Return predicted/true STL robustness for a finite-range obstacle sensor.

    At issue time t, only static obstacles within lidar_range of the current ego position
    are detected. The predicted future path is checked against those currently visible
    obstacles. Estimated distance is capped at lidar_range, so an empty detection set
    produces finite observed-free-space robustness lidar_range - safety_distance.

    Ground-truth robustness always uses all obstacles and the true future path. The same
    pre-generated standard-normal tensor is reused across range conditions.
    """
    if not np.isfinite(lidar_range) or lidar_range <= 0:
        raise ValueError("lidar_range must be finite and positive")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    positions, obstacles, z = base["positions"], base["obstacles"], base["z"]
    T, H, c = base["T"], base["H"], base["c"]
    look = np.sqrt(np.arange(1, H + 1, dtype=float))[:, None]
    rho_hat = np.empty(T, dtype=float)
    rho = np.empty(T, dtype=float)
    visible_counts = np.empty(T, dtype=int)
    for t in range(T):
        current = positions[t]
        visible = obstacles[np.linalg.norm(obstacles - current[None, :], axis=1) <= lidar_range]
        visible_counts[t] = len(visible)
        true_window = positions[t + 1:t + H + 1]
        predicted_window = true_window + z[t] * sigma * look
        rho[t] = window_collision_robustness(true_window, obstacles, c)
        if len(visible):
            distances = np.linalg.norm(predicted_window[:, None, :] - visible[None, :, :], axis=-1)
            nearest_predicted_distance = float(np.min(distances))
        else:
            nearest_predicted_distance = float("inf")
        rho_hat[t] = min(nearest_predicted_distance, float(lidar_range)) - c
    return rho_hat, rho, {
        "visible_obstacle_count": visible_counts,
        "no_detection": visible_counts == 0,
    }


def lidar_calibration_scores(seed: int, n: int = 300, H: int = 5,
                             sigma: float = DEFAULT_BASELINE_SIGMA, baseline_range: float = 5.0):
    """Calibration scores from the nominal 5 m LiDAR regime, independent of test data."""
    base = make_base(seed + 1_000_003, n, H=max(1, H))
    rh, rho, _ = lidar_range_series(base, baseline_range, sigma)
    scores = rh - rho
    if not np.all(np.isfinite(scores)):
        raise ValueError("non-finite LiDAR calibration scores")
    return scores


def make_vehicle_base(seed: int, T: int, H: int = 5, max_vehicles: int = 8,
                      safety_distance: float = 4.0):
    """Generate paired synthetic traffic trajectories for up to max_vehicles total cars.

    Vehicle 0 is the ego vehicle. All vehicles travel on a shared elliptical circuit with
    seeded phase offsets and slightly different angular speeds. The first n trajectories
    are used for an n-vehicle condition, so conditions are nested within the same scenario.
    """
    if T <= H or H < 1:
        raise ValueError("T must be greater than H and H must be at least 1")
    if not (2 <= max_vehicles <= 20):
        raise ValueError("max_vehicles must be between 2 and 20")
    if not np.isfinite(safety_distance) or safety_distance <= 0:
        raise ValueError("safety_distance must be finite and positive")
    rng = np.random.default_rng(int(seed))
    # Nested phase design: four cars are roughly evenly spaced; extra cars fill the gaps.
    # This keeps the baseline scenario from starting with accidental near-collisions.
    phase_slots = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2,
                            np.pi / 4, 5 * np.pi / 4, 3 * np.pi / 4, 7 * np.pi / 4])
    rotation = rng.uniform(0, 2 * np.pi)
    phases = (phase_slots[:max_vehicles] + rotation) % (2 * np.pi)
    # Keep relative speeds close so the count effect is not overwhelmed by repeated overtaking.
    speeds = rng.normal(0.018, 0.00015, max_vehicles)
    speeds = np.maximum(speeds, 0.004)
    # Small lane/radius variations avoid forcing every car onto exactly the same curve.
    rx = rng.uniform(7.4, 8.6, max_vehicles)
    ry = rng.uniform(4.5, 5.5, max_vehicles)
    times = np.arange(T + H + 1, dtype=float)
    angles = phases[None, :] + times[:, None] * speeds[None, :]
    positions = np.empty((T + H + 1, max_vehicles, 2), dtype=float)
    positions[:, :, 0] = rx[None, :] * np.cos(angles)
    positions[:, :, 1] = ry[None, :] * np.sin(angles)
    z = rng.normal(0.0, 1.0, size=(T, H, max_vehicles, 2))
    return {"positions": positions, "z": z, "T": T, "H": H,
            "max_vehicles": max_vehicles, "safety_distance": float(safety_distance)}


def _dynamic_collision_formula(safety_distance: float, horizon: int):
    """STL formula G[0,H-1] (minimum ego-to-other-vehicle distance > d)."""
    def clearance(state):
        state = np.asarray(state, dtype=float)
        ego = state[:2]
        others = state[2:].reshape(-1, 2)
        return float(np.linalg.norm(others - ego[None, :], axis=1).min() - safety_distance)
    return Always(0, horizon - 1, Predicate(clearance, "ego-other clearance"))


def vehicle_count_series(base, total_vehicles: int, sigma: float = DEFAULT_BASELINE_SIGMA):
    """Compute true and predicted STL robustness for a total-vehicle-count condition.

    `total_vehicles` includes the ego vehicle. Predicted states for ego and other cars receive
    matched-seed Gaussian trajectory-prediction noise, scaled by sqrt(lookahead). Ground truth
    is computed from all actual vehicle trajectories in the selected condition.
    """
    if not isinstance(total_vehicles, (int, np.integer)) or not (2 <= total_vehicles <= base["max_vehicles"]):
        raise ValueError(f"total_vehicles must be an integer in [2, {base['max_vehicles']}]")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    positions, z = base["positions"], base["z"]
    T, H = base["T"], base["H"]
    formula = _dynamic_collision_formula(base["safety_distance"], H)
    look = np.sqrt(np.arange(1, H + 1, dtype=float))[:, None, None]
    rho_hat, rho = np.empty(T, dtype=float), np.empty(T, dtype=float)
    for t in range(T):
        true_window = positions[t + 1:t + H + 1, :total_vehicles, :]
        predicted_window = true_window + z[t, :, :total_vehicles, :] * sigma * look
        rho[t] = formula.rho(true_window.reshape(H, -1), 0)
        rho_hat[t] = formula.rho(predicted_window.reshape(H, -1), 0)
    return rho_hat, rho


def vehicle_calibration_scores(seed: int, n: int = 300, H: int = 5,
                               sigma: float = DEFAULT_BASELINE_SIGMA,
                               baseline_vehicles: int = 4, safety_distance: float = 4.0):
    """Calibration scores from the nominal four-vehicle traffic condition."""
    base = make_vehicle_base(seed + 1_000_003, n, H=max(1, H), max_vehicles=max(8, baseline_vehicles),
                             safety_distance=safety_distance)
    rh, rho = vehicle_count_series(base, baseline_vehicles, sigma)
    scores = rh - rho
    if not np.all(np.isfinite(scores)):
        raise ValueError("non-finite vehicle calibration scores")
    return scores




def run_lidar_experiment(seeds=10, T=1500, ranges=(5.0, 3.0, 2.0),
                         lags=DEFAULT_LAGS, t0=DEFAULT_T0, seed0=4000,
                         sigma=DEFAULT_BASELINE_SIGMA, baseline_range=5.0,
                         out="results_case_study_i_lidar_range"):
    import pandas as pd
    ranges = tuple(float(x) for x in ranges)
    if not ranges or any(not np.isfinite(x) or x <= 0 for x in ranges):
        raise ValueError("ranges must be a non-empty list of finite positive values")
    if seeds < 1 or T <= max(lags) + max(t0, 1):
        raise ValueError("seeds must be positive and T must be long enough for the evaluation window")
    all_rows, all_checks, all_traces = [], [], {}
    for seed in range(seed0, seed0 + seeds):
        base = make_base(seed, T, H=max(1, max(lags)))
        cal = lidar_calibration_scores(seed, H=max(1, max(lags)), sigma=sigma, baseline_range=baseline_range)
        for r in ranges:
            rh, rho, sensor = lidar_range_series(base, r, sigma)
            rows, traces, checks, window = compare_methods(rh, rho, cal, lags=lags, t0=t0)
            condition = f"range_{r:g}m"
            rows["seed"] = seed
            rows["condition"] = condition
            rows["lidar_range_m"] = r
            rows["prediction_sigma"] = sigma
            rows["mean_visible_obstacles"] = float(np.mean(sensor["visible_obstacle_count"]))
            rows["no_detection_fraction"] = float(np.mean(sensor["no_detection"]))
            rows["evaluation_start"], rows["evaluation_end"] = window
            all_rows.append(rows)
            checks["seed"], checks["condition"] = seed, condition
            all_checks.append(checks)
            for (method, H), trace in traces.items():
                all_traces[(condition, method, H, seed)] = trace
    metrics, checks = pd.concat(all_rows, ignore_index=True), pd.concat(all_checks, ignore_index=True)
    metadata = {
        "experiment": "Case Study I extension — reduced LiDAR range",
        "seed_start": seed0, "seeds": seeds, "T": T, "ranges_m": list(ranges),
        "baseline_range_m": baseline_range, "prediction_sigma": sigma,
        "lags": list(lags), "t0": t0, "delta": DELTA, "gamma": GAMMA,
        "sensor_model": "At each issue time, detect only static obstacles within range of current ego position; predicted path robustness is evaluated against currently visible obstacles and capped at range minus safety distance when no nearer obstacle is known. Ground truth uses all obstacles.",
        "calibration": f"Independent nominal calibration at {baseline_range} m range; same calibration reused across range conditions and methods.",
        "limitation": "Synthetic range-limited sensing proxy, not a physical LiDAR ray-casting or fog/attenuation model.",
        "theory_note": "Empirical results only; no theoretical guarantee is claimed for lag-aware deployed coverage."
    }
    save_outputs(metrics, all_traces, checks, metadata, out, "Case Study I — Reduced LiDAR Range")
    return metrics, checks


def run_vehicle_experiment(seeds=10, T=1500, vehicle_counts=(4, 6, 8),
                           lags=DEFAULT_LAGS, t0=DEFAULT_T0, seed0=5000,
                           sigma=DEFAULT_BASELINE_SIGMA, safety_distance=4.0,
                           out="results_case_study_ii_vehicle_count"):
    import pandas as pd
    raw_counts = tuple(vehicle_counts)
    if not raw_counts or any(not isinstance(x, (int, np.integer)) for x in raw_counts):
        raise ValueError("vehicle_counts must be integers")
    counts = tuple(int(x) for x in raw_counts)
    if any(x < 2 or x > 8 for x in counts):
        raise ValueError("vehicle_counts must contain total vehicle counts in [2, 8], including ego")
    if len(set(counts)) != len(counts):
        raise ValueError("vehicle_counts must not contain duplicates")
    if seeds < 1 or T <= max(lags) + max(t0, 1):
        raise ValueError("seeds must be positive and T must be long enough for the evaluation window")
    all_rows, all_checks, all_traces = [], [], {}
    for seed in range(seed0, seed0 + seeds):
        base = make_vehicle_base(seed, T, H=max(1, max(lags)), max_vehicles=max(8, max(counts)),
                                 safety_distance=safety_distance)
        cal = vehicle_calibration_scores(seed, H=max(1, max(lags)), sigma=sigma,
                                         baseline_vehicles=4, safety_distance=safety_distance)
        for count in counts:
            rh, rho = vehicle_count_series(base, count, sigma)
            rows, traces, checks, window = compare_methods(rh, rho, cal, lags=lags, t0=t0)
            condition = f"vehicles_{count}_total"
            rows["seed"] = seed
            rows["condition"] = condition
            rows["total_vehicles"] = count
            rows["other_vehicles"] = count - 1
            rows["prediction_sigma"] = sigma
            rows["safety_distance"] = safety_distance
            rows["evaluation_start"], rows["evaluation_end"] = window
            all_rows.append(rows)
            checks["seed"], checks["condition"] = seed, condition
            all_checks.append(checks)
            for (method, H), trace in traces.items():
                all_traces[(condition, method, H, seed)] = trace
    metrics, checks = pd.concat(all_rows, ignore_index=True), pd.concat(all_checks, ignore_index=True)
    metadata = {
        "experiment": "Case Study II extension — increased number of vehicles",
        "seed_start": seed0, "seeds": seeds, "T": T, "total_vehicle_counts_including_ego": list(counts),
        "prediction_sigma": sigma, "safety_distance": safety_distance,
        "lags": list(lags), "t0": t0, "delta": DELTA, "gamma": GAMMA,
        "traffic_model": "Seeded synthetic vehicles follow an elliptical circuit with different initial phases, speeds, and small lane-radius variations. Conditions use nested subsets of the same generated trajectories; ego is vehicle 0.",
        "robustness": "STL G[0,H-1] of minimum ego-to-any-other-vehicle distance minus the fixed safety distance.",
        "calibration": "Independent nominal calibration on four total vehicles (ego plus three others), reused across conditions and methods.",
        "limitation": "Synthetic multi-vehicle traffic proxy, not the paper's full racing simulator or learned predictor.",
        "theory_note": "Empirical results only; no theoretical guarantee is claimed for lag-aware deployed coverage."
    }
    save_outputs(metrics, all_traces, checks, metadata, out, "Case Study II — More Vehicles")
    return metrics, checks