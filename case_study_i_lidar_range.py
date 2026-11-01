"""Case Study I extension: reduced LiDAR range.

Quick run:
  python case_study_i_lidar_range.py --seeds 2 --T 300 --ranges 5 3 2 --out smoke_lidar
Full run:
  python case_study_i_lidar_range.py --seeds 10 --T 1500 --ranges 5 3 2 --lags 0 5 --out results_case_study_i_lidar_range
"""
import argparse
from pathlib import Path
from case_study_common import run_lidar_experiment


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--T", type=int, default=1500)
    p.add_argument("--ranges", type=float, nargs="+", default=[5.0, 3.0, 2.0],
                   help="LiDAR range conditions in metres; 5 m is the nominal baseline")
    p.add_argument("--lags", type=int, nargs="+", default=[0, 5])
    p.add_argument("--t0", type=int, default=15)
    p.add_argument("--seed0", type=int, default=4000)
    p.add_argument("--sigma", type=float, default=0.05,
                   help="prediction-noise scale; held constant across range conditions")
    p.add_argument("--baseline-range", type=float, default=5.0)
    p.add_argument("--out", default="results_case_study_i_lidar_range")
    a = p.parse_args(argv)
    metrics, checks = run_lidar_experiment(
        seeds=a.seeds, T=a.T, ranges=a.ranges, lags=tuple(a.lags), t0=a.t0,
        seed0=a.seed0, sigma=a.sigma, baseline_range=a.baseline_range, out=a.out)
    print(f"LiDAR range experiment complete: {len(metrics)} metric rows")
    print(f"Outputs: {Path(a.out).resolve()}")
    print(metrics.groupby(["condition", "method", "H"])[
        ["deployed_cov", "arrival_scored_cov", "alarm_rate", "mean_visible_obstacles", "no_detection_fraction"]
    ].mean().round(4).to_string())
    h0 = checks.loc[checks["H"] == 0]
    required_checks = ["H0_thresholds_match", "H0_delta_match"]
    if h0.empty:
        print("H=0 consistency checks: no H=0 rows were generated.")
    else:
        missing = [c for c in required_checks if c not in h0.columns]
        if missing:
            print(f"H=0 consistency checks could not run; missing columns: {missing}")
        else:
            passed = bool(h0[required_checks].fillna(False).all().all())
            print("H=0 original-vs-lag-aware consistency checks pass:", passed)
            if not passed:
                raise SystemExit("H=0 consistency validation failed.")


if __name__ == "__main__":
    main()