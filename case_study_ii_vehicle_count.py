"""Case Study II extension: increase total number of vehicles.

Quick run:
  python case_study_ii_vehicle_count.py --seeds 2 --T 300 --vehicle-counts 4 6 8 --out smoke_vehicles
Full run:
  python case_study_ii_vehicle_count.py --seeds 10 --T 1500 --vehicle-counts 4 6 8 --lags 0 5 --out results_case_study_ii_vehicle_count
"""
import argparse
from pathlib import Path
from case_study_common import run_vehicle_experiment


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--T", type=int, default=1500)
    p.add_argument("--vehicle-counts", type=int, nargs="+", default=[4, 6, 8],
                   help="total number of vehicles INCLUDING the ego vehicle")
    p.add_argument("--lags", type=int, nargs="+", default=[0, 5])
    p.add_argument("--t0", type=int, default=15)
    p.add_argument("--seed0", type=int, default=5000)
    p.add_argument("--sigma", type=float, default=0.05,
                   help="trajectory-prediction noise scale, held constant across counts")
    p.add_argument("--safety-distance", type=float, default=4.0)
    p.add_argument("--out", default="results_case_study_ii_vehicle_count")
    a = p.parse_args(argv)
    metrics, checks = run_vehicle_experiment(
        seeds=a.seeds, T=a.T, vehicle_counts=a.vehicle_counts, lags=tuple(a.lags),
        t0=a.t0, seed0=a.seed0, sigma=a.sigma,
        safety_distance=a.safety_distance, out=a.out)
    print(f"Multi-vehicle experiment complete: {len(metrics)} metric rows")
    print(f"Outputs: {Path(a.out).resolve()}")
    print(metrics.groupby(["condition", "method", "H"])[
        ["deployed_cov", "arrival_scored_cov", "alarm_rate", "true_violation_rate"]
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