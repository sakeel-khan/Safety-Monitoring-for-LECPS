"""
acp_coverage.py -- ACP empirical coverage over time, ordinary vs shifted/OOD data.

Pipeline (one pass per seed and scenario):
  1. generate true robustness rho_t and predicted robustness rho_hat_t           (synthetic.py)
  2. nonconformity score  R_t = rho_hat_{t-lag} - rho_{t-lag}                    (acp.run_lagged)
  3. adaptive threshold update  delta_{t+1}=delta_t+gamma(delta-e_t), C_t = quantile of scores   (acp.py)
  4. running coverage  = 1 - (1/T) sum_{s<=T} e_s     (e_s = 1 if R_s > C_s)
  5. repeat over seeds; plot mean and min-max band against the Lemma 1 envelope and the target 1-delta.

Usage
  python acp_coverage.py                       # defaults: 10 seeds, T=1500, lag 5, t0 15
  python acp_coverage.py --literal --init none # literal Algorithm 2 ordering, empty start
  python acp_coverage.py --cp                  # also overlay non-adaptive split CP (dashed)
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from acp import AdaptiveConformal, SplitConformal, run_lagged, lemma1_bounds
import synthetic as syn

DELTA, GAMMA = 0.1, 0.005                     # paper: delta = 0.1, gamma = 0.005
PANELS = {
    "Ordinary (in-distribution) data": ["iid_id", "trajectory_id"],
    "Shifted / OOD data": ["abrupt_shift", "gradual_drift", "adversarial_monotone", "trajectory_noise_shift"],
}


def get_series(rng, name, T):
    if name == "trajectory_id":                # trajectory-derived robustness with NO noise shift
        rho_hat, rho, _ = syn.trajectory_series(rng, T, ood=False)
        return rho_hat, rho
    return syn.robustness_series(rng, name, T)


def get_cal(rng, name, n=100):
    return syn.trajectory_calibration(rng) if name.startswith("trajectory") else syn.calibration_scores(rng, n)


def run_experiment(scenarios, seeds=10, T=1500, lag=5, t0=15, init="cal", literal=False, cp=False, seed0=2000):
    """Returns {scenario: dict(cov=(seeds, T_eff) running coverage of ACP, cov_cp=..., T_eff=int)}."""
    res = {}
    for name in scenarios:
        covs, covs_cp = [], []
        for s in range(seeds):
            rng = np.random.default_rng(seed0 + s)
            cal = get_cal(rng, name)
            rho_hat, rho = get_series(rng, name, T)
            acp = AdaptiveConformal(DELTA, GAMMA, init_scores=cal if init == "cal" else (), include_current=literal)
            out = run_lagged(rho_hat, rho, acp, lag=lag, t0=t0)
            e = out["e"][~np.isnan(out["e"])]
            covs.append(1 - np.cumsum(e) / np.arange(1, len(e) + 1))
            if cp:
                o2 = run_lagged(rho_hat, rho, SplitConformal(DELTA, cal), lag=lag, t0=t0)
                e2 = o2["e"][~np.isnan(o2["e"])]
                covs_cp.append(1 - np.cumsum(e2) / np.arange(1, len(e2) + 1))
        res[name] = dict(cov=np.array(covs), cov_cp=np.array(covs_cp) if cp else None, T_eff=len(covs[0]))
    return res


def envelope_stats(cov):
    """cov: (seeds, T). Number of (seed, prefix) points outside the Lemma 1 envelope, and seeds affected."""
    Ts = np.arange(1, cov.shape[1] + 1)
    lo, hi = lemma1_bounds(Ts, DELTA, GAMMA)
    out = (cov < lo - 1e-12) | (cov > hi + 1e-12)
    return int(out.sum()), int(out.any(axis=1).sum()), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--lag", type=int, default=5, help="H-step lag of the ground truth (paper: H=5)")
    ap.add_argument("--t0", type=int, default=15, help="start of ACP updates (paper: 15)")
    ap.add_argument("--init", choices=["cal", "none"], default="cal", help="warm-start NCS set from calibration data, or empty")
    ap.add_argument("--literal", action="store_true", help="literal Algorithm 2 order: append R_t before taking the quantile")
    ap.add_argument("--cp", action="store_true", help="overlay split CP (non-adaptive) for comparison")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    names = [n for v in PANELS.values() for n in v]
    res = run_experiment(names, a.seeds, a.T, a.lag, a.t0, a.init, a.literal, a.cp)

    # ------------------------------------------------------------- plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (title, scs) in zip(axes, PANELS.items()):
        for i, sc in enumerate(scs):
            cov = res[sc]["cov"]; Ts = np.arange(1, cov.shape[1] + 1)
            ax.plot(Ts, cov.mean(0), color=colors[i], label=f"ACP: {sc}")
            ax.fill_between(Ts, cov.min(0), cov.max(0), color=colors[i], alpha=0.18)
            if a.cp:
                ax.plot(Ts, res[sc]["cov_cp"].mean(0), "--", color=colors[i], lw=1, label=f"split CP: {sc}")
        Te = res[scs[0]]["T_eff"]; Ts = np.arange(1, Te + 1)
        lo, hi = lemma1_bounds(Ts, DELTA, GAMMA)
        ax.plot(Ts, lo, "k:", lw=1.6, label="Lemma 1 lower bound  1-δ-p1")
        ax.plot(Ts, hi, "k:", lw=1.6, label="Lemma 1 upper bound  1-δ+p2")
        ax.axhline(1 - DELTA, color="red", ls="--", lw=1.4, label=f"target coverage 1-δ = {1 - DELTA:.1f}")
        ax.set_ylim(0.5, 1.02); ax.set_xlim(0, Te)
        ax.set_xlabel("T (number of scored steps)"); ax.set_title(title)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("empirical coverage  (1/T) Σ (1 - e_t)")
    mode = "literal Alg. 2 order" if a.literal else "threshold before score"
    fig.suptitle(f"ACP coverage over time: mean and min-max band over {a.seeds} seeds "
                 f"(δ={DELTA}, γ={GAMMA}, lag={a.lag}, t0={a.t0}, init={a.init}, {mode})", fontsize=10)
    fig.text(0.5, 0.005, "Bounds are wide for small T (p1, p2 ~ 1/T): the upper bound exceeds 1 and the lower bound is far below 0.5 early on.",
             ha="center", fontsize=8)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    path = os.path.join(a.out, "acp_coverage.png")
    plt.savefig(path, dpi=140); plt.close()

    # ------------------------------------------------------------- summary table
    rows = []
    print(f"\nPart B | delta={DELTA} gamma={GAMMA} seeds={a.seeds} T={a.T} lag={a.lag} t0={a.t0} init={a.init} literal={a.literal}")
    print(f"{'scenario':24s}{'T_eff':>6s}{'final cov mean±std':>22s}{'[min,max]':>16s}{'envelope at T_eff':>20s}{'pts outside (seeds)':>22s}")
    for sc in names:
        cov = res[sc]["cov"]; n_pts, n_seeds, lo, hi = envelope_stats(cov)
        fin = cov[:, -1]
        print(f"{sc:24s}{res[sc]['T_eff']:>6d}{fin.mean():>14.3f} ± {fin.std():.3f}"
              f"{'[%.3f,%.3f]' % (fin.min(), fin.max()):>16s}{'[%.3f,%.3f]' % (lo[-1], hi[-1]):>20s}{n_pts:>16d} ({n_seeds}/{a.seeds})")
        rows.append([sc, res[sc]["T_eff"], fin.mean(), fin.std(), fin.min(), fin.max(), lo[-1], hi[-1], n_pts, n_seeds])
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "T_eff", "final_cov_mean", "final_cov_std", "final_cov_min", "final_cov_max",
                    "envelope_lo_at_T", "envelope_hi_at_T", "points_outside_envelope", "seeds_with_any_outside"])
        w.writerows(rows)
    print(f"\nSaved {path} and summary.csv")


if __name__ == "__main__":
    main()