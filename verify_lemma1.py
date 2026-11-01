"""
verify_lemma1.py -- EXPERIMENTAL check of Lemma 1 (Lin et al. 2025, Sec. 6.1).

Lemma 1:  1-delta-p1 <= (1/T) sum_t [coverage_t] <= 1-delta+p2,
          p1=(delta+gamma)/(T gamma), p2=((1-delta)+gamma)/(T gamma).

What this script does and does NOT show
  * Computes, for synthetic series with several kinds of distribution shift, the running empirical
    coverage of ACP and checks that it stays inside the envelope for EVERY prefix T, for many seeds.
  * Compares against split conformal prediction (CP), which has no such guarantee under shift.
  * Includes a negative control (sign-flipped update) to show the check CAN fail.
  * Reports two coverage notions when there is an H-step lag:
        scored   : R_t <= C_t   (what ACP's delta-update controls; this is what Lemma 1 bounds)
        deployed : (rho_hat_t - rho_t) <= C_t with the SAME C_t that raised/withheld the alarm at t.
    For lag = 0 they coincide. For lag > 0 Lemma 1 says nothing directly about 'deployed'; we only measure it.
  * This is evidence about the CODE and about behaviour on these synthetic scenarios. It is not a proof.
    The scored-coverage statement is a deterministic consequence of the update rule (telescoping, see acp.py);
    finite experiments cannot add to that, they can only catch implementation errors or wrong assumptions.

Usage: python verify_lemma1.py [--T 2000] [--seeds 30] [--out results]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from acp import AdaptiveConformal, SplitConformal, run_lagged, lemma1_bounds
import synthetic as syn

DELTA, GAMMA = 0.1, 0.005
CONFIGS = {
    "A_lag0_calinit": dict(lag=0, t0=0, init="cal", include_current=False),
    "B_lag5_calinit": dict(lag=5, t0=5, init="cal", include_current=False),
    "C_paper_literal": dict(lag=5, t0=15, init="none", include_current=True),   # Alg. 2 ordering, empty start
}


class SignFlipped(AdaptiveConformal):
    """NEGATIVE CONTROL: wrong sign in the delta-update. The envelope check must fail for this."""
    def step(self, score):
        import bisect
        C = self.threshold(); bisect.insort(self._scores, float(score))
        e = 0 if score <= C else 1
        self.delta_t = self.delta_t - self.gamma * (self.delta - e)
        return C, e


def envelope(e):
    T = np.arange(1, len(e) + 1)
    cov = 1 - np.cumsum(e) / T
    lo, hi = lemma1_bounds(T, DELTA, GAMMA)
    viol = (cov < lo - 1e-12) | (cov > hi + 1e-12)
    return cov, lo, hi, viol


def make(method, cfg, cal):
    if method == "acp":
        init = cal if cfg["init"] == "cal" else ()
        return AdaptiveConformal(DELTA, GAMMA, init_scores=init, include_current=cfg["include_current"])
    if method == "cp":
        return SplitConformal(DELTA, cal)
    if method == "flip":
        return SignFlipped(DELTA, GAMMA, init_scores=cal)
    raise ValueError(method)


def evaluate(rho_hat, rho, method, cfg, cal):
    out = run_lagged(rho_hat, rho, make(method, cfg, cal), lag=cfg["lag"], t0=cfg["t0"])
    mon = ~np.isnan(out["e"])
    e = out["e"][mon]
    cov, lo, hi, viol = envelope(e)
    deployed = float(((rho_hat - rho) <= out["C"])[mon].mean())
    return dict(cov=cov, final=float(cov[-1]), n_viol=int(viol.sum()), deployed=deployed,
                T_eff=int(mon.sum()), delta_t=out["delta_t"][mon])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    summary, curves = {}, {}
    for sc in syn.SCENARIOS:
        rec = {}
        for seed in range(a.seeds):
            rng = np.random.default_rng(1000 + seed)
            cal = syn.trajectory_calibration(rng) if sc == "trajectory_noise_shift" else syn.calibration_scores(rng)
            rho_hat, rho = syn.robustness_series(rng, sc, a.T)
            for cname, cfg in CONFIGS.items():
                for method in (["acp", "cp"] + (["flip"] if cname.startswith("A") else [])):
                    r = evaluate(rho_hat, rho, method, cfg, cal)
                    rec.setdefault((cname, method), []).append(r)
                    if seed == 0 and cname.startswith("A") and method in ("acp", "cp"):
                        curves[(sc, method)] = r
                        if method == "acp":
                            curves[(sc, "delta_t")] = r["delta_t"]
        summary[sc] = {}
        for (cname, method), rs in rec.items():
            summary[sc][f"{cname}|{method}"] = dict(
                seeds_violating_envelope=int(sum(r["n_viol"] > 0 for r in rs)),
                n_seeds=len(rs),
                final_scored_cov_mean=float(np.mean([r["final"] for r in rs])),
                final_scored_cov_min=float(np.min([r["final"] for r in rs])),
                final_scored_cov_max=float(np.max([r["final"] for r in rs])),
                deployed_cov_mean=float(np.mean([r["deployed"] for r in rs])),
                T_eff=rs[0]["T_eff"])

    # ------------------------------------------------------------------ report
    lo_T, hi_T = lemma1_bounds(a.T, DELTA, GAMMA)
    print(f"\nLemma 1 check | delta={DELTA} gamma={GAMMA} T={a.T} seeds={a.seeds} | target coverage {1-DELTA:.2f}")
    print(f"Envelope at T={a.T}: [{lo_T:.4f}, {hi_T:.4f}]   (envelope is wide for small T: p1,p2 ~ 1/T)\n")
    print(f"{'scenario':24s}{'config|method':26s}{'#seeds out of envelope':>24s}{'scored cov mean [min,max]':>34s}{'deployed':>10s}")
    for sc in syn.SCENARIOS:
        for key, s in summary[sc].items():
            print(f"{sc:24s}{key:26s}{s['seeds_violating_envelope']:>10d}/{s['n_seeds']:<13d}"
                  f"{s['final_scored_cov_mean']:>14.3f} [{s['final_scored_cov_min']:.3f},{s['final_scored_cov_max']:.3f}]"
                  f"{s['deployed_cov_mean']:>10.3f}")
    with open(os.path.join(a.out, "lemma1_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # ------------------------------------------------------------------ figure 1: coverage vs T
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, sc in zip(axes.ravel(), syn.SCENARIOS):
        acp_r, cp_r = curves[(sc, "acp")], curves[(sc, "cp")]
        Ts = np.arange(1, len(acp_r["cov"]) + 1)
        lo, hi = lemma1_bounds(Ts, DELTA, GAMMA)
        ax.plot(Ts, acp_r["cov"], label="ACP (running coverage)")
        ax.plot(Ts, cp_r["cov"], "r--", label="split CP")
        ax.plot(Ts, lo, "k:", label="Lemma 1 envelope"); ax.plot(Ts, hi, "k:")
        ax.axhline(1 - DELTA, color="gray", lw=0.8)
        ax.set_ylim(0.0, 1.05); ax.set_xlim(0, len(Ts)); ax.set_title(sc, fontsize=10)
        ax.set_xlabel("T"); ax.set_ylabel("running coverage")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    d = curves[("abrupt_shift", "delta_t")]
    axes[1, 3].plot(d); axes[1, 3].axhline(DELTA, color="gray", lw=0.8)
    axes[1, 3].set_title("delta_t under abrupt_shift (ACP adapts)", fontsize=10); axes[1, 3].set_xlabel("t")
    plt.suptitle("Lemma 1: empirical running coverage vs theoretical envelope (lag 0, seed 0)")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "lemma1_coverage_curves.png"), dpi=130); plt.close()

    # ------------------------------------------------------------------ figure 2: final coverage comparison
    fig, ax = plt.subplots(figsize=(13, 4.5)); w = 0.2; x = np.arange(len(syn.SCENARIOS))
    series = [("split CP (lag 0)", "A_lag0_calinit|cp", "final_scored_cov_mean"),
              ("ACP (lag 0)", "A_lag0_calinit|acp", "final_scored_cov_mean"),
              ("ACP lag 5, scored", "B_lag5_calinit|acp", "final_scored_cov_mean"),
              ("ACP lag 5, deployed", "B_lag5_calinit|acp", "deployed_cov_mean")]
    for i, (lab, key, fld) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, [summary[sc][key][fld] for sc in syn.SCENARIOS], w, label=lab)
    ax.axhline(1 - DELTA, color="k", ls="--", label="target 1-delta")
    ax.set_xticks(x); ax.set_xticklabels(syn.SCENARIOS, rotation=20, fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_ylabel("final coverage (mean over seeds)"); ax.legend(fontsize=8, ncol=3, loc="lower left")
    ax.set_title("ACP vs split CP, and scored vs deployed coverage under H-step lag")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "lemma1_final_coverage.png"), dpi=130); plt.close()
    print(f"\nSaved figures + JSON to ./{a.out}/")


if __name__ == "__main__":
    main()