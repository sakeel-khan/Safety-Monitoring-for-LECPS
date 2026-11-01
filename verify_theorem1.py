"""
verify_theorem1.py -- EXPERIMENTAL check of Theorem 1 (Lin et al. 2025, Sec. 6.1).

Shows two complementary views of Theorem 1 across hidden-violation probability q:
  1. Pathwise premise check: fraction of paths with zero alarms (rho_hat_t > C_t for all monitored t).
  2. On paths satisfying that premise, compare the empirical safety rate with the theorem's lower bound.
     The theorem is conditional on the no-alarm premise; it does not say every unsafe path must alarm.
  3. P(satisfied OR alarmed) is reported only as an additional empirical diagnostic, not as the theorem's
     conditional guarantee. These synthetic experiments do not prove the theorem.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from acp import AdaptiveConformal, SplitConformal, run_lagged, theorem1_lower_bound

DELTA, GAMMA, T0 = 0.1, 0.005, 15
QS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.20]
LAGS = [0, 5]


def make_series(rng, T, q):
    rho_hat = 3 + 0.1 * rng.normal(size=T)
    big = rng.random(T) < q
    R = np.where(big, rng.uniform(3.8, 6.0, T), rng.normal(0, 0.1, T))
    return rho_hat, rho_hat - R


def analyse(rho_hat, rho, method, lag, cal):
    obj = AdaptiveConformal(DELTA, GAMMA) if method == "acp" else SplitConformal(DELTA, cal)
    out = run_lagged(rho_hat, rho, obj, lag=lag, t0=T0)
    mon = ~np.isnan(out["e"])
    C, rh, r = out["C"][mon], rho_hat[mon], rho[mon]
    no_alarm = rh > C                                   # strict
    sat = r > 0
    covered = (rh - r) <= C                             # deployed coverage
    T_eff = int(mon.sum())
    # Descriptive rate on no-alarm steps. The theorem's pathwise premise is checked separately below.
    sat_no_alarm_steps = float(sat[no_alarm].mean()) if no_alarm.any() else float("nan")
    # Effective lower bound on each path: if covered_t => sat_t on no_alarm steps
    safe_or_alarmed = float((sat | (~no_alarm)).mean())
    return dict(
        premise=bool(no_alarm.all()),
        sat_rate=float(sat.mean()),
        sat_no_alarm_steps=sat_no_alarm_steps,
        safe_or_alarmed=safe_or_alarmed,
        T_eff=T_eff,
        bound=float(theorem1_lower_bound(T_eff, DELTA, GAMMA)),
        alarm_rate=float((~no_alarm).mean()),
        impl_viol=int((no_alarm & covered & ~sat).sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    table = {}
    for lag in LAGS:
        for method in ("acp", "cp"):
            for q in QS:
                rs = []
                for seed in range(a.seeds):
                    rng = np.random.default_rng(5000 + seed)
                    cal = rng.normal(0, 0.1, 200)
                    rho_hat, rho = make_series(rng, a.T, q)
                    rs.append(analyse(rho_hat, rho, method, lag, cal))
                prem = [r for r in rs if r["premise"]]
                table[(lag, method, q)] = dict(
                    n_paths=len(rs),
                    n_premise=len(prem),
                    sat_premise_mean=float(np.mean([r["sat_rate"] for r in prem])) if prem else float("nan"),
                    sat_premise_min=float(np.min([r["sat_rate"] for r in prem])) if prem else float("nan"),
                    sat_safe_or_alarmed_mean=float(np.mean([r["safe_or_alarmed"] for r in rs])),
                    sat_safe_or_alarmed_min=float(np.min([r["safe_or_alarmed"] for r in rs])),
                    n_premise_below_bound=int(sum(r["sat_rate"] < r["bound"] for r in prem)),
                    sat_all_mean=float(np.mean([r["sat_rate"] for r in rs])),
                    alarm_rate_mean=float(np.mean([r["alarm_rate"] for r in rs])),
                    impl_viol=int(sum(r["impl_viol"] for r in rs)),
                    bound=rs[0]["bound"],
                )

    print(f"\nTheorem 1 check | delta={DELTA} gamma={GAMMA} T={a.T} seeds={a.seeds} t0={T0}")
    hdr = (f"{'lag':>3} {'method':>6} {'q':>5} | {'premise paths':>13} {'sat|premise mean [min]':>24} "
           f"{'P(sat v alarm)':>15} {'bound':>7} {'#below':>7} | {'alarm rate':>10} {'impl.viol':>9}")
    print(hdr); print("-" * len(hdr))
    for (lag, method, q), s in table.items():
        sp = "   n/a (alarms!)" if s["n_premise"] == 0 else f"{s['sat_premise_mean']:.3f} [{s['sat_premise_min']:.3f}]"
        print(f"{lag:>3} {method:>6} {q:>5.2f} | {s['n_premise']:>6d}/{s['n_paths']:<6d} {sp:>24} "
              f"{s['sat_safe_or_alarmed_mean']:>15.3f} {s['bound']:>7.3f} {s['n_premise_below_bound']:>7d} | "
              f"{s['alarm_rate_mean']:>10.3f} {s['impl_viol']:>9d}")

    with open(os.path.join(a.out, "theorem1_summary.json"), "w") as f:
        json.dump({f"lag{l}|{m}|q{q}": v for (l, m, q), v in table.items()}, f, indent=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for i, lag in enumerate(LAGS):
        for method, col in (("acp", "tab:blue"), ("cp", "tab:red")):
            fr = [table[(lag, method, q)]["n_premise"] / a.seeds for q in QS]
            sp = [table[(lag, method, q)]["sat_premise_mean"] for q in QS]
            sa = [table[(lag, method, q)]["sat_safe_or_alarmed_mean"] for q in QS]
            axes[i, 0].plot(QS, fr, "o-", color=col, lw=2, label=f"{method.upper()} (zero-alarm paths)")
            axes[i, 1].plot(QS, sp, "o-", color=col, lw=2, label=f"{method.upper()}: sat on zero-alarm paths")
            axes[i, 1].plot(QS, sa, "s--", color=col, alpha=0.75,
                            label=f"{method.upper()}: empirical P(safe OR alarm)")
        bound_val = table[(lag, "acp", QS[0])]["bound"]
        axes[i, 1].axhline(bound_val, color="k", ls="--", lw=1.5, label=f"Thm 1 bound 1-δ-p1 = {bound_val:.3f}")
        axes[i, 1].plot(QS, [1 - q for q in QS], "g:", lw=1.5, label="1 - q (unmonitored true safety)")
        axes[i, 0].axvline(DELTA, color="gray", ls=":", label="target δ = 0.10")
        axes[i, 0].set_title(f"lag {lag}: fraction of paths with zero alarms (premise holds)")
        axes[i, 1].set_title(f"lag {lag}: conditional safety check + empirical diagnostic")
        axes[i, 0].set_xlabel("q = probability of hidden violation"); axes[i, 1].set_xlabel("q")
        axes[i, 0].set_ylabel("Fraction of paths"); axes[i, 1].set_ylabel("Rate")
        axes[i, 0].set_ylim(-0.05, 1.05); axes[i, 1].set_ylim(0.70, 1.02)
        axes[i, 0].legend(fontsize=8); axes[i, 1].legend(fontsize=7, loc="lower left")
        axes[i, 0].grid(alpha=0.3); axes[i, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "theorem1_satisfaction.png"), dpi=140)
    plt.close()
    print(f"\nSaved updated figure + JSON to ./{a.out}/")


if __name__ == "__main__":
    main()