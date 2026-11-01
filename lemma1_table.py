"""
lemma1_table.py -- Part C: empirically check Lemma 1 for increasing T.

Lemma 1:  1 - delta - p1  <=  (1/T) sum_t Pr(rho_hat_t - rho_t <= C_t)  <=  1 - delta + p2
          p1 = (delta + gamma)/(T gamma),   p2 = ((1 - delta) + gamma)/(T gamma)

Useful consequence (derived here, not stated in the paper):
          width = p1 + p2 = (1 + 2 gamma) / (T gamma)       -> independent of delta, shrinks like 1/T.

What is measured
  For each scenario and each T in a grid, run ACP for T SCORED steps on `seeds` independent series
  (a fresh experiment per T, so every row has its shift inside its own horizon).
  observed coverage(T) = 1 - (1/T) sum_{t<=T} e_t   per seed.
  The paper's quantity (1/T) sum Pr(...) is estimated by the mean over seeds; the bound is also checked per seed.

Usage
  python lemma1_table.py                    # 10 seeds, T up to 10000, lag 5, t0 15
  python lemma1_table.py --Tmax 2000 --seeds 5
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from acp import lemma1_bounds
from acp_coverage import run_experiment, PANELS, DELTA, GAMMA

T_GRID = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
TOL = 1e-12


def width_formula(T, delta=DELTA, gamma=GAMMA):
    return (1 + 2 * gamma) / (np.asarray(T, float) * gamma)


def build_table(scenarios, Ts, seeds=10, lag=5, t0=15, init="cal", literal=False, p_scale=1.0):
    """One row per (scenario, T). p_scale != 1 deliberately distorts the correction terms (used by the
    test that shows this check has power: a too-tight 'bound' must be violated)."""
    start = max(t0, lag)
    rows = []
    for sc in scenarios:
        for T in Ts:
            res = run_experiment([sc], seeds=seeds, T=T + start, lag=lag, t0=t0, init=init, literal=literal)
            assert res[sc]["T_eff"] == T
            cov = res[sc]["cov"][:, -1]                                  # coverage over all T scored steps
            lo0, hi0 = lemma1_bounds(T, DELTA, GAMMA)
            p1, p2 = (1 - DELTA) - float(lo0), float(hi0) - (1 - DELTA)
            lo, hi = (1 - DELTA) - p_scale * p1, (1 - DELTA) + p_scale * p2
            each = (cov >= lo - TOL) & (cov <= hi + TOL)
            rows.append(dict(scenario=sc, T=T, obs_mean=float(cov.mean()), obs_std=float(cov.std()),
                             obs_min=float(cov.min()), obs_max=float(cov.max()), lower=lo, upper=hi,
                             width=hi - lo, seeds=seeds, seeds_within=int(each.sum()),
                             within=bool(each.all() and lo - TOL <= cov.mean() <= hi + TOL)))
    return rows


def to_markdown(rows):
    out = []
    for sc in dict.fromkeys(r["scenario"] for r in rows):
        out += [f"### {sc}", "",
                "| T | observed coverage (mean over seeds) | [min, max] | lower bound | upper bound | width | within bounds? |",
                "|---:|---:|---:|---:|---:|---:|:---:|"]
        for r in (r for r in rows if r["scenario"] == sc):
            out.append(f"| {r['T']} | {r['obs_mean']:.4f} | [{r['obs_min']:.3f}, {r['obs_max']:.3f}] | "
                       f"{r['lower']:.4f} | {r['upper']:.4f} | {r['width']:.4f} | "
                       f"{'YES' if r['within'] else 'NO'} ({r['seeds_within']}/{r['seeds']} seeds) |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--Tmax", type=int, default=10000)
    ap.add_argument("--lag", type=int, default=5)
    ap.add_argument("--t0", type=int, default=15)
    ap.add_argument("--init", choices=["cal", "none"], default="cal")
    ap.add_argument("--literal", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    Ts = [T for T in T_GRID if T <= a.Tmax]
    scenarios = [n for v in PANELS.values() for n in v]
    rows = build_table(scenarios, Ts, a.seeds, a.lag, a.t0, a.init, a.literal)

    md = to_markdown(rows)
    header = (f"# Lemma 1 check: delta={DELTA}, gamma={GAMMA}, seeds={a.seeds}, lag={a.lag}, t0={a.t0}, init={a.init}, "
              f"literal={a.literal}\n\nwidth = p1 + p2 = (1+2*gamma)/(T*gamma). T needed for width <= eps: "
              + ", ".join(f"eps={e}: T>={int(np.ceil((1 + 2 * GAMMA) / (GAMMA * e)))}" for e in (0.5, 0.1, 0.05, 0.01)) + "\n\n")
    print("\n" + header + md)
    with open(os.path.join(a.out, "lemma1_table.md"), "w") as f:
        f.write(header + md + "\n")
    with open(os.path.join(a.out, "lemma1_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    n_bad = sum(not r["within"] for r in rows)
    print(f"{len(rows) - n_bad}/{len(rows)} (scenario, T) rows within the Lemma 1 bounds "
          f"({sum(r['seeds_within'] for r in rows)}/{sum(r['seeds'] for r in rows)} individual seed runs)")
    assert np.allclose([r["width"] for r in rows], width_formula([r["T"] for r in rows]))

    # ---------------------------------------------------------------- figure: tightening with T
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    Tf = np.array(Ts, float)
    ax[0].loglog(Tf, [r["width"] for r in rows[:len(Ts)]], "ko-", label="p1 + p2 (computed)")
    ax[0].loglog(Tf, 1.0 / Tf * (1 + 2 * GAMMA) / GAMMA, "r--", label="(1+2γ)/(Tγ)  ∝ 1/T")
    ax[0].set_xlabel("T"); ax[0].set_ylabel("bound width  upper - lower")
    ax[0].set_title("Lemma 1 bounds tighten like 1/T"); ax[0].legend(); ax[0].grid(True, which="both", alpha=0.3)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, sc in enumerate(dict.fromkeys(r["scenario"] for r in rows)):
        rr = [r for r in rows if r["scenario"] == sc]
        ax[1].semilogx([r["T"] for r in rr], [r["obs_mean"] for r in rr], "o-", color=colors[i % 10], label=sc, ms=4)
    lo, hi = lemma1_bounds(Tf, DELTA, GAMMA)
    ax[1].fill_between(Tf, lo, hi, color="gray", alpha=0.18, label="Lemma 1 bounds")
    ax[1].plot(Tf, lo, "k:", lw=1.2); ax[1].plot(Tf, hi, "k:", lw=1.2)
    ax[1].axhline(1 - DELTA, color="red", ls="--", lw=1.3, label="target 1-δ")
    ax[1].set_ylim(0.6, 1.05); ax[1].set_xlabel("T (scored steps)"); ax[1].set_ylabel("observed coverage (mean over seeds)")
    ax[1].set_title("Observed coverage vs bounds for increasing T"); ax[1].legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    path = os.path.join(a.out, "lemma1_tightening.png")
    plt.savefig(path, dpi=140); plt.close()
    print(f"\nSaved lemma1_table.md, lemma1_table.csv and {path}")


if __name__ == "__main__":
    main()