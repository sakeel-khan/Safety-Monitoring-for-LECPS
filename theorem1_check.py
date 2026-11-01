"""
theorem1_check.py -- Part D: empirically check Theorem 1 (Lin et al. 2025, Sec. 6.1).

Theorem 1: if rho_hat_t > C_t for ALL monitored t, then
           1 - delta - p1  <=  (1/T) sum_t Prob((s,t) |= phi),     p1 = (delta + gamma)/(T gamma).

Pipeline (every robustness value below is computed by the STL module, stl.py):
  trajectory window -> STL collision-avoidance robustness -> (rho_hat_t, rho_t)
  -> nonconformity scores R_t = rho_hat_{t-lag} - rho_{t-lag} -> ACP threshold C_t (acp.py)
  -> monitoring condition  rho_hat_t > C_t  (no alarm)       [the theorem's premise, strict inequality]
  -> measured satisfaction  (1/T) sum_t 1[rho_t > 0]          [(s,t) |= phi  <=>  rho_t > 0]
  -> compare with 1 - delta - p1.

Reading used: the premise is an event about a whole run, so the theorem is checked PER RUN.
A run is a "premise run" if no alarm fires at any monitored step. The theorem says nothing about other runs;
their satisfaction is reported for context only.

Two experiments
  1 "trajectories"  : a unicycle drives among point obstacles; the predictor's noise grows with severity (OOD).
                      Realistic, but on premise runs satisfaction is ~1, so the bound is not tested near tightness.
  2 "stress"        : each step is a constructed scene (obstacle + 5-point windows) in which the predictor hides a
                      true violation with probability q (severity). This puts satisfaction on premise runs close
                      to the bound, and shows ACP raising alarms (premise stops holding) once hidden violations
                      become frequent. Split CP (no adaptation) is shown as a contrast.

Evidence, not proof: see the notes printed at the end and in the README discussion.
Usage: python theorem1_check.py [--seeds 100] [--lag 5] [--init cal|none]
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from acp import AdaptiveConformal, SplitConformal, run_lagged, theorem1_lower_bound
from stl import window_collision_robustness
import synthetic as syn

DELTA, GAMMA, T0 = 0.1, 0.005, 15
H = 5
# experiment settings
TRAJ = dict(levels=[0.05, 0.5, 1.0, 2.0], T=300, n_obs=2, c=0.5, label="predictor noise sigma (OOD segment)")
STRESS = dict(levels=[0.02, 0.05, 0.08, 0.10, 0.12, 0.20], T=1500, c=5.0, label="hidden-violation probability q")


# ----------------------------------------------------------------------------
# data generators (all robustness values come from stl.window_collision_robustness)
# ----------------------------------------------------------------------------
def trajectory_world(rng, level, T=300, n_obs=2, c=0.5):
    """Unicycle trajectory among obstacles. Returns (rho_hat, rho, cal_errors)."""
    cal_hat, cal_rho, _ = syn.trajectory_series(rng, 300, H=H, n_obs=n_obs, c=c, sigma_id=0.05, sigma_ood=0.05, ood=False)
    rho_hat, rho, _ = syn.trajectory_series(rng, T + T0, H=H, n_obs=n_obs, c=c, sigma_id=0.05, sigma_ood=level)
    return rho_hat, rho, cal_hat - cal_rho


def stress_scenes(rng, q, T=1500, c=5.0):
    """Constructed scenes: obstacle at the origin; the true and the predicted windows are 5 points on a ray, nearest
    point at distance c + rho (so the STL robustness equals the designed rho). With probability q the predictor
    misses a real violation: designed R_t ~ U(3.8, 6) (rho_t < 0 while rho_hat_t ~ 3), else R_t ~ N(0, 0.1).
    The robustness values are then recomputed by the STL module from the window geometry."""
    obstacles = np.array([[0.0, 0.0]])
    L = T + T0
    rho_hat_design = 3 + 0.1 * rng.normal(size=L)
    R = np.where(rng.random(L) < q, rng.uniform(3.8, 6.0, L), rng.normal(0, 0.1, L))
    rho_design = rho_hat_design - R
    k = 0.2 * np.arange(H)[:, None]
    rho_hat, rho = np.empty(L), np.empty(L)
    for t in range(L):
        pred = np.hstack([c + rho_hat_design[t] + k, np.zeros((H, 1))])
        true = np.hstack([c + rho_design[t] + k, np.zeros((H, 1))])
        rho_hat[t] = window_collision_robustness(pred, obstacles, c)
        rho[t] = window_collision_robustness(true, obstacles, c)
    cal = rng.normal(0, 0.1, 200)                                       # ID-only calibration errors
    return rho_hat, rho, cal


# ----------------------------------------------------------------------------
# analysis of ONE run
# ----------------------------------------------------------------------------
def analyse_run(rho_hat, rho, cal, method="acp", lag=H, t0=T0, init="cal"):
    if method == "acp":
        obj = AdaptiveConformal(DELTA, GAMMA, init_scores=cal if init == "cal" else ())
    elif method == "cp":
        obj = SplitConformal(DELTA, cal)
    else:
        raise ValueError(method)
    out = run_lagged(rho_hat, rho, obj, lag=lag, t0=t0)
    mon = ~np.isnan(out["e"])
    C, rh, r = out["C"][mon], rho_hat[mon], rho[mon]
    no_alarm = rh > C                                  # strict: the theorem's premise at each step
    sat = r > 0                                        # (s,t) |= phi  <=>  rho_t > 0
    covered = (rh - r) <= C                            # rho_hat_t - rho_t <= C_t, same C_t that gated the alarm
    T_eff = int(mon.sum())
    return dict(premise=bool(no_alarm.all()), sat=float(sat.mean()), alarm_rate=float((~no_alarm).mean()),
                T_eff=T_eff, bound=float(theorem1_lower_bound(T_eff, DELTA, GAMMA)),
                impl_viol=int((no_alarm & covered & ~sat).sum()))   # pure algebra: must be 0


def run_experiment(kind, seeds=100, lag=H, t0=T0, init="cal", methods=("acp", "cp"), seed0=7000, levels=None):
    cfg = TRAJ if kind == "trajectories" else STRESS
    levels = cfg["levels"] if levels is None else levels
    runs = []
    for li, level in enumerate(levels):
        for s in range(seeds):
            rng = np.random.default_rng(seed0 + 1000 * li + s)
            if kind == "trajectories":
                rho_hat, rho, cal = trajectory_world(rng, level, cfg["T"], cfg["n_obs"], cfg["c"])
            else:
                rho_hat, rho, cal = stress_scenes(rng, level, cfg["T"], cfg["c"])
            for m in methods:
                r = analyse_run(rho_hat, rho, cal, m, lag, t0, init)
                r.update(experiment=kind, level=level, method=m, seed=s)
                runs.append(r)
    return runs


def summarise(runs):
    rows = []
    keys = dict.fromkeys((r["experiment"], r["method"], r["level"]) for r in runs)
    for (ex, m, lv) in keys:
        rs = [r for r in runs if (r["experiment"], r["method"], r["level"]) == (ex, m, lv)]
        prem = [r for r in rs if r["premise"]]
        rows.append(dict(
            experiment=ex, method=m, level=lv, n_runs=len(rs), n_premise=len(prem),
            frac_premise=len(prem) / len(rs), T_eff=rs[0]["T_eff"], bound=rs[0]["bound"],
            sat_premise_mean=float(np.mean([r["sat"] for r in prem])) if prem else float("nan"),
            sat_premise_min=float(np.min([r["sat"] for r in prem])) if prem else float("nan"),
            n_premise_below_bound=int(sum(r["sat"] < r["bound"] for r in prem)),
            sat_nonpremise_mean=float(np.mean([r["sat"] for r in rs if not r["premise"]])) if len(prem) < len(rs) else float("nan"),
            alarm_rate=float(np.mean([r["alarm_rate"] for r in rs])),
            impl_viol=int(sum(r["impl_viol"] for r in rs))))
    return rows


def to_markdown(rows):
    out = []
    for ex in dict.fromkeys(r["experiment"] for r in rows):
        out += [f"### Experiment: {ex}", "",
                "| method | severity | runs where condition holds | satisfaction on those runs: mean [min] | theorem bound | # of those runs below bound | satisfaction on other runs (context) | alarm rate | implication violations |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in (r for r in rows if r["experiment"] == ex):
            sp = "n/a" if r["n_premise"] == 0 else f"{r['sat_premise_mean']:.3f} [{r['sat_premise_min']:.3f}]"
            so = "n/a" if np.isnan(r["sat_nonpremise_mean"]) else f"{r['sat_nonpremise_mean']:.3f}"
            out.append(f"| {r['method'].upper()} | {r['level']} | {r['n_premise']}/{r['n_runs']} | {sp} | {r['bound']:.3f} | "
                       f"{r['n_premise_below_bound']} | {so} | {r['alarm_rate']:.3f} | {r['impl_viol']} |")
        out.append("")
    return "\n".join(out)


def plot(runs, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey="row")
    rng = np.random.default_rng(0)
    for i, kind in enumerate(("trajectories", "stress")):
        cfg = TRAJ if kind == "trajectories" else STRESS
        for j, m in enumerate(("acp", "cp")):
            ax = axes[i, j]
            rs = [r for r in runs if r["experiment"] == kind and r["method"] == m]
            xi = np.array([cfg["levels"].index(r["level"]) for r in rs], float) + rng.uniform(-0.3, 0.3, len(rs))
            y = np.array([r["sat"] for r in rs]); prem = np.array([r["premise"] for r in rs])
            ax.scatter(xi[~prem], y[~prem], s=14, facecolors="none", edgecolors="gray", alpha=0.7,
                       label="theorem condition FAILS (alarm raised) -- theorem silent")
            ax.scatter(xi[prem], y[prem], s=16, color="tab:blue" if m == "acp" else "tab:red", alpha=0.85,
                       label="theorem condition HOLDS (no alarm all run)")
            ax.axhline(rs[0]["bound"], color="k", ls="--", label=f"Theorem 1 bound 1-δ-p1 = {rs[0]['bound']:.3f}")
            ax.set_xticks(range(len(cfg["levels"]))); ax.set_xticklabels(cfg["levels"])
            ax.set_xlabel(cfg["label"]); ax.set_ylabel("measured satisfaction rate per run")
            ax.set_title(f"{'Experiment 1: trajectories' if i == 0 else 'Experiment 2: stress test'} -- {m.upper()}")
            ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--lag", type=int, default=H)
    ap.add_argument("--t0", type=int, default=T0)
    ap.add_argument("--init", choices=["cal", "none"], default="cal")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = run_experiment("trajectories", a.seeds, a.lag, a.t0, a.init) + run_experiment("stress", a.seeds, a.lag, a.t0, a.init)
    rows = summarise(runs)
    md = to_markdown(rows)
    head = (f"# Theorem 1 check: delta={DELTA}, gamma={GAMMA}, seeds={a.seeds}, lag={a.lag}, t0={a.t0}, init={a.init}\n\n"
            "Condition = no alarm at any monitored step (rho_hat_t > C_t for all t). Satisfaction = mean 1[rho_t > 0].\n"
            "'Implication violations' counts steps with no alarm, rho_hat_t - rho_t <= C_t, yet rho_t <= 0: must be 0 (algebra).\n\n")
    print("\n" + head + md)
    with open(os.path.join(a.out, "theorem1_check.md"), "w") as f:
        f.write(head + md + "\n")
    with open(os.path.join(a.out, "theorem1_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    plot(runs, os.path.join(a.out, "theorem1_check.png"))

    acp_rows = [r for r in rows if r["method"] == "acp"]
    print("ACP: runs where the condition holds =", sum(r["n_premise"] for r in acp_rows),
          "| of those, below the bound =", sum(r["n_premise_below_bound"] for r in acp_rows),
          "| implication violations =", sum(r["impl_viol"] for r in rows))
    print("Saved theorem1_check.md, theorem1_check.csv, theorem1_check.png in ./" + a.out)


if __name__ == "__main__":
    main()