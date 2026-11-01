"""
compare_lag_aware_acp.py -- fair comparison: ORIGINAL ACP (acp.run_lagged) vs LAG-AWARE ACP (acp_lag_aware).

Fairness
  * the same (rho_hat, rho, calibration set) is generated ONCE per (scenario, seed) by the generators of
    acp_coverage.py / synthetic.py (same seeds 2000+s as acp_coverage.py) and fed to every method and lag;
  * both methods use the same delta, gamma, t0, warm start and update-order flag;
  * every metric is keyed to ISSUED predictions in ONE common evaluation window
        [max(t0, max(lags)), T-1-max(lags)]
    so H=0 and H=5 rows are computed on exactly the same predictions;
  * all numbers are computed by this script; nothing is hard-coded.

Metric naming (never confuse the two coverages!)
  deployed_cov        coverage of the region that was ACTUALLY deployed: 1[ S_t <= C_issue[t] ]  (issuance threshold)
  scored_cov_arrival  coverage under the ORIGINAL scoring convention: 1[ S_t <= C at ARRIVAL time t+H ]
  scored_cov_native   (original only) 1 - mean(e) exactly as acp_coverage.py reports it (all scored steps)

Usage:  python compare_lag_aware_acp.py [--seeds 10] [--T 1500] [--lags 0 5] [--t0 15] [--out results_lag_aware]
"""
import argparse
import hashlib
import warnings
import importlib.util
import json
import os
import platform
import sys

import numpy as np
import pandas as pd


def _ensure_stl():
    """synthetic.py needs `stl.window_collision_robustness`. Use the project's module if importable,
    otherwise the clearly-labelled stand-in in _stl_fallback.py."""
    if importlib.util.find_spec("stl") is None:
        import _stl_fallback
        sys.modules["stl"] = _stl_fallback
        return "STAND-IN _stl_fallback.py (project stl.py not importable)"
    return "project stl module"


STL_SOURCE = _ensure_stl()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from acp import AdaptiveConformal, run_lagged, lemma1_bounds  # noqa: E402
from acp_coverage import get_series, get_cal, DELTA, GAMMA  # noqa: E402
from acp_lag_aware import run_lag_aware, deployed_view  # noqa: E402

SCENARIOS = ["iid_id", "trajectory_id", "abrupt_shift", "gradual_drift", "adversarial_monotone",
             "trajectory_noise_shift", "ar1_dependent", "heavy_tail_switch"]
REQUIRED = ["iid_id", "abrupt_shift", "gradual_drift", "adversarial_monotone", "trajectory_noise_shift"]
SHIFT_ONSET = {                                   # issue index at which the shift starts (None = no discrete onset)
    "abrupt_shift": lambda T: T // 3,             # synthetic.error_sequence: R[T//3:] ~ N(0.6, 0.5)
    "heavy_tail_switch": lambda T: T // 2,        # R[T//2:] scaled/shifted
    "trajectory_noise_shift": lambda T: T // 3,   # noise ramps from sigma_id to sigma_ood on [T/3, 2T/3]
}
METHODS = ["original_acp", "lag_aware_acp"]


# ----------------------------------------------------------------------------- metrics
def forward_window_stats(e_dep, onset, end, delta, w=100, tol=0.05):
    """Response to a shift. Forward windows [t, t+w) of DEPLOYED coverage, t >= onset, all inside [.., end].
    delay      = first t - onset with window coverage >= 1 - delta - tol      (NaN + censored=True if never)
    min_fwd    = minimum forward-window coverage after onset
    post_cov   = deployed coverage over all predictions issued in [onset, end]"""
    seg = 1.0 - e_dep[onset:end + 1]
    n = len(seg)
    post = float(seg.mean())
    if n < w:
        return np.nan, True, np.nan, post
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    fwd = (cs[w:] - cs[:-w]) / w                       # fwd[k] = coverage of [onset+k, onset+k+w)
    hit = np.nonzero(fwd >= 1 - delta - tol)[0]
    if len(hit) == 0:
        return np.nan, True, float(fwd.min()), post
    return float(hit[0]), False, float(fwd.min()), post


def method_metrics(view, rho_hat, rho, C_issue, delta_issue, lag, window, onset, T, gamma, delta):
    ws, we = window
    idx = np.arange(T)
    m = (idx >= ws) & (idx <= we)
    assert np.all(view["evaluated"][m]), "evaluation window must lie inside the evaluated range"
    e_dep, e_arr, alarm = view["e_deployed"], view["e_arrival"], view["alarm"]
    viol = rho <= 0
    a, v = alarm[m], viol[m]
    cov_dep = 1 - e_dep[m].mean()
    cov_arr = 1 - e_arr[m].mean()
    full = view["evaluated"]
    Nf = int(full.sum())
    cov_full = 1 - np.nanmean(e_dep[full]) if Nf else np.nan
    lo0, hi0 = lemma1_bounds(Nf, delta, gamma) if Nf else (np.nan, np.nan)
    Cm = C_issue[m]
    d = delta_issue[~np.isnan(delta_issue)]
    row = dict(
        n_eval=int(m.sum()),
        deployed_cov=cov_dep, scored_cov_arrival=cov_arr,
        absdev_deployed=abs(cov_dep - (1 - delta)), absdev_scored_arrival=abs(cov_arr - (1 - delta)),
        gap_scored_minus_deployed=cov_arr - cov_dep,
        alarm_rate=a.mean(), violation_rate=v.mean(),
        recall=(a & v).sum() / v.sum() if v.sum() else np.nan,
        precision=(a & v).sum() / a.sum() if a.sum() else np.nan,
        mean_C_finite=float(np.mean(Cm[np.isfinite(Cm)])) if np.isfinite(Cm).any() else np.nan,
        frac_C_inf=float(np.mean(~np.isfinite(Cm))),
        delta_t_min=float(d.min()), delta_t_max=float(d.max()),
        N_full=Nf, deployed_cov_full=cov_full,
        # This is only a diagnostic against the original envelope, NOT a guarantee
        # for the lag-aware method.
        full_in_original_lemma1_envelope_diagnostic=bool(lo0 - 1e-12 <= cov_full <= hi0 + 1e-12),
    )
    if onset is not None:
        dl, cens, mn, post = forward_window_stats(e_dep, onset, we, delta)
        row.update(response_delay=dl, response_censored=cens, min_forward_cov_post_shift=mn,
                   post_shift_deployed_cov=post)
    else:
        row.update(response_delay=np.nan, response_censored=np.nan, min_forward_cov_post_shift=np.nan,
                   post_shift_deployed_cov=np.nan)
    return row


def rolling_mean(x, lo, hi, w):
    """trailing mean of x[t-w+1..t] for t in [lo+w-1, hi]; NaN elsewhere."""
    out = np.full(len(x), np.nan)
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    for t in range(lo + w - 1, hi + 1):
        out[t] = (cs[t + 1] - cs[t + 1 - w]) / w
    return out


# ----------------------------------------------------------------------------- experiment
def run(args):
    lags = list(args.lags)
    Hmax = max(lags)
    window = (max(args.t0, Hmax), args.T - 1 - Hmax)
    rows, checks = [], []
    store = {}      # (scenario, H, method) -> dict(C=[seeds,T], delta=[seeds,T], e_dep=[seeds,T])

    for name in args.scenarios:
        onset_fn = SHIFT_ONSET.get(name)
        onset = onset_fn(args.T) if onset_fn else None
        for s in range(args.seeds):
            rng = np.random.default_rng(args.seed0 + s)
            cal = get_cal(rng, name)                            # same call order as acp_coverage.run_experiment
            rho_hat, rho = get_series(rng, name, args.T)
            for H in lags:
                # ---- ORIGINAL: existing implementation, existing timing convention
                ctl = AdaptiveConformal(DELTA, GAMMA, init_scores=cal, include_current=args.literal)
                out = run_lagged(rho_hat, rho, ctl, lag=H, t0=args.t0)
                v_o = deployed_view(rho_hat, rho, out["C"], H, out["start"])
                # ---- LAG-AWARE: issuance-threshold evaluation drives the controller
                la = run_lag_aware(rho_hat, rho, DELTA, GAMMA, H, args.t0, cal, args.literal, "issuance")
                v_l = deployed_view(rho_hat, rho, la["C_issue"], H, la["start"])
                # ---- diagnostic: arrival-time update with issuance-time evaluation == original dynamics
                dg = run_lag_aware(rho_hat, rho, DELTA, GAMMA, H, args.t0, cal, args.literal, "arrival")

                f = lambda x: np.nan_to_num(x, nan=7e9, posinf=8e9, neginf=9e9)
                checks.append(dict(
                    scenario=name, seed=args.seed0 + s, H=H,
                    arrival_variant_C_identical_to_original=bool(np.array_equal(f(dg["C_issue"]), f(out["C"]))),
                    arrival_variant_delta_identical_to_original=bool(np.array_equal(f(dg["delta_issue"]), f(out["delta_t"]))),
                    orig_e_matches_arrival_convention=bool(np.array_equal(
                        out["e"][np.arange(args.T)[v_o["evaluated"]] + H], v_o["e_arrival"][v_o["evaluated"]])),
                    lag_aware_C_identical_to_original=bool(np.array_equal(f(la["C_issue"]), f(out["C"]))),
                    lag_aware_internal_e_equals_view=bool(np.array_equal(
                        np.nan_to_num(la["e_update"][v_l["evaluated"]], nan=-1), v_l["e_deployed"][v_l["evaluated"]])),
                ))
                for meth, view, C, dl in (("original_acp", v_o, out["C"], out["delta_t"]),
                                          ("lag_aware_acp", v_l, la["C_issue"], la["delta_issue"])):
                    r = dict(scenario=name, seed=args.seed0 + s, H=H, method=meth)
                    r.update(method_metrics(view, rho_hat, rho, C, dl, H, window, onset, args.T, GAMMA, DELTA))
                    if meth == "original_acp":
                        e = out["e"][~np.isnan(out["e"])]
                        r["scored_cov_native"] = 1 - e.mean() if len(e) else np.nan
                    else:
                        r["scored_cov_native"] = np.nan
                    rows.append(r)
                    st = store.setdefault((name, H, meth), dict(C=[], delta=[], e_dep=[]))
                    st["C"].append(C.copy()); st["delta"].append(dl.copy()); st["e_dep"].append(view["e_deployed"].copy())
    return pd.DataFrame(rows), pd.DataFrame(checks), store, window


# ----------------------------------------------------------------------------- tables
AGG = ["deployed_cov", "scored_cov_arrival", "absdev_deployed", "absdev_scored_arrival", "gap_scored_minus_deployed",
       "alarm_rate", "violation_rate", "recall", "precision", "mean_C_finite", "frac_C_inf", "delta_t_min",
       "delta_t_max", "deployed_cov_full", "response_delay", "min_forward_cov_post_shift", "post_shift_deployed_cov"]


def summarize(df):
    g = df.groupby(["scenario", "H", "method"], sort=False)
    out = g[AGG].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    extra = g.agg(n_seeds=("seed", "count"), n_eval=("n_eval", "first"),
                  n_censored=("response_censored", lambda x: int(np.nansum(x.astype(float)))),
                  n_full_outside_original_lemma1_diagnostic=("full_in_original_lemma1_envelope_diagnostic",
                                                               lambda x: int((~x).sum())))
    return pd.concat([extra, out], axis=1).reset_index()


def paired(df):
    """lag_aware - original, paired by seed. Negative absdev/alarm/delay difference = lag-aware lower."""
    o = df[df.method == "original_acp"].set_index(["scenario", "H", "seed"])
    l = df[df.method == "lag_aware_acp"].set_index(["scenario", "H", "seed"])
    cols = ["absdev_deployed", "deployed_cov", "alarm_rate", "recall", "precision", "response_delay",
            "min_forward_cov_post_shift", "post_shift_deployed_cov", "mean_C_finite"]
    d = (l[cols] - o[cols]).reset_index()
    rows = []
    for (sc, H), g in d.groupby(["scenario", "H"], sort=False):
        r = dict(scenario=sc, H=H, n_seeds=len(g))
        for c in cols:
            x = g[c].dropna()
            r[f"diff_{c}_mean"] = x.mean() if len(x) else np.nan
            r[f"diff_{c}_std"] = x.std() if len(x) > 1 else np.nan
        x = g["absdev_deployed"]
        r["seeds_lagaware_closer_to_target"] = int((x < -1e-12).sum())
        r["seeds_lagaware_farther_from_target"] = int((x > 1e-12).sum())
        r["seeds_identical_absdev"] = int((x.abs() <= 1e-12).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def markdown_table(summ):
    cols = [("deployed_cov_mean", "deployed cov"), ("scored_cov_arrival_mean", "scored cov (arrival conv.)"),
            ("absdev_deployed_mean", "|deployed-0.9|"), ("alarm_rate_mean", "alarm rate"),
            ("recall_mean", "recall"), ("precision_mean", "precision"), ("response_delay_mean", "resp. delay")]
    lines = ["| scenario | H | method | " + " | ".join(c[1] for c in cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    for _, r in summ.iterrows():
        vals = []
        for c, _n in cols:
            x = r[c]
            vals.append("-" if pd.isna(x) else f"{x:.3f}")
        lines.append(f"| {r['scenario']} | {r['H']} | {r['method']} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- plots
COL = {"original_acp": "tab:blue", "lag_aware_acp": "tab:orange"}


def finite_mean(a):
    a = np.where(np.isfinite(a), a, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN columns (e.g. t < start) -> NaN
        return np.nanmean(a, axis=0)


def plot_traces(store, scenarios, lags, key, ylabel, fname, window, title, hline=None):
    fig, axes = plt.subplots(len(scenarios), len(lags), figsize=(5.2 * len(lags), 2.2 * len(scenarios)),
                             sharex=True, squeeze=False)
    for i, sc in enumerate(scenarios):
        for j, H in enumerate(lags):
            ax = axes[i, j]
            for meth in METHODS:
                arr = np.array(store[(sc, H, meth)][key])
                ax.plot(finite_mean(arr), color=COL[meth], lw=1.1, label=meth)
            if hline is not None:
                ax.axhline(hline, color="k", ls=":", lw=0.8)
            ax.axvspan(window[0], window[1], color="gray", alpha=0.07)
            if SHIFT_ONSET.get(sc):
                ax.axvline(SHIFT_ONSET[sc](arr.shape[1]), color="red", ls="--", lw=0.8)
            ax.set_title(f"{sc}  (H={H})", fontsize=8)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)
    axes[0, 0].legend(fontsize=7)
    axes[-1, 0].set_xlabel("issue time t")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def plot_rolling(store, scenarios, lags, window, w, fname, T):
    fig, axes = plt.subplots(len(scenarios), len(lags), figsize=(5.2 * len(lags), 2.2 * len(scenarios)),
                             sharex=True, squeeze=False)
    for i, sc in enumerate(scenarios):
        for j, H in enumerate(lags):
            ax = axes[i, j]
            for meth in METHODS:
                e = np.array(store[(sc, H, meth)]["e_dep"])
                lo = max(window[0], 0)
                rm = np.array([rolling_mean(1 - x, lo, window[1], w) for x in e])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ax.plot(np.nanmean(rm, axis=0), color=COL[meth], lw=1.1, label=meth)
            ax.axhline(1 - DELTA, color="k", ls=":", lw=0.8)
            if SHIFT_ONSET.get(sc):
                ax.axvline(SHIFT_ONSET[sc](T), color="red", ls="--", lw=0.8)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{sc}  (H={H})", fontsize=8)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(f"rolling DEPLOYED cov (w={w})", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    axes[-1, 0].set_xlabel("issue time t")
    fig.suptitle("Rolling deployed coverage (mean over seeds); dotted = target 1-delta, red dashed = shift onset", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def plot_coverage_bars(df, scenarios, lags, fname):
    fig, axes = plt.subplots(1, len(lags), figsize=(6.5 * len(lags), 4.2), sharey=True, squeeze=False)
    series = [("original_acp", "deployed_cov", "orig: DEPLOYED", "tab:blue", None),
              ("original_acp", "scored_cov_arrival", "orig: scored (arrival conv.)", "tab:blue", "//"),
              ("lag_aware_acp", "deployed_cov", "lag-aware: DEPLOYED", "tab:orange", None),
              ("lag_aware_acp", "scored_cov_arrival", "lag-aware: counterfactual arrival conv.", "tab:orange", "//")]
    x = np.arange(len(scenarios)); wd = 0.2
    for j, H in enumerate(lags):
        ax = axes[0, j]
        for k, (meth, col, lab, c, h) in enumerate(series):
            g = df[(df.H == H) & (df.method == meth)].groupby("scenario")[col].agg(["mean", "std"]).reindex(scenarios)
            ax.bar(x + (k - 1.5) * wd, g["mean"], wd, yerr=g["std"], color=c, hatch=h, alpha=0.8 if h is None else 0.45,
                   label=lab, capsize=1.5, edgecolor="k", linewidth=0.3)
        ax.axhline(1 - DELTA, color="red", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=35, ha="right", fontsize=7)
        ax.set_ylim(0.75, 0.97); ax.set_title(f"H={H}  (evaluation window, mean ± std over seeds)", fontsize=9)
    axes[0, 0].set_ylabel("coverage"); axes[0, 0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout(); fig.savefig(fname, dpi=140); plt.close(fig)


def plot_delay(df, lags, fname):
    shift = [s for s in SHIFT_ONSET if s in set(df.scenario)]
    if not shift:
        return
    fig, axes = plt.subplots(1, len(lags), figsize=(5 * len(lags), 3.6), sharey=True, squeeze=False)
    x = np.arange(len(shift)); wd = 0.35
    for j, H in enumerate(lags):
        ax = axes[0, j]
        for k, meth in enumerate(METHODS):
            g = df[(df.H == H) & (df.method == meth) & df.scenario.isin(shift)].groupby("scenario")
            m = g["response_delay"].mean().reindex(shift); sd = g["response_delay"].std().reindex(shift)
            cens = g["response_censored"].apply(lambda s: int(np.nansum(s.astype(float)))).reindex(shift)
            ax.bar(x + (k - .5) * wd, m, wd, yerr=sd, color=COL[meth], label=meth, capsize=2)
            for xi, (mi, ci) in enumerate(zip(m, cens)):
                ax.text(xi + (k - .5) * wd, 0 if np.isnan(mi) else mi, f"c={ci}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels(shift, fontsize=7, rotation=15)
        ax.set_title(f"H={H}", fontsize=9)
    axes[0, 0].set_ylabel("response delay (steps)\nmean over non-censored seeds; c = #censored")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Delay until forward-window deployed coverage >= 1-delta-0.05 (w=100) after shift onset", fontsize=9)
    fig.tight_layout(); fig.savefig(fname, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--lags", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--t0", type=int, default=15)
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--literal", action="store_true", help="literal Algorithm-2 order (include_current=True)")
    ap.add_argument("--out", default="results_lag_aware")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    df, checks, store, window = run(args)
    summ, pair = summarize(df), paired(df)
    df.to_csv(os.path.join(args.out, "per_seed_metrics.csv"), index=False)
    summ.to_csv(os.path.join(args.out, "summary_by_scenario.csv"), index=False)
    pair.to_csv(os.path.join(args.out, "paired_differences.csv"), index=False)
    checks.to_csv(os.path.join(args.out, "consistency_checks.csv"), index=False)
    with open(os.path.join(args.out, "summary_table.md"), "w") as f:
        f.write(markdown_table(summ))

    # threshold / delta evolution (seed-mean) as CSV
    tr = []
    for (sc, H, meth), st in store.items():
        C = np.array(st["C"]); D = np.array(st["delta"])
        cm = finite_mean(C); nf = np.isfinite(C).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dm = np.nanmean(D, axis=0)
        tr.append(pd.DataFrame(dict(scenario=sc, H=H, method=meth, t=np.arange(C.shape[1]),
                                    C_issue_mean_finite=cm, delta_issue_mean=dm, n_seeds_C_finite=nf)))
    pd.concat(tr).to_csv(os.path.join(args.out, "threshold_evolution_mean.csv"), index=False)

    sc_plot = [s for s in args.scenarios]
    plot_traces(store, sc_plot, args.lags, "C", "mean C at issuance", os.path.join(args.out, "threshold_evolution.png"),
                window, "Threshold C_issue[t] over time (mean over seeds, non-finite excluded); shaded = common evaluation window")
    plot_traces(store, sc_plot, args.lags, "delta", "mean delta_t", os.path.join(args.out, "delta_evolution.png"),
                window, "Controller level delta_t over time (mean over seeds)", hline=DELTA)
    plot_rolling(store, sc_plot, args.lags, window, 100, os.path.join(args.out, "rolling_deployed_coverage.png"), args.T)
    plot_coverage_bars(df, sc_plot, args.lags, os.path.join(args.out, "coverage_deployed_vs_scored.png"))
    plot_delay(df, args.lags, os.path.join(args.out, "response_delay.png"))
    np.savez_compressed(os.path.join(args.out, "traces_per_seed.npz"),
                        **{f"{sc}|H{H}|{m}|{k}": np.array(v) for (sc, H, m), st in store.items() for k, v in st.items()})

    with open(os.path.join(args.out, "run_metadata.json"), "w") as f:
        json.dump(dict(args=vars(args), delta=DELTA, gamma=GAMMA, evaluation_window=window, stl_source=STL_SOURCE,
                       python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                       acp_py_sha256=hashlib.sha256(open("acp.py", "rb").read()).hexdigest()), f, indent=2)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"STL source: {STL_SOURCE}\nevaluation window (issue index): {window}\n")
    print(summ[["scenario", "H", "method", "deployed_cov_mean", "scored_cov_arrival_mean", "absdev_deployed_mean",
                "alarm_rate_mean", "recall_mean", "precision_mean", "response_delay_mean", "n_censored"]]
          .round(3).to_string(index=False))
    print("\nconsistency checks, all seeds/scenarios, by lag H (expected: lag_aware_C_identical_to_original is True only at H=0):")
    print(checks.drop(columns=["scenario", "seed"]).groupby("H").all().T.to_string())
    print(f"\nSaved to ./{args.out}/")
    return df, summ, pair, checks


if __name__ == "__main__":
    main()
