"""
Tests for acp_lag_aware.py / compare_lag_aware_acp.py.   Run:  python -m pytest -q test_acp_lag_aware.py

NOTE: passing tests establish properties of the CODE. They do not prove any theoretical guarantee.
The tests labelled "empirical" only confirm that claimed ranges were not violated on the instances tried.
"""
import hashlib
import math
import os

import numpy as np
import pandas as pd
import pytest

import compare_lag_aware_acp as cmp
from acp import AdaptiveConformal, run_lagged, lemma1_bounds, prediction_region
from acp_lag_aware import run_lag_aware, deployed_view

HERE = os.path.dirname(os.path.abspath(__file__))
NAN = float("nan")


def _series(seed=0, T=500, shift_at=200):
    rng = np.random.default_rng(seed)
    rho = rng.normal(2, 0.5, T)
    err = np.where(np.arange(T) >= shift_at, rng.normal(0.6, 0.5, T), rng.normal(0, 0.2, T))
    return rho + err, rho, rng.normal(0, 0.2, 100)


def _same(a, b):
    f = lambda x: np.nan_to_num(np.asarray(x, float), nan=7e9, posinf=8e9, neginf=9e9)
    return np.array_equal(f(a), f(b))


# ============================================================ independent brute-force reference of the SPEC
def reference(rho_hat, rho, delta, gamma, lag, t0, init, literal, mode):
    T = len(rho); start = max(t0, lag); S = rho_hat - rho
    scores = list(init); dt = delta
    thr_store, C, dis, eupd = {}, [NAN] * T, [NAN] * T, [NAN] * T

    def thr():
        n = len(scores)
        if n == 0 or dt < 0:
            return math.inf
        if dt > 1:
            return -math.inf
        k = min(max(math.ceil(round(n * (1 - dt), 9)), 1), n)
        return sorted(scores)[k - 1]

    for t in range(T):
        a = t - lag
        done = False
        if a >= 0 and literal:
            scores.append(S[a]); done = True
        if t >= start:
            C[t] = thr(); dis[t] = dt; thr_store[t] = C[t]
        if a >= 0:
            if t >= start:
                if mode == "issuance" and a >= start:
                    e = 0 if S[a] <= thr_store[a] else 1
                    dt = dt + gamma * (delta - e); eupd[a] = e
                elif mode == "arrival":
                    e = 0 if S[a] <= C[t] else 1
                    dt = dt + gamma * (delta - e); eupd[a] = e
            if not done:
                scores.append(S[a])
    return np.array(C), np.array(dis), np.array(eupd), dt


# ============================================================ 1. H = 0 agrees with the zero-lag baseline
@pytest.mark.parametrize("literal", [False, True])
@pytest.mark.parametrize("warm", [True, False])
def test_h0_matches_zero_lag_baseline(literal, warm):
    rh, rho, cal = _series()
    init = cal if warm else ()
    o = run_lagged(rh, rho, AdaptiveConformal(0.1, 0.005, init_scores=init, include_current=literal), lag=0, t0=15)
    for mode in ("issuance", "arrival"):
        r = run_lag_aware(rh, rho, 0.1, 0.005, 0, 15, init, literal, mode)
        assert _same(r["C_issue"], o["C"])
        assert _same(r["delta_issue"], o["delta_t"])
        ok = r["evaluated"]
        assert np.array_equal(r["e_deployed"][ok], o["e"][ok])         # deployed e == original scored e at H=0
        assert np.array_equal(r["e_arrival"][ok], r["e_deployed"][ok])  # the two conventions coincide at H=0


# ============================================================ 2. 'arrival' variant == original dynamics for any lag
@pytest.mark.parametrize("lag", [0, 1, 5, 20])
@pytest.mark.parametrize("literal", [False, True])
def test_arrival_variant_reproduces_original(lag, literal):
    rh, rho, cal = _series(seed=3)
    o = run_lagged(rh, rho, AdaptiveConformal(0.1, 0.005, init_scores=cal, include_current=literal), lag=lag, t0=15)
    r = run_lag_aware(rh, rho, 0.1, 0.005, lag, 15, cal, literal, "arrival")
    assert _same(r["C_issue"], o["C"]) and _same(r["delta_issue"], o["delta_t"])
    v = deployed_view(rh, rho, o["C"], lag, o["start"])
    ok = v["evaluated"]
    assert np.array_equal(o["e"][np.arange(len(rho))[ok] + lag], v["e_arrival"][ok])   # original e == arrival-convention e


# ============================================================ 3. H > 0: stored issuance threshold is what evaluates
@pytest.mark.parametrize("lag", [1, 3, 5, 12])
@pytest.mark.parametrize("literal", [False, True])
@pytest.mark.parametrize("mode", ["issuance", "arrival"])
def test_matches_independent_reference(lag, literal, mode):
    rh, rho, cal = _series(seed=lag)
    r = run_lag_aware(rh, rho, 0.1, 0.01, lag, 10, cal, literal, mode)
    C, dis, eupd, dfin = reference(rh, rho, 0.1, 0.01, lag, 10, cal, literal, mode)
    assert _same(r["C_issue"], C) and _same(r["delta_issue"], dis) and _same(r["e_update"], eupd)
    assert r["delta_final"] == pytest.approx(dfin, abs=1e-12)


def test_hand_computed_example_stored_threshold_differs_from_arrival_threshold():
    """lag=1, delta=.5, gamma=.25, scores S=[10,1.5,-5,.5,8], init scores [0,1,2,3].  Worked by hand:
       t=1 issue C=1.0 (stored); t=2 issue C=2.0; prediction 1 (S=1.5) arrives at t=2:
         stored-threshold evaluation: 1.5 > 1.0 -> MISS (e=1);  arrival-threshold evaluation: 1.5 <= 2.0 -> covered."""
    S = np.array([10.0, 1.5, -5.0, 0.5, 8.0])
    r = run_lag_aware(S, np.zeros(5), delta=0.5, gamma=0.25, lag=1, t0=0, init_scores=[0, 1, 2, 3])
    assert _same(r["C_issue"], [NAN, 1.0, 2.0, 2.0, 1.5])
    assert _same(r["delta_issue"], [NAN, 0.5, 0.5, 0.375, 0.5])
    assert _same(r["e_deployed"], [NAN, 1, 0, 0, NAN])          # prediction 1 is a deployed-region MISS
    assert _same(r["e_arrival"], [NAN, 0, 0, 0, NAN])           # ...but would be 'covered' under the arrival threshold
    assert list(r["alarm"]) == [False, False, True, True, False]
    assert r["delta_final"] == pytest.approx(0.625)
    # ORIGINAL timing, worked by hand: at t=1 it already updates on the arriving S_0=10 (C[1]=1 -> e=1, delta .5 -> .375),
    # so at t=2 its threshold is the 4th smallest of [0,1,2,3,10] = 3.0, and prediction 1 (S=1.5) is scored as covered with it.
    o = run_lagged(S, np.zeros(5), AdaptiveConformal(0.5, 0.25, init_scores=[0, 1, 2, 3]), lag=1, t0=0)
    assert _same(o["C"], [NAN, 1.0, 3.0, 1.5, 1.0])      # t=4: delta=.625, 7 scores, k=ceil(2.625)=3 -> 3rd smallest = 1.0
    assert _same(o["e"], [NAN, 1, 0, 0, 0])
    assert _same(o["delta_t"], [NAN, 0.5, 0.375, 0.5, 0.625])
    # lag-aware does NOT update on prediction 0 (never deployed: issued before start=1), so its delta path differs
    assert r["delta_issue"][2] == pytest.approx(0.5) and o["delta_t"][2] == pytest.approx(0.375)


def test_deployed_region_semantics_use_prediction_region():
    """e_deployed == 0  <=>  true robustness lies in the region [rho_hat - C_issue, inf) that was deployed."""
    rh, rho, cal = _series(seed=5)
    r = run_lag_aware(rh, rho, 0.1, 0.005, 5, 15, cal)
    for t in np.nonzero(r["evaluated"])[0]:
        lo, hi = prediction_region(rh[t], r["C_issue"][t])
        assert (lo <= rho[t] <= hi) == (r["e_deployed"][t] == 0)


# ============================================================ 4. delayed outcomes matched to the right predictions
@pytest.mark.parametrize("lag", [0, 1, 4, 5, 9])
def test_delayed_outcomes_matched(lag):
    T = 60
    rho = 0.25 * np.arange(T)                  # unique, recognisable
    rh = rho + np.arange(T) ** 1.5 + 1.0       # unique score per prediction
    r = run_lag_aware(rh, rho, 0.1, 0.005, lag, 5)
    log = r["arrival_log"]
    assert [x[1] for x in log] == list(range(T - lag))                       # each issue index once, in order
    assert all(t - a == lag for t, a, _, _ in log)                           # arrives exactly `lag` steps later
    assert all(s == pytest.approx(rh[a] - rho[a]) for _, a, s, _ in log)     # score belongs to ITS prediction
    assert max(x[0] for x in log) <= T - 1


# ============================================================ 5. determinism
def test_deterministic_for_fixed_seed():
    a = run_lag_aware(*_series(seed=9)[:2], 0.1, 0.005, 5, 15, _series(seed=9)[2])
    b = run_lag_aware(*_series(seed=9)[:2], 0.1, 0.005, 5, 15, _series(seed=9)[2])
    for k in ("C_issue", "delta_issue", "e_deployed", "e_arrival", "e_update", "alarm"):
        assert _same(a[k], b[k])
    assert a["arrival_log"] == b["arrival_log"] and a["delta_final"] == b["delta_final"]
    c = run_lag_aware(*_series(seed=10)[:2], 0.1, 0.005, 5, 15)
    assert not _same(a["C_issue"], c["C_issue"])                              # different seed -> different run


def test_experiment_script_is_deterministic(tmp_path):
    argv = ["--seeds", "2", "--T", "300", "--scenarios", "iid_id", "abrupt_shift", "adversarial_monotone"]
    d1, *_ = cmp.main(argv + ["--out", str(tmp_path / "a")])
    d2, *_ = cmp.main(argv + ["--out", str(tmp_path / "b")])
    pd.testing.assert_frame_equal(d1, d2)
    assert (tmp_path / "a" / "per_seed_metrics.csv").read_text() == (tmp_path / "b" / "per_seed_metrics.csv").read_text()


# ============================================================ 6. original files are unchanged
def test_original_files_unchanged_and_not_mutated():
    """Hashes were recorded when the project files were inspected. If you EDIT these files on purpose, regenerate
    original_files.sha256 (sha256sum acp.py acp_coverage.py synthetic.py > original_files.sha256)."""
    ref = {}
    for line in open(os.path.join(HERE, "original_files.sha256")):
        h, name = line.split()
        ref[name] = h
    assert set(ref) >= {"acp.py", "acp_coverage.py", "synthetic.py"}
    for name, h in ref.items():
        assert hashlib.sha256(open(os.path.join(HERE, name), "rb").read()).hexdigest() == h, f"{name} was modified"


def test_inputs_not_mutated_and_original_behaviour_unaffected():
    rh, rho, cal = _series(seed=2)
    rh0, rho0, cal0 = rh.copy(), rho.copy(), cal.copy()
    before = run_lagged(rh, rho, AdaptiveConformal(0.1, 0.005, init_scores=cal), lag=5, t0=15)
    run_lag_aware(rh, rho, 0.1, 0.005, 5, 15, cal)
    after = run_lagged(rh, rho, AdaptiveConformal(0.1, 0.005, init_scores=cal), lag=5, t0=15)
    assert np.array_equal(rh, rh0) and np.array_equal(rho, rho0) and np.array_equal(cal, cal0)
    assert _same(before["C"], after["C"]) and _same(before["e"], after["e"])


# ============================================================ 7. edge cases / initialisation / boundaries
def test_boundaries_and_pending_predictions():
    T, lag, t0 = 100, 5, 15
    rh, rho, cal = _series(T=T)
    r = run_lag_aware(rh, rho, 0.1, 0.005, lag, t0, cal)
    start = r["start"]; assert start == 15
    assert np.isnan(r["C_issue"][:start]).all() and not r["alarm"][:start].any()      # warm-up: no deployment
    assert np.isfinite(r["C_issue"][start:]).all()
    assert np.isnan(r["e_deployed"][:start]).all() and np.isnan(r["e_deployed"][T - lag:]).all()   # pending: no outcome yet
    assert np.isfinite(r["e_deployed"][start:T - lag]).all()
    assert r["evaluated"].sum() == T - lag - start
    assert not np.isnan(r["delta_issue"][start:]).any()


def test_warmup_predictions_do_not_update_in_issuance_mode_but_do_in_arrival_mode():
    rh, rho, cal = _series(T=200)
    ri = run_lag_aware(rh, rho, 0.1, 0.005, 5, 15, cal, update_error="issuance")
    ra = run_lag_aware(rh, rho, 0.1, 0.005, 5, 15, cal, update_error="arrival")
    assert np.isnan(ri["e_update"][:15]).all() and np.isfinite(ri["e_update"][15:195]).all()
    assert np.isnan(ra["e_update"][:10]).all() and np.isfinite(ra["e_update"][10:195]).all()   # arrivals at t>=start update


def test_t0_smaller_than_lag_uses_lag_as_start():
    rh, rho, _ = _series(T=100)
    r = run_lag_aware(rh, rho, 0.1, 0.005, lag=10, t0=3)
    assert r["start"] == 10
    o = run_lagged(rh, rho, AdaptiveConformal(0.1, 0.005), lag=10, t0=3)
    assert o["start"] == 10


def test_empty_init_first_threshold_is_infinite_and_alarms():
    S = np.array([1.0, 2.0, 3.0, 4.0])
    r = run_lag_aware(S, np.zeros(4), 0.1, 0.005, lag=0, t0=0)
    assert r["C_issue"][0] == math.inf and r["alarm"][0] and r["e_deployed"][0] == 0
    assert np.isfinite(r["C_issue"][1])


@pytest.mark.parametrize("T,lag,t0", [(0, 0, 0), (1, 0, 0), (1, 3, 0), (5, 5, 0), (5, 4, 0), (5, 0, 9), (7, 100, 0)])
def test_tiny_and_degenerate_lengths(T, lag, t0):
    rng = np.random.default_rng(0)
    rh, rho = rng.normal(size=T), rng.normal(size=T)
    for mode in ("issuance", "arrival"):
        r = run_lag_aware(rh, rho, 0.1, 0.005, lag, t0, update_error=mode)
        assert len(r["C_issue"]) == T
        n_eval = max(0, T - lag - max(t0, lag))
        assert int(r["evaluated"].sum()) == n_eval
        assert int(np.isfinite(r["e_deployed"]).sum()) == n_eval
        if n_eval == 0:
            assert r["delta_final"] == pytest.approx(0.1) or mode == "arrival"   # arrival mode may still update on warm arrivals


def test_invalid_inputs_raise():
    x = np.zeros(10)
    for kw, exc in [(dict(lag=-1), ValueError), (dict(t0=-1), ValueError), (dict(lag=1.5), TypeError),
                    (dict(lag=True), TypeError), (dict(delta=0.0), ValueError), (dict(delta=1.0), ValueError),
                    (dict(gamma=0.0), ValueError), (dict(gamma=float("inf")), ValueError),
                    (dict(update_error="x"), ValueError)]:
        with pytest.raises(exc):
            run_lag_aware(x, x, **kw)
    with pytest.raises(ValueError):
        run_lag_aware(x, x[:5])
    bad = x.copy(); bad[3] = np.nan
    with pytest.raises(ValueError):
        run_lag_aware(bad, x)
    with pytest.raises(ValueError):
        run_lag_aware(x, x, init_scores=[0.0, float("nan")])


# ============================================================ 8. response-delay helper (known answers)
def test_response_delay_known_answer_and_censoring():
    T, onset, end = 1000, 300, 900
    e = np.full(T, 0.0); e[onset:onset + 150] = 1.0                 # 150 consecutive misses, then all covered
    d, cens, mn, post = cmp.forward_window_stats(e, onset, end, 0.1, w=100, tol=0.05)
    assert (d, cens) == (135.0, False)                              # window [onset+k,+100) has min(100,150-k) misses; <=15 first at k=135
    assert mn == pytest.approx(0.0) and post == pytest.approx(1 - 150 / 601)
    e2 = np.ones(T)                                                  # never recovers
    d2, c2, *_ = cmp.forward_window_stats(e2, onset, end, 0.1)
    assert np.isnan(d2) and c2


# ============================================================ 9. theoretical caution / counterexamples
def test_delta_is_not_clipped_and_can_leave_unit_interval():
    """The unchanged ACP update is deliberately not clipped to [0,1]."""
    T = 80
    R = np.linspace(1.0, 80.0, T)  # all scores exceed the early issuance thresholds
    r = run_lag_aware(R, np.zeros(T), delta=0.05, gamma=0.05,
                      lag=5, t0=5, init_scores=[0.0] * 10)
    assert r["delta_final"] < 0.0
    # A controller value outside [0,1] is passed to the original threshold function,
    # which applies the baseline's documented +/- infinity convention.
    assert np.any(r["delta_issue"] < 0.0)


def test_original_lemma1_envelope_is_not_automatically_valid_for_lag_aware():
    """A deterministic counterexample to simply reusing the original Lemma-1 envelope.

    This is a counterexample for the tested finite sample, not a proof of any replacement
    theorem. No H-only lag-aware envelope is claimed by this implementation.
    """
    T = 60
    R = 0.05 * np.arange(T)
    init = [0.0] * 10
    r = run_lag_aware(R, np.zeros(T), 0.05, 0.05,
                      lag=20, t0=9, init_scores=init, update_error="issuance")
    N = int(r["evaluated"].sum())
    assert N == 20
    cov = 1 - np.nanmean(r["e_deployed"])
    lo, hi = lemma1_bounds(N, 0.05, 0.05)
    assert cov == pytest.approx(0.0)
    assert cov < lo
    # The unchanged update rule can drift outside [0,1]; this is not silently clipped.
    assert r["delta_final"] == pytest.approx(-0.9, abs=1e-9)

    # The original controller's native score/update convention is separately checked
    # against the original Lemma-1 envelope for this same example only.
    o = run_lagged(R, np.zeros(T), AdaptiveConformal(0.05, 0.05, init_scores=init),
                   lag=20, t0=9)
    e = o["e"][~np.isnan(o["e"])]
    lo_o, hi_o = lemma1_bounds(len(e), 0.05, 0.05)
    assert lo_o <= 1 - e.mean() <= hi_o
