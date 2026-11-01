"""Tests for acp.py: arithmetic, edge cases, deterministic invariants, lag handling, mutation tests."""
import math
import numpy as np
import pytest
from acp import (AdaptiveConformal, SplitConformal, conformal_threshold, run_lagged, lemma1_bounds,
                 theorem1_lower_bound, gibbs_candes_bound, predicts_violation, prediction_region)


# ---- threshold index conventions ------------------------------------------------
def test_threshold_quantile_index():
    s = list(range(1, 11))                         # 1..10, n=10
    assert conformal_threshold(s, 0.1) == 9        # ceil(10*0.9)=9 -> 9th smallest
    assert conformal_threshold(s, 0.0) == 10       # ceil(10)=10 -> max
    assert conformal_threshold(s, 0.5) == 5
    assert conformal_threshold(s, 1.0) == 1        # k=0 clipped to 1
    assert conformal_threshold(s, 0.95) == 1       # ceil(0.5)=1


def test_threshold_edge_conventions():
    s = [1.0, 2.0, 3.0]
    assert conformal_threshold(s, -0.001) == math.inf        # delta_t < 0 -> everything covered
    assert conformal_threshold(s, 1.001) == -math.inf        # delta_t > 1 -> empty region
    assert conformal_threshold([], 0.1) == math.inf          # no scores yet
    assert conformal_threshold([5.0], 0.3) == 5.0            # n=1


def test_float_ceiling_guard():
    s = list(range(1, 11))
    # 10*(1-0.1) is 9.000000000000002 in floating point; must still pick the 9th element
    assert conformal_threshold(s, 0.1) == 9


# ---- update rule ------------------------------------------------------------------
def test_delta_update_arithmetic():
    acp = AdaptiveConformal(delta=0.1, gamma=0.01, init_scores=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    C, e = acp.step(100.0)                         # big score -> error
    assert (C, e) == (9.0, 1)
    assert acp.delta_t == pytest.approx(0.1 + 0.01 * (0.1 - 1))
    C, e = acp.step(0.0)                           # small score -> covered
    assert e == 0
    assert acp.delta_t == pytest.approx(0.1 + 0.01 * (0.1 - 1) + 0.01 * 0.1)


def test_error_indicator_boundary_is_covered():
    acp = AdaptiveConformal(0.1, 0.01, init_scores=[1.0, 2.0, 3.0])
    C, _ = AdaptiveConformal(0.1, 0.01, init_scores=[1.0, 2.0, 3.0]).step(2.0)
    assert acp.step(C)[1] == 0                     # R_t == C_t counts as covered (R_t <= C_t)


def test_include_current_flag_changes_threshold():
    a = AdaptiveConformal(0.1, 0.01, init_scores=[1, 2, 3], include_current=False)
    b = AdaptiveConformal(0.1, 0.01, init_scores=[1, 2, 3], include_current=True)
    assert a.step(10.0)[0] == 3.0                  # threshold before seeing 10
    assert b.step(10.0)[0] == 10.0                 # literal Alg. 2 order: 10 is already in the set


def test_validation_errors():
    with pytest.raises(ValueError): AdaptiveConformal(delta=0.0)
    with pytest.raises(ValueError): AdaptiveConformal(delta=1.0)
    with pytest.raises(ValueError): AdaptiveConformal(gamma=0.0)
    with pytest.raises(ValueError): AdaptiveConformal().step(float("nan"))
    with pytest.raises(ValueError): AdaptiveConformal().accumulate(float("inf"))
    with pytest.raises(ValueError): SplitConformal(0.1, [])
    with pytest.raises(ValueError): run_lagged([1, 2], [1], AdaptiveConformal())


def test_empty_set_gives_infinite_region_and_no_error():
    acp = AdaptiveConformal()
    C, e = acp.step(3.0)
    assert C == math.inf and e == 0


# ---- deterministic invariants (hold for ANY score sequence) ----------------------
SEQS = {
    "increasing": lambda rng, T: np.linspace(0, 50, T),
    "decreasing": lambda rng, T: np.linspace(50, 0, T),
    "constant": lambda rng, T: np.full(T, 1.0),
    "alternating_extremes": lambda rng, T: np.where(np.arange(T) % 2 == 0, 1e6, -1e6).astype(float),
    "heavy_tail": lambda rng, T: rng.standard_cauchy(T),
    "iid": lambda rng, T: rng.normal(size=T),
}


@pytest.mark.parametrize("include_current", [False, True])
@pytest.mark.parametrize("name", list(SEQS))
@pytest.mark.parametrize("gamma", [0.001, 0.005, 0.05, 0.5])
def test_delta_t_stays_in_band_and_coverage_in_envelope(name, gamma, include_current):
    rng = np.random.default_rng(0)
    T, delta = 3000, 0.1
    scores = SEQS[name](rng, T)
    acp = AdaptiveConformal(delta, gamma, include_current=include_current)
    es, deltas = [], []
    for r in scores:
        deltas.append(acp.delta_t)
        es.append(acp.step(r)[1])
    deltas.append(acp.delta_t)
    assert min(deltas) >= -gamma - 1e-12 and max(deltas) <= 1 + gamma + 1e-12   # invariance
    es = np.array(es); Ts = np.arange(1, T + 1)
    cov = 1 - np.cumsum(es) / Ts
    lo, hi = lemma1_bounds(Ts, delta, gamma)
    assert np.all(cov >= lo - 1e-12) and np.all(cov <= hi + 1e-12)               # Lemma-1 envelope, every prefix
    # also the original (symmetric) Gibbs-Candes bound
    assert np.all(np.abs(np.cumsum(es) / Ts - delta) <= gibbs_candes_bound(Ts, delta, gamma) + 1e-12)


def test_paper_p1_p2_values():
    lo, hi = lemma1_bounds(1000, 0.1, 0.005)
    assert lo == pytest.approx(0.9 - 0.105 / 5.0) and hi == pytest.approx(0.9 + 0.905 / 5.0)
    assert theorem1_lower_bound(1000, 0.1, 0.005) == pytest.approx(lo)
    lo2, hi2 = lemma1_bounds(10 ** 8, 0.1, 0.005)
    assert lo2 == pytest.approx(0.9, abs=1e-5) and hi2 == pytest.approx(0.9, abs=1e-5)   # limits p1,p2 -> 0


# ---- mutation tests: a wrong implementation MUST be caught ------------------------
class SignFlipped(AdaptiveConformal):
    def step(self, score):
        C = self.threshold(); import bisect; bisect.insort(self._scores, float(score))
        e = 0 if score <= C else 1
        self.delta_t = self.delta_t - self.gamma * (self.delta - e)   # wrong sign
        return C, e


def test_mutant_sign_flip_violates_envelope():
    rng = np.random.default_rng(0)
    T, delta, gamma = 3000, 0.1, 0.005
    acp = SignFlipped(delta, gamma)
    es = np.array([acp.step(r)[1] for r in 0.002 * np.arange(T) + rng.normal(0, 0.01, T)])
    cov = 1 - np.cumsum(es) / np.arange(1, T + 1)
    lo, hi = lemma1_bounds(np.arange(1, T + 1), delta, gamma)
    assert np.any((cov < lo) | (cov > hi))


def test_split_cp_fails_under_monotone_shift_but_acp_does_not():
    T, delta, gamma = 3000, 0.1, 0.005
    rng = np.random.default_rng(0)
    cal = rng.normal(0, 0.2, 100)
    shifted = 0.002 * np.arange(T) + rng.normal(0, 0.01, T)
    cp = SplitConformal(delta, cal); acp = AdaptiveConformal(delta, gamma, init_scores=cal)
    cp_cov = 1 - np.mean([cp.step(r)[1] for r in shifted])
    acp_cov = 1 - np.mean([acp.step(r)[1] for r in shifted])
    assert cp_cov < 0.5 and abs(acp_cov - 0.9) < 0.02


# ---- lag handling -------------------------------------------------------------------
def test_run_lagged_uses_lagged_score_and_start():
    T, lag = 20, 3
    rho = np.zeros(T); rho_hat = np.arange(T, dtype=float)       # R_s = s
    acp = AdaptiveConformal(0.1, 0.01)
    out = run_lagged(rho_hat, rho, acp, lag=lag, t0=6)
    assert out["start"] == 6
    assert np.isnan(out["C"][:6]).all() and np.isfinite(out["e"][6:]).all()
    # scores accumulated for t=3..5 are R=0,1,2 ; at t=6 score is R_{3}=3 and C comes from {0,1,2}
    assert out["C"][6] == 2.0


def test_run_lagged_lag_zero_equals_manual_loop():
    rng = np.random.default_rng(5)
    rho = rng.normal(2, 0.5, 200); rho_hat = rho + rng.normal(0, 0.3, 200)
    out = run_lagged(rho_hat, rho, AdaptiveConformal(0.1, 0.01), lag=0, t0=0)
    ref = AdaptiveConformal(0.1, 0.01)
    e_ref = [ref.step(rho_hat[t] - rho[t])[1] for t in range(200)]
    assert np.array_equal(out["e"], e_ref)


def test_alarm_and_region_helpers():
    assert predicts_violation(0.5, 0.5) and predicts_violation(0.4, 0.5) and not predicts_violation(0.6, 0.5)
    assert prediction_region(2.0, 0.5) == (1.5, math.inf)
    assert not predicts_violation(np.nan, 0.5)

def test_split_conformal_validates_delta_and_calibration_scores():
    with pytest.raises(ValueError):
        SplitConformal(0.0, [1.0, 2.0])
    with pytest.raises(ValueError):
        SplitConformal(1.0, [1.0, 2.0])
    with pytest.raises(ValueError):
        SplitConformal(0.1, [1.0, float("nan")])
