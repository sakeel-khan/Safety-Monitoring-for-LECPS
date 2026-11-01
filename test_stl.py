"""Tests for stl.py: hand-computed cases, algebraic identities, rho>0 <=> sat, edge cases."""
import numpy as np
import pytest
from stl import (Predicate, Not, And, Or, Always, Eventually, Until, below, above,
                 abs_below, cartpole_spec, collision_avoidance_spec, window_collision_robustness)


def sig(vals):
    return np.array(vals, float).reshape(-1, 1)


# ---- hand-computed values --------------------------------------------------
def test_predicate_is_f_of_state():
    mu = above(0, 1.0)
    assert mu.rho(sig([3, 0]), 0) == 2.0 and mu.rho(sig([3, 0]), 1) == -1.0


def test_always_is_window_min_eventually_is_window_max():
    s = sig([3, 1, 2, 5, 4])
    mu = above(0, 0)
    assert Always(0, 2, mu).rho(s, 0) == 1.0
    assert Eventually(0, 2, mu).rho(s, 0) == 3.0
    assert Always(1, 2, mu).rho(s, 0) == 1.0          # offset window [1,2] -> min(1,2)
    assert Eventually(1, 3, mu).rho(s, 0) == 5.0


def test_and_not_or():
    s = np.array([[2.0, -1.0]])
    p, q = above(0, 0), above(1, 0)                    # rho 2 and -1
    assert And(p, q).rho(s, 0) == -1.0
    assert Or(p, q).rho(s, 0) == 2.0
    assert Not(p).rho(s, 0) == -2.0


def test_until_hand_computed():
    # p = s>0, q = s>4 on signal [1,2,5,0]; p U[0,3] q at t=0:
    s = sig([1, 2, 5, 0])
    p, q = above(0, 0), above(0, 4)
    # u=2: min(rho_q(2)=1, min(rho_p(0..2)=1,2,5)=1) = 1 ; u=0,1 negative ; u=3: min(-4, ..)<0
    assert Until(0, 3, p, q).rho(s, 0) == 1.0


# ---- identities ------------------------------------------------------------
def test_duality_not_always_equals_eventually_not():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(30, 1))
    mu = above(0, 0.1)
    for t in range(20):
        assert Not(Always(0, 4, mu)).rho(s, t) == pytest.approx(Eventually(0, 4, Not(mu)).rho(s, t))


def test_eventually_is_true_until():
    rng = np.random.default_rng(1)
    s = rng.normal(size=(30, 1))
    true = Predicate(lambda x: np.inf)
    mu = above(0, 0)
    for t in range(20):
        assert Until(0, 5, true, mu).rho(s, t) == pytest.approx(Eventually(0, 5, mu).rho(s, t))


def test_nested_horizon():
    phi = Always(0, 2, Eventually(0, 3, above(0, 0)))
    assert phi.horizon == 5
    with pytest.raises(IndexError):
        phi.rho(sig(np.ones(5)), 0)                      # needs 6 samples
    assert np.isfinite(phi.rho(sig(np.ones(6)), 0))


# ---- rho > 0  <=>  Boolean satisfaction (tested over random formulas) -------
def rand_formula(rng, depth, dim=2):
    if depth == 0 or rng.random() < 0.25:
        i, thr = int(rng.integers(dim)), float(rng.normal(0, 0.5))
        return below(i, thr) if rng.random() < 0.5 else above(i, thr)
    kind = rng.choice(["not", "and", "or", "G", "F", "U"])
    sub = lambda: rand_formula(rng, depth - 1, dim)
    a = int(rng.integers(0, 2)); b = a + int(rng.integers(0, 3))
    if kind == "not": return Not(sub())
    if kind == "and": return And(sub(), sub())
    if kind == "or": return Or(sub(), sub())
    if kind == "G": return Always(a, b, sub())
    if kind == "F": return Eventually(a, b, sub())
    return Until(a, b, sub(), sub())


def test_sign_of_robustness_matches_boolean_semantics():
    rng = np.random.default_rng(2)
    n_checked = 0
    for _ in range(300):
        phi = rand_formula(rng, 3)
        s = rng.normal(size=(40, 2))
        for t in range(0, 40 - phi.horizon, 7):
            r = phi.rho(s, t)
            assert r != 0.0                              # continuous data -> never exactly 0
            assert (r > 0) == phi.sat(s, t)
            n_checked += 1
    assert n_checked > 500


# ---- edge cases --------------------------------------------------------------
def test_zero_robustness_is_not_satisfied():
    mu = above(0, 1.0)
    assert mu.rho(sig([1.0]), 0) == 0.0 and not mu.sat(sig([1.0]), 0)


def test_bad_intervals_and_empty_and():
    with pytest.raises(ValueError):
        Always(3, 1, above(0, 0))
    with pytest.raises(ValueError):
        Always(-1, 1, above(0, 0))
    with pytest.raises(TypeError):
        Always(0.5, 1, above(0, 0))
    with pytest.raises(ValueError):
        And()


def test_trace_marks_undefined_tail_as_nan():
    tr = Always(0, 2, above(0, 0)).trace(sig([1, 2, 3, 4, 5]))
    assert np.isnan(tr[-2:]).all() and np.isfinite(tr[:3]).all()


# ---- specs from the paper ------------------------------------------------------
def test_cartpole_matches_closed_form():
    rng = np.random.default_rng(3)
    s = np.column_stack([rng.normal(0, 1.2, 60), rng.normal(size=60),
                         rng.normal(0, 6, 60), rng.normal(size=60)])   # [x, xdot, theta, thetadot]
    phi = cartpole_spec(19)
    for t in range(60 - 19):
        w = s[t:t + 20]
        closed = np.min(np.minimum(12 - np.abs(w[:, 2]), 2.4 - np.abs(w[:, 0])))
        assert phi.rho(s, t) == pytest.approx(closed)


def test_collision_spec_matches_fast_form_and_sign():
    rng = np.random.default_rng(4)
    obs = rng.uniform(-5, 5, (6, 2)); c = 0.4; H = 4
    phi = collision_avoidance_spec(obs, c, H)
    for _ in range(30):
        xy = rng.uniform(-5, 5, (H + 1, 2))
        s = np.column_stack([xy, np.zeros(H + 1), np.zeros(H + 1)])
        assert phi.rho(s, 0) == pytest.approx(window_collision_robustness(xy, obs, c))
        assert (phi.rho(s, 0) > 0) == phi.sat(s, 0)

def test_boolean_satisfaction_uses_strict_positive_robustness_at_boundaries():
    # The implementation documents the convention sat iff rho > 0 for every formula.
    s = sig([1.0])
    p = above(0, 1.0)  # rho == 0
    formulas = [p, Not(p), And(p, above(0, 2.0)), Or(p, above(0, 2.0)),
                Always(0, 0, p), Eventually(0, 0, p), Until(0, 0, p, p)]
    for phi in formulas:
        assert phi.sat(s, 0) == (phi.rho(s, 0) > 0)
