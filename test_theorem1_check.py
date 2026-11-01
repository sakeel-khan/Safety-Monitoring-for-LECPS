"""Part D tests: STL is the source of robustness, premise detection is correct, no premise run falls below the bound
(ACP), the algebraic implication never fails, and the check has power (split CP under hidden violations must fail)."""
from typing import Literal

import numpy as np
import pytest
from acp import theorem1_lower_bound
from stl import collision_avoidance_spec, window_collision_robustness
from theorem1_check import (analyse_run, run_experiment, summarise, stress_scenes, trajectory_world,
                            DELTA, GAMMA, H)


def test_bound_by_hand():
    # T=1000: p1 = (0.1+0.005)/(1000*0.005) = 0.021  ->  0.9 - 0.021
    assert theorem1_lower_bound(1000, DELTA, GAMMA) == pytest.approx(0.879)


def test_stress_scene_robustness_comes_from_stl_and_matches_the_design():
    rng = np.random.default_rng(0)
    rho_hat, rho, _ = stress_scenes(rng, q=0.1, T=200)
    assert np.all(np.isfinite(rho_hat)) and np.all(np.isfinite(rho))
    assert rho_hat.min() > 2.0                                   # predictor 'looks safe' (designed ~3)
    assert (rho < 0).any() and (rho_hat - rho).max() > 3.8       # hidden violations exist


def test_fast_window_robustness_equals_stl_formula_tree():
    rng = np.random.default_rng(1)
    obs = rng.uniform(-5, 5, (4, 2)); c = 0.5
    spec = collision_avoidance_spec(obs, c, H - 1)               # H predicted states == G[0, H-1]
    for _ in range(20):
        xy = rng.uniform(-5, 5, (H, 2))
        s = np.column_stack([xy, np.ones(H), np.zeros(H)])
        assert window_collision_robustness(xy, obs, c) == pytest.approx(spec.rho(s, 0))


def test_premise_detection_on_handmade_arrays():
    cal = np.zeros(50)                                           # ACP/CP thresholds ~0 from zero calibration errors
    T = 100
    rho_true = np.full(T, 2.0)
    safe = analyse_run(np.full(T, 2.0), rho_true, cal, "cp", lag=0, t0=15)
    assert safe["premise"] and safe["sat"] == 1.0 and safe["alarm_rate"] == 0.0
    dip = np.full(T, 2.0); dip[40] = -1.0                        # one alarm step breaks the premise
    unsafe = analyse_run(dip, rho_true, cal, "cp", lag=0, t0=15)
    assert not unsafe["premise"] and unsafe["alarm_rate"] > 0
    eq = np.full(T, 0.0)                                         # rho_hat == C (=0) is an alarm (strict >)
    assert not analyse_run(eq, rho_true, cal, "cp", lag=0, t0=15)["premise"]


@pytest.mark.parametrize("kind", ["trajectories", "stress"])
@pytest.mark.parametrize("lag", [0, 5])
def test_acp_premise_runs_never_below_bound_and_implication_holds(kind: Literal['trajectories'] | Literal['stress'], lag: Literal[0] | Literal[5]):
    runs = run_experiment(kind, seeds=12, lag=lag, methods=("acp",))
    rows = summarise(runs)
    assert sum(r["n_premise_below_bound"] for r in rows) == 0
    assert sum(r["impl_viol"] for r in rows) == 0
    assert sum(r["n_premise"] for r in rows) > 0                 # test is not vacuous: premise runs exist


def test_acp_raises_alarms_when_hidden_violations_are_frequent():
    rows = summarise(run_experiment("stress", seeds=10, methods=("acp",), levels=[0.2]))
    assert rows[0]["n_premise"] == 0 and rows[0]["alarm_rate"] > 0.5


def test_check_has_power_split_cp_breaks_the_bound():
    rows = summarise(run_experiment("stress", seeds=10, methods=("cp",), levels=[0.2]))
    assert rows[0]["n_premise"] == 10                            # CP never alarms ...
    assert rows[0]["n_premise_below_bound"] == 10                # ... and satisfaction is below the bound


def test_premise_fraction_decreases_with_hidden_violation_rate_for_acp():
    rows = summarise(run_experiment("stress", seeds=15, methods=("acp",), levels=[0.02, 0.08, 0.2]))
    fr = [r["frac_premise"] for r in rows]
    assert fr[0] > fr[1] >= fr[2] and fr[2] == 0.0