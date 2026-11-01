"""ACP Coverage tests: ACP running coverage must stay inside the Lemma 1 envelope for ordinary AND shifted data,
for both threshold orderings and both initializations; split CP must NOT (so the test has power)."""
import numpy as np
import pytest
from acp_coverage import run_experiment, envelope_stats, PANELS

ALL = [n for v in PANELS.values() for n in v]


@pytest.mark.parametrize("literal", [False, True])
@pytest.mark.parametrize("init", ["cal", "none"])
def test_acp_inside_envelope_all_scenarios(literal, init):
    res = run_experiment(ALL, seeds=3, T=600, lag=5, t0=15, init=init, literal=literal)
    for name, r in res.items():
        n_pts, n_seeds, _, _ = envelope_stats(r["cov"])
        assert n_pts == 0, f"{name}: {n_pts} points outside envelope ({literal=}, {init=})"


def test_acp_final_coverage_near_target():
    res = run_experiment(ALL, seeds=3, T=1200, lag=5, t0=15, init="cal")
    for name, r in res.items():
        assert abs(r["cov"][:, -1].mean() - 0.9) < 0.03, name


def test_cp_leaves_envelope_under_shift():
    res = run_experiment(["adversarial_monotone", "abrupt_shift"], seeds=3, T=1200, cp=True)
    for name, r in res.items():
        n_pts, _, _, _ = envelope_stats(r["cov_cp"])
        assert n_pts > 0, f"{name}: CP unexpectedly stayed inside the envelope"


def test_curve_shape_and_lag_handling():
    res = run_experiment(["iid_id"], seeds=2, T=300, lag=5, t0=15)
    assert res["iid_id"]["cov"].shape == (2, 300 - 15)          # monitoring starts at max(t0, lag)
    assert np.all((res["iid_id"]["cov"] >= 0) & (res["iid_id"]["cov"] <= 1))