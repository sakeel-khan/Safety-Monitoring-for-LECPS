"""Part C tests: bound formulas, tightening with T, observed coverage inside bounds for all scenarios,
and a power check (a deliberately too-tight 'bound' must be violated)."""
import numpy as np
import pytest
from acp import lemma1_bounds
from acp_coverage import PANELS, DELTA, GAMMA
from lemma1_table import build_table, width_formula, to_markdown

ALL = [n for v in PANELS.values() for n in v]
TS = [20, 100, 500]


def test_p1_p2_formulas_by_hand():
    # T=1000, delta=0.1, gamma=0.005 -> p1=0.105/5=0.021, p2=0.905/5=0.181
    lo, hi = lemma1_bounds(1000, 0.1, 0.005)
    assert lo == pytest.approx(0.9 - 0.021) and hi == pytest.approx(0.9 + 0.181)


def test_width_formula_and_independence_of_delta():
    for T in (10, 100, 5000):
        for d in (0.05, 0.1, 0.3):
            lo, hi = lemma1_bounds(T, d, GAMMA)
            assert hi - lo == pytest.approx((1 + 2 * GAMMA) / (T * GAMMA))
    assert width_formula(1000) == pytest.approx(0.202)


def test_bounds_tighten_monotonically_and_converge_to_target():
    Ts = np.array([10, 100, 1000, 10000, 10 ** 7], float)
    lo, hi = lemma1_bounds(Ts, DELTA, GAMMA)
    assert np.all(np.diff(lo) > 0) and np.all(np.diff(hi) < 0)
    assert lo[-1] == pytest.approx(0.9, abs=1e-4) and hi[-1] == pytest.approx(0.9, abs=1e-4)


def test_small_T_bounds_are_vacuous():
    lo, hi = lemma1_bounds(10, DELTA, GAMMA)
    assert lo < 0 and hi > 1


def test_observed_coverage_within_bounds_all_scenarios():
    rows = build_table(ALL, TS, seeds=3)
    assert len(rows) == len(ALL) * len(TS)
    bad = [(r["scenario"], r["T"]) for r in rows if not r["within"]]
    assert not bad, f"outside bounds: {bad}"


@pytest.mark.parametrize("literal,init", [(False, "none"), (True, "none"), (True, "cal")])
def test_within_bounds_for_other_configurations(literal, init):
    rows = build_table(["abrupt_shift", "adversarial_monotone"], [100, 500], seeds=3, init=init, literal=literal)
    assert all(r["within"] for r in rows)


def test_check_has_power_too_tight_bounds_are_violated():
    # adversarial_monotone sits almost exactly on the lower bound, so halving p1 must be violated
    rows = build_table(["adversarial_monotone"], [500, 1000], seeds=3, p_scale=0.5)
    assert any(not r["within"] for r in rows)


def test_markdown_table_has_required_columns():
    md = to_markdown(build_table(["iid_id"], [20], seeds=2))
    for col in ("T", "observed coverage", "lower bound", "upper bound", "within bounds?"):
        assert col in md