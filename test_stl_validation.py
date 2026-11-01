"""Part A tests: every hand-calculated case in stl_validation.CASES must match, including the sign
(satisfied / not satisfied), and the spec parameters must match the paper."""
import pytest
from stl_validation import CASES, run_case, param_checks


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_hand_calculated_case(case):
    r = run_case(case)
    assert r["diff"] <= 1e-9, f"{r['id']}: expected {r['expected']}, computed {r['computed']}"
    assert r["sat_computed"] == r["sat_expected"], f"{r['id']}: wrong satisfied/violated verdict"


@pytest.mark.parametrize("p", param_checks(), ids=lambda p: p["item"])
def test_spec_parameters_match_paper(p):
    assert p["ok"], f"{p['item']}: paper {p['paper']} ({p['source']}) vs code {p['code']}"


def test_boundary_is_not_satisfied_in_every_spec():
    boundary = [c for c in CASES if c["id"] in ("col_boundary", "cp_boundary")]
    for c in boundary:
        r = run_case(c)
        assert r["computed"] == pytest.approx(0.0, abs=1e-12) and r["sat_computed"] is False