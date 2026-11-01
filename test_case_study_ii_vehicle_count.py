import numpy as np
import pytest
from case_study_common import compare_methods, make_vehicle_base, vehicle_count_series, vehicle_calibration_scores


def test_vehicle_count_validation():
    base = make_vehicle_base(21, 120, H=5, max_vehicles=8)
    with pytest.raises(ValueError):
        vehicle_count_series(base, 1)
    with pytest.raises(ValueError):
        vehicle_count_series(base, 9)
    with pytest.raises(ValueError):
        vehicle_count_series(base, 4, sigma=-0.1)


def test_vehicle_series_reproducible_and_finite():
    base = make_vehicle_base(22, 180, H=5, max_vehicles=8)
    a, rho_a = vehicle_count_series(base, 6, 0.05)
    b, rho_b = vehicle_count_series(base, 6, 0.05)
    assert np.array_equal(a, b)
    assert np.array_equal(rho_a, rho_b)
    assert np.isfinite(a).all() and np.isfinite(rho_a).all()


def test_vehicle_count_changes_dynamic_robustness():
    base = make_vehicle_base(23, 180, H=5, max_vehicles=8)
    _, rho4 = vehicle_count_series(base, 4, 0.05)
    _, rho8 = vehicle_count_series(base, 8, 0.05)
    assert not np.allclose(rho4, rho8)


def test_vehicle_calibration_finite_and_reproducible():
    a = vehicle_calibration_scores(24, n=120, H=5)
    b = vehicle_calibration_scores(24, n=120, H=5)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()


def test_h0_original_and_lag_aware_consistency_on_vehicle_case():
    base = make_vehicle_base(25, 160, H=5, max_vehicles=8)
    rh, rho = vehicle_count_series(base, 6, 0.05)
    cal = vehicle_calibration_scores(25, n=120, H=5)
    _, _, checks, _ = compare_methods(rh, rho, cal, lags=(0, 3), t0=10)
    h0 = checks.loc[checks.H == 0]
    assert h0.H0_thresholds_match.all()
    assert h0.H0_delta_match.all()