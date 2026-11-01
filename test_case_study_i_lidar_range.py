import numpy as np
import pytest
from case_study_common import make_base, compare_methods, lidar_range_series, lidar_calibration_scores


def test_lidar_range_validation():
    base = make_base(11, 120, H=5)
    with pytest.raises(ValueError):
        lidar_range_series(base, 0)
    with pytest.raises(ValueError):
        lidar_range_series(base, 3, sigma=-0.1)


def test_lower_range_never_increases_visible_obstacle_count():
    base = make_base(12, 200, H=5)
    _, rho5, info5 = lidar_range_series(base, 5.0, 0.05)
    _, rho2, info2 = lidar_range_series(base, 2.0, 0.05)
    assert np.all(info2["visible_obstacle_count"] <= info5["visible_obstacle_count"])
    assert np.array_equal(rho2, rho5)  # true safety is independent of sensor range
    assert np.all(np.isfinite(rho2))


def test_lidar_series_reproducible_and_range_affects_prediction_estimate():
    base = make_base(13, 180, H=5)
    a, rho_a, _ = lidar_range_series(base, 3.0, 0.05)
    b, rho_b, _ = lidar_range_series(base, 3.0, 0.05)
    assert np.array_equal(a, b)
    assert np.array_equal(rho_a, rho_b)
    wide, _, _ = lidar_range_series(base, 5.0, 0.05)
    assert not np.allclose(a, wide)


def test_lidar_calibration_is_finite_and_reproducible():
    a = lidar_calibration_scores(17, n=120, H=5)
    b = lidar_calibration_scores(17, n=120, H=5)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()


def test_h0_original_and_lag_aware_consistency_on_lidar_case():
    base = make_base(18, 160, H=5)
    rh, rho, _ = lidar_range_series(base, 3.0, 0.05)
    cal = lidar_calibration_scores(18, n=120, H=5)
    _, _, checks, _ = compare_methods(rh, rho, cal, lags=(0, 3), t0=10)
    h0 = checks.loc[checks.H == 0]
    assert h0.H0_thresholds_match.all()
    assert h0.H0_delta_match.all()