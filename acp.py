"""
acp.py -- Adaptive Conformal Prediction for robustness-value prediction regions.
Follows Sections 4.2 and 6.1 / Algorithm 2 (lines 8-16, 20-21) of Lin et al. (2025),
which in turn uses Gibbs & Candes (2021) and Dixit et al. (2023).

Quantities (delta = target MISCOVERAGE level, e.g. 0.1 -> 90% coverage):
  nonconformity score   R_t   = rho_hat_{t-H} - rho_{t-H}       (signed; overestimating safety is the danger)
  threshold             C_t   = ceil(n (1 - delta_t))-th smallest score in the NCS set (n = set size)
  error indicator       e_t   = 0 if R_t <= C_t else 1
  significance update   delta_{t+1} = delta_t + gamma (delta - e_t)
  prediction region     rho_t in [rho_hat_t - C_t, +inf)  with prob ~ 1-delta (on average over time)
  alarm                 rho_hat_t - C_t <= 0   (paper: < 0; we treat equality as an alarm, conservative)

Invariance that makes Lemma 1 a theorem about the code (see verify_lemma1.py):
    delta_t > 1  => region empty      => e_t = 1  (delta_t decreases)
    delta_t < 0  => region = R (all)  => e_t = 0  (delta_t increases)
 hence delta_t stays in [-gamma, 1+gamma] forever, and telescoping the update gives
    gamma * sum_t (delta - e_t) = delta_{T+1} - delta_1  in  [-(delta+gamma), (1-delta+gamma)]   (delta_1 = delta)
 i.e.  1-delta-p1 <= (1/T) sum_t (1-e_t) <= 1-delta+p2.
"""
from __future__ import annotations
import bisect
import math
from typing import Iterable
import numpy as np


# ----------------------------------------------------------------------------
# Threshold from a sorted score list
# ----------------------------------------------------------------------------
def conformal_threshold(sorted_scores, delta_t: float) -> float:
    """C = ceil(n (1-delta_t))-th smallest score, with the edge conventions that keep
    delta_t in [-gamma, 1+gamma]:
        n == 0 or delta_t < 0 -> +inf   (everything covered)
        delta_t > 1           -> -inf   (empty region, every score is an error)
        otherwise k = ceil(n(1-delta_t)) clipped to [1, n]  (k=0 only when delta_t == 1)"""
    n = len(sorted_scores)
    if n == 0 or delta_t < 0:
        return math.inf
    if delta_t > 1:
        return -math.inf
    k = math.ceil(round(n * (1.0 - delta_t), 9))     # round() guards 9.000000000000002 -> 10
    k = min(max(k, 1), n)
    return float(sorted_scores[k - 1])


# ----------------------------------------------------------------------------
# Adaptive conformal prediction
# ----------------------------------------------------------------------------
class AdaptiveConformal:
    """
    Parameters
    ----------
    delta, gamma : target miscoverage and learning rate (paper: 0.1, 0.005)
    init_scores  : optional warm-start NCS set (e.g. from an offline calibration set)
    include_current : False (default, Gibbs-Candes order): C_t is computed BEFORE R_t joins the set.
                      True  : literal Algorithm 2 order (line 11 appends R_t, then line 14 takes the quantile).
    """

    def __init__(self, delta: float = 0.1, gamma: float = 0.005,
                 init_scores: Iterable[float] = (), include_current: bool = False):
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0,1)")
        if not gamma > 0:
            raise ValueError("gamma must be > 0")
        self.delta, self.gamma = float(delta), float(gamma)
        self.delta_t = float(delta)                     # delta_1 = delta
        self.include_current = include_current
        self._scores = sorted(float(x) for x in init_scores)

    @property
    def n(self) -> int:
        return len(self._scores)

    def threshold(self) -> float:
        return conformal_threshold(self._scores, self.delta_t)

    def accumulate(self, score: float):
        """Add a score WITHOUT adapting delta_t (Algorithm 2 gathers scores from t>h+H but only
        runs lines 13-16 after t0)."""
        self._check(score)
        bisect.insort(self._scores, float(score))

    def step(self, score: float):
        """One ACP iteration for a new nonconformity score. Returns (C_t, e_t)."""
        self._check(score)
        if self.include_current:
            bisect.insort(self._scores, float(score))
            C = self.threshold()
        else:
            C = self.threshold()
            bisect.insort(self._scores, float(score))
        e = 0 if score <= C else 1
        self.delta_t = self.delta_t + self.gamma * (self.delta - e)
        return C, e

    @staticmethod
    def _check(score):
        if not np.isfinite(score):
            raise ValueError(f"score must be finite, got {score}")


class SplitConformal:
    """Non-adaptive inductive CP baseline: one fixed quantile from an offline calibration set,
    k = ceil((n+1)(1-delta)).  Same interface as AdaptiveConformal so it can be dropped into run_lagged."""

    def __init__(self, delta: float, cal_scores: Iterable[float]):
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0,1)")
        s = np.asarray(list(cal_scores), dtype=float)
        if s.ndim != 1 or len(s) == 0:
            raise ValueError("need a non-empty one-dimensional calibration set")
        if not np.all(np.isfinite(s)):
            raise ValueError("calibration scores must all be finite")
        s = np.sort(s)
        k = math.ceil(round((len(s) + 1) * (1 - delta), 9))
        self.C = math.inf if k > len(s) else float(s[k - 1])
        self.delta = self.delta_t = float(delta)

    def accumulate(self, score):
        pass

    def step(self, score):
        return self.C, (0 if score <= self.C else 1)


# ----------------------------------------------------------------------------
# Prediction region / alarm
# ----------------------------------------------------------------------------
def prediction_region(rho_hat: float, C: float):
    """One-sided region [rho_hat - C, +inf) for the true robustness (Lemma 1)."""
    return rho_hat - C, math.inf


def predicts_violation(rho_hat, C):
    """Alarm: lower end of the region is not strictly positive (works on arrays; NaN -> False)."""
    return (np.asarray(rho_hat) - np.asarray(C)) <= 0


# ----------------------------------------------------------------------------
# Algorithm 2, lines 8-16 on a pre-recorded (rho_hat, rho) series, with H-step lag
# ----------------------------------------------------------------------------
def run_lagged(rho_hat, rho, acp, lag: int = 0, t0: int = 0):
    """At time t the monitor has the prediction rho_hat_t but only the ground truth rho_{t-lag}.
    So the score available at t is R_t = rho_hat_{t-lag} - rho_{t-lag}.
    Monitoring (ACP steps) starts at max(t0, lag); earlier scores are only accumulated.

    Returns arrays of length T (NaN where undefined):
        C[t]       threshold used at time t (for both scoring and the alarm, as in Algorithm 2)
        e[t]       ACP error indicator of the SCORED event  R_t <= C_t
        delta_t[t] significance level in force at time t (before the update)
        start      first monitored index"""
    rho_hat, rho = np.asarray(rho_hat, float), np.asarray(rho, float)
    if rho_hat.shape != rho.shape:
        raise ValueError("rho_hat and rho must have equal length")
    if not isinstance(lag, (int, np.integer)) or not isinstance(t0, (int, np.integer)):
        raise TypeError("lag and t0 must be integers")
    if lag < 0 or t0 < 0:
        raise ValueError("lag and t0 must be >= 0")
    if not (np.all(np.isfinite(rho_hat)) and np.all(np.isfinite(rho))):
        raise ValueError("rho_hat and rho must contain only finite values")
    T = len(rho)
    start = max(t0, lag)
    C, e, dl = (np.full(T, np.nan) for _ in range(3))
    for t in range(lag, T):
        R = rho_hat[t - lag] - rho[t - lag]
        if t < start:
            acp.accumulate(R)
            continue
        dl[t] = acp.delta_t
        C[t], e[t] = acp.step(R)
    return dict(C=C, e=e, delta_t=dl, start=start)


# ----------------------------------------------------------------------------
# Bounds from the paper (Lemma 1, Theorem 1) and the original Gibbs-Candes bound (Prop. 1)
# ----------------------------------------------------------------------------
def lemma1_bounds(T, delta: float, gamma: float):
    """(lower, upper) = (1-delta-p1, 1-delta+p2), p1=(delta+gamma)/(T gamma), p2=((1-delta)+gamma)/(T gamma)."""
    T = np.asarray(T, float)
    p1 = (delta + gamma) / (T * gamma)
    p2 = ((1 - delta) + gamma) / (T * gamma)
    return 1 - delta - p1, 1 - delta + p2


def theorem1_lower_bound(T, delta: float, gamma: float):
    """1 - delta - p1  (Theorem 1)."""
    return lemma1_bounds(T, delta, gamma)[0]


def gibbs_candes_bound(T, delta: float, gamma: float, delta1: float | None = None):
    """Prop. 1:  |mean(e_t) - delta| <= (max(delta1, 1-delta1) + gamma) / (T gamma)."""
    d1 = delta if delta1 is None else delta1
    return (max(d1, 1 - d1) + gamma) / (np.asarray(T, float) * gamma)