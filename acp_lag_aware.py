"""
acp_lag_aware.py -- Lag-aware Adaptive Conformal Prediction (research extension).

Builds ON TOP of acp.py (which is left untouched): it reuses `conformal_threshold`
(including its delta_t < 0 / delta_t > 1 conventions) and the *same* update expression
    delta_{t+1} = delta_t + gamma * (delta - e)
and changes only WHICH threshold decides the error indicator e of a delayed outcome.

Timeline (prediction issued at t, outcome rho[t] arrives at time t + H; H = "lag")
-------------------------------------------------------------------------------
At every time step t (in this order; this reproduces the ordering of acp.run_lagged):
  (0) [include_current=True only] the score of the ARRIVING outcome is inserted first.
  (1) ISSUE:   C_issue[t] = conformal_threshold(scores, delta_t)       <- threshold in force at issuance
               alarm[t]   = rho_hat[t] - C_issue[t] <= 0                <- the deployed decision
               C_issue[t] is STORED together with the prediction.
  (2) ARRIVE:  the outcome of the prediction issued at a = t - H is now known.
               score S_a = rho_hat[a] - rho[a].
               EVALUATION threshold  = the STORED C_issue[a]            (lag-aware)
               e_a = 0 if S_a <= C_issue[a] else 1                      (deployed-region miss indicator)
               CONTROLLER UPDATE     = delta_{t+1} = delta_t + gamma (delta - e_a)
  (3) S_a is inserted into the score set (unless already inserted in (0)).

Two thresholds are therefore distinguished for every delayed outcome:
  * evaluation threshold  C_issue[a]    -- the one that produced the alarm for that prediction
  * arrival threshold     C_issue[t]    -- the one the ORIGINAL implementation scores it with
They coincide iff H = 0 (or if the controller did not move in between).

`update_error` selects what drives the controller:
  "issuance" (lag-aware): e_a from the stored issuance threshold.
  "arrival"  (diagnostic): e from the arrival-time threshold. This is exactly the ORIGINAL
             acp.run_lagged dynamics (verified in the tests); the deployed coverage is still
             evaluated with the stored issuance threshold, so it can be used to separate
             "same controller, different measurement" from "different controller".

Warm-up: predictions issued before `start = max(t0, H)` are never deployed (no threshold, no alarm).
Their scores are inserted when they arrive. In "issuance" mode they trigger NO delta update (there
is no deployed region to evaluate); in "arrival" mode they do (as in acp.run_lagged).

NOTHING in this file asserts that Lemma 1 still holds. See README.md / REPORT.md for the analysis.
"""
from __future__ import annotations

import bisect
import math
from typing import Iterable

import numpy as np

from acp import conformal_threshold, predicts_violation  # reused, not re-implemented


# --------------------------------------------------------------------------- controller
class _Controller:
    """Score set + delta_t, with the *same* update expression as acp.AdaptiveConformal."""

    def __init__(self, delta: float, gamma: float, init_scores: Iterable[float] = ()):
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0,1)")
        if not math.isfinite(float(gamma)) or not gamma > 0:
            raise ValueError("gamma must be finite and > 0")
        init_scores = [float(x) for x in init_scores]
        if not all(math.isfinite(x) for x in init_scores):
            raise ValueError("init_scores must contain only finite values")
        self.delta, self.gamma = float(delta), float(gamma)
        self.delta_t = float(delta)                      # delta_1 = delta
        self.scores = sorted(init_scores)

    def threshold(self) -> float:
        return conformal_threshold(self.scores, self.delta_t)

    def insert(self, s: float):
        bisect.insort(self.scores, float(s))

    def update(self, e: int):
        self.delta_t = self.delta_t + self.gamma * (self.delta - e)   # identical to acp.py


# --------------------------------------------------------------------------- main routine
def run_lag_aware(rho_hat, rho, delta: float = 0.1, gamma: float = 0.005, lag: int = 0, t0: int = 0,
                  init_scores: Iterable[float] = (), include_current: bool = False,
                  update_error: str = "issuance"):
    """Run lag-aware ACP on a pre-recorded (rho_hat, rho) series (both indexed by ISSUE time).

    Returns a dict (arrays of length T, NaN where undefined):
      C_issue      threshold stored at issuance (NaN before `start`)
      delta_issue  delta_t in force at issuance
      alarm        bool, rho_hat - C_issue <= 0 (False where undeployed)
      deployed     bool, issued at t >= start
      evaluated    bool, deployed AND outcome arrived within the series (t + lag <= T-1)
      e_deployed   0/1 miss indicator of the stored deployed region (NaN if not evaluated)
      e_update     indicator actually fed to the controller for prediction t (NaN if no update)
      e_arrival    counterfactual indicator using the ARRIVAL-time threshold C_issue[t+lag]
      score        S_t = rho_hat[t] - rho[t]
      arrival_log  list of (arrival_time, issue_index, score, updated_controller)
      delta_final  delta after the last update
      start        first deployed issue index
    """
    rho_hat, rho = _validate(rho_hat, rho, lag, t0)
    if update_error not in ("issuance", "arrival"):
        raise ValueError("update_error must be 'issuance' or 'arrival'")
    T = len(rho)
    start = max(t0, lag)
    S = rho_hat - rho

    ctl = _Controller(delta, gamma, init_scores)
    C_issue = np.full(T, np.nan)
    d_issue = np.full(T, np.nan)
    alarm = np.zeros(T, dtype=bool)
    e_update = np.full(T, np.nan)
    stored = {}                                   # issue index -> (rho_hat, C) stored with the prediction
    log = []

    for t in range(T):
        a = t - lag                               # issue index of the outcome that arrives now
        arrives = a >= 0
        inserted = False
        if arrives and include_current:           # literal Algorithm-2 order
            ctl.insert(S[a]); inserted = True

        # (1) ISSUE
        if t >= start:
            d_issue[t] = ctl.delta_t
            C = ctl.threshold()
            C_issue[t] = C
            alarm[t] = bool(predicts_violation(rho_hat[t], C))
            stored[t] = (rho_hat[t], C)           # threshold stored WITH the prediction

        # (2) ARRIVE
        if arrives:
            updated = False
            if t >= start:                        # controller is active at time t
                if update_error == "issuance":
                    if a >= start:                # a was deployed -> it has a stored threshold
                        _, C_eval = stored[a]
                        e = 0 if S[a] <= C_eval else 1
                        ctl.update(e); e_update[a] = e; updated = True
                else:                             # "arrival": original timing
                    C_arr = C_issue[t]
                    e = 0 if S[a] <= C_arr else 1
                    ctl.update(e); e_update[a] = e; updated = True
            stored.pop(a, None)
            log.append((t, a, float(S[a]), updated))
            if not inserted:
                ctl.insert(S[a])

    idx = np.arange(T)
    deployed = idx >= start
    evaluated = deployed & (idx + lag <= T - 1)
    view = deployed_view(rho_hat, rho, C_issue, lag, start)
    return dict(C_issue=C_issue, delta_issue=d_issue, alarm=alarm, deployed=deployed,
                evaluated=evaluated, e_deployed=view["e_deployed"], e_update=e_update,
                e_arrival=view["e_arrival"], score=S, arrival_log=log,
                delta_final=ctl.delta_t, start=start, lag=lag)


# --------------------------------------------------------------------------- common issuance-indexed view
def deployed_view(rho_hat, rho, C_issue, lag: int, start: int):
    """Issuance-indexed view of ANY run, defined only by its per-prediction issuance thresholds.

    For the ORIGINAL implementation pass out['C'] from acp.run_lagged (its C[t] is the threshold
    in force when prediction t is issued, i.e. what drives the alarm at t).

      e_deployed[t] = 1[ S_t >  C_issue[t]       ]  deployed-region miss  (outcome of t inside [rho_hat-C, inf)?)
      e_arrival[t]  = 1[ S_t >  C_issue[t+lag]   ]  miss under the ARRIVAL-time threshold, i.e. the
                                                    original scoring convention applied to prediction t
    Both are NaN unless t >= start and t + lag <= T-1."""
    rho_hat, rho = np.asarray(rho_hat, float), np.asarray(rho, float)
    C = np.asarray(C_issue, float)
    T = len(rho)
    S = rho_hat - rho
    idx = np.arange(T)
    ok = (idx >= start) & (idx + lag <= T - 1)
    e_dep = np.full(T, np.nan)
    e_arr = np.full(T, np.nan)
    ii = idx[ok]
    e_dep[ii] = (S[ii] > C[ii]).astype(float)
    e_arr[ii] = (S[ii] > C[ii + lag]).astype(float)
    alarm = np.zeros(T, dtype=bool)
    dep = idx >= start
    alarm[dep] = predicts_violation(rho_hat[dep], C[dep])
    return dict(e_deployed=e_dep, e_arrival=e_arr, alarm=alarm, score=S, evaluated=ok, deployed=dep)


# --------------------------------------------------------------------------- theoretical note
# No finite coverage envelope depending only on lag H is asserted here.
# The update rule is intentionally not clipped: delta_t may leave [0, 1].
# Under issuance-time feedback, the error used to update the controller is
# evaluated against a threshold issued H steps earlier, so the original
# Lemma-1 telescoping argument cannot be transferred without a separate proof.


# --------------------------------------------------------------------------- validation
def _validate(rho_hat, rho, lag, t0):
    rho_hat, rho = np.asarray(rho_hat, float), np.asarray(rho, float)
    if rho_hat.ndim != 1 or rho.ndim != 1 or rho_hat.shape != rho.shape:
        raise ValueError("rho_hat and rho must be 1-D arrays of equal length")
    if isinstance(lag, bool) or isinstance(t0, bool) or \
            not isinstance(lag, (int, np.integer)) or not isinstance(t0, (int, np.integer)):
        raise TypeError("lag and t0 must be integers")
    if lag < 0 or t0 < 0:
        raise ValueError("lag and t0 must be >= 0")
    if not (np.all(np.isfinite(rho_hat)) and np.all(np.isfinite(rho))):
        raise ValueError("rho_hat and rho must contain only finite values")
    return rho_hat, rho
