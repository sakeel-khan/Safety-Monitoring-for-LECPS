"""
stl.py -- Signal Temporal Logic (STL) on discrete-time signals.
Follows Section 4.1 of Lin et al. (2025), arXiv:2504.13478.

Syntax:  phi := mu | not phi | phi1 and phi2 | G[a,b] phi | F[a,b] phi | phi1 U[a,b] phi2
Robustness (quantitative):
    rho(mu)            = f(s_t)
    rho(not p)         = -rho(p)
    rho(p and q)       = min(rho(p), rho(q))
    rho(G[a,b] p)(t)   = min_{t' in [t+a, t+b]} rho(p)(t')
    rho(F[a,b] p)(t)   = max_{t' in [t+a, t+b]} rho(p)(t')
    rho(p U[a,b] q)(t) = max_{t' in [t+a,t+b]} min( rho(q)(t'), min_{t'' in [t,t']} rho(p)(t'') )
Boolean (qualitative) semantics are implemented SEPARATELY (`sat`) directly from the
satisfaction table, so that  (s,t) |= phi  <=>  rho(s,t) > 0  can be *tested*, not assumed.

Conventions (documented because the paper leaves them open):
  * Discrete time; a window [t+a, t+b] means the integer steps t+a, ..., t+b.
  * A formula is only defined at t if the whole look-ahead window fits in the signal:
    t + horizon <= len(signal)-1.  Otherwise IndexError (no silent truncation).
  * Satisfaction is strict: rho == 0 means NOT satisfied (matches 'mu > 0' in the paper).
"""
from __future__ import annotations
from typing import Callable
import numpy as np


class Formula:
    horizon: int = 0  # how many steps into the future the formula looks

    def rho(self, s, t: int) -> float:
        raise NotImplementedError

    def sat(self, s, t: int) -> bool:
        raise NotImplementedError

    def _check(self, s, t: int):
        if t < 0 or t + self.horizon >= len(s):
            raise IndexError(
                f"t={t} with horizon={self.horizon} needs signal length > {t + self.horizon}, got {len(s)}")

    def trace(self, s) -> np.ndarray:
        """rho(s, t) for every t where it is defined; NaN elsewhere."""
        out = np.full(len(s), np.nan)
        for t in range(len(s) - self.horizon):
            out[t] = self.rho(s, t)
        return out

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


class Predicate(Formula):
    """Atomic predicate mu: f(state) > 0 ; robustness is f(state) itself."""

    def __init__(self, f: Callable, name: str = "mu"):
        self.f, self.name, self.horizon = f, name, 0

    def rho(self, s, t):
        self._check(s, t)
        return float(self.f(s[t]))

    def sat(self, s, t):
        # Use the same strict quantitative convention for every formula:
        # satisfaction iff robustness is strictly positive.
        return self.rho(s, t) > 0


class Not(Formula):
    def __init__(self, p):
        self.p, self.horizon = p, p.horizon

    def rho(self, s, t):
        self._check(s, t)
        return -self.p.rho(s, t)

    def sat(self, s, t):
        return self.rho(s, t) > 0


class And(Formula):
    def __init__(self, *ps):
        if len(ps) < 1:
            raise ValueError("And needs at least one operand")
        self.ps, self.horizon = ps, max(p.horizon for p in ps)

    def rho(self, s, t):
        self._check(s, t)
        return min(p.rho(s, t) for p in self.ps)

    def sat(self, s, t):
        return self.rho(s, t) > 0


class Or(Formula):
    """Derived operator: p or q = not(not p and not q); rho = max."""

    def __init__(self, *ps):
        if len(ps) < 1:
            raise ValueError("Or needs at least one operand")
        self.ps, self.horizon = ps, max(p.horizon for p in ps)

    def rho(self, s, t):
        self._check(s, t)
        return max(p.rho(s, t) for p in self.ps)

    def sat(self, s, t):
        return self.rho(s, t) > 0


def _check_interval(a, b):
    if not (isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer))):
        raise TypeError("interval bounds must be integers (discrete time)")
    if a < 0 or b < a:
        raise ValueError("need 0 <= a <= b")


class Always(Formula):
    """G[a,b] p  (the paper's box operator)."""

    def __init__(self, a, b, p):
        _check_interval(a, b)
        self.a, self.b, self.p, self.horizon = a, b, p, b + p.horizon

    def rho(self, s, t):
        self._check(s, t)
        return min(self.p.rho(s, u) for u in range(t + self.a, t + self.b + 1))

    def sat(self, s, t):
        return self.rho(s, t) > 0


class Eventually(Formula):
    """F[a,b] p  (the paper's diamond operator)."""

    def __init__(self, a, b, p):
        _check_interval(a, b)
        self.a, self.b, self.p, self.horizon = a, b, p, b + p.horizon

    def rho(self, s, t):
        self._check(s, t)
        return max(self.p.rho(s, u) for u in range(t + self.a, t + self.b + 1))

    def sat(self, s, t):
        return self.rho(s, t) > 0


class Until(Formula):
    """p U[a,b] q.  (Paper's last line writes (x,t'') -- a typo for (s,t'').)"""

    def __init__(self, a, b, p, q):
        _check_interval(a, b)
        self.a, self.b, self.p, self.q = a, b, p, q
        self.horizon = b + max(p.horizon, q.horizon)

    def rho(self, s, t):
        self._check(s, t)
        best = -np.inf
        for u in range(t + self.a, t + self.b + 1):
            guard = min(self.p.rho(s, v) for v in range(t, u + 1))
            best = max(best, min(self.q.rho(s, u), guard))
        return best

    def sat(self, s, t):
        return self.rho(s, t) > 0


# ----------------------------------------------------------------------------
# Convenience predicates
# ----------------------------------------------------------------------------
def below(index: int, thr: float) -> Predicate:
    """s[index] < thr   (rho = thr - s[index])"""
    return Predicate(lambda s: thr - s[index], f"s[{index}]<{thr}")


def above(index: int, thr: float) -> Predicate:
    """s[index] > thr   (rho = s[index] - thr)"""
    return Predicate(lambda s: s[index] - thr, f"s[{index}]>{thr}")


def abs_below(index: int, thr: float) -> Predicate:
    """|s[index]| < thr  (rho = thr - |s[index]|)"""
    return Predicate(lambda s: thr - abs(s[index]), f"|s[{index}]|<{thr}")


# ----------------------------------------------------------------------------
# Specs used in the paper
# ----------------------------------------------------------------------------
def cartpole_spec(horizon: int = 19, x_idx: int = 0, theta_idx: int = 2,
                  max_angle: float = 12.0, max_pos: float = 2.4) -> Formula:
    """Section 2:  phi = G[0,19] ( |theta| < 12 ) and ( |x| < 2.4 ).
    State order [x, xdot, theta, thetadot] => theta is index 2 (0-based). The paper's
    formula prints index 3, which looks like an indexing slip (see README notes)."""
    return Always(0, horizon, abs_below(theta_idx, max_angle) & abs_below(x_idx, max_pos))


def collision_avoidance_spec(obstacles, c: float, H: int, pos=(0, 1)) -> Formula:
    """Section 7.3:  phi = G[0,H] AND_i  d((x,y),(xo_i,yo_i)) > c."""
    obstacles = np.atleast_2d(np.asarray(obstacles, float))
    pos = list(pos)
    preds = [Predicate(lambda s, o=o: float(np.linalg.norm(np.asarray(s)[pos] - o)) - c, f"d>{c}")
             for o in obstacles]
    return Always(0, H, And(*preds))


def window_collision_robustness(xy, obstacles, c: float) -> float:
    """Fast closed form of the collision spec on a window of positions xy (n,2):
        rho = min_{t'} min_i d(xy[t'], obstacle_i) - c
    NOTE: Algorithm 2 feeds the H predicted states s_hat_{t+1..t+H}; that is H samples, i.e.
    effectively G[0,H-1] on the window (a G[0,H] window would need H+1 samples)."""
    xy = np.atleast_2d(np.asarray(xy, float))
    obstacles = np.atleast_2d(np.asarray(obstacles, float))
    d = np.linalg.norm(xy[:, None, :] - obstacles[None, :, :], axis=-1)
    return float(d.min() - c)