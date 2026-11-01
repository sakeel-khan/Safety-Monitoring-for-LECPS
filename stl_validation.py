"""
stl_validation.py -- Part A: verify STL robustness against HAND-CALCULATED expected values.

Every `expected` number below was worked out by hand (derivation in the table), NOT produced by stl.py.
Running this file prints and saves the "expected vs computed" table and exits non-zero if anything fails.

    python stl_validation.py          -> results/stl_expected_vs_computed.md and .csv
    python -m pytest tests/test_stl_validation.py
"""
import csv, inspect, os, sys
import numpy as np
from stl import (above, below, abs_below, Not, And, Or, Always, Eventually, Until,
                 cartpole_spec, collision_avoidance_spec)

TOL = 1e-9


def traj(points, v=1.0, th=0.0):
    """Car trajectory with state order s = [x, y, v, theta] (paper Sec. 5/7: s=[x y v theta]^T)."""
    return np.array([[x, y, v, th] for x, y in points], float)


def cart(**entries):
    """20-step cartpole window, state order [x, xdot, theta, thetadot]; entries like t7_theta=9."""
    s = np.zeros((20, 4))
    idx = {"x": 0, "xdot": 1, "theta": 2, "thetadot": 3}
    for key, val in entries.items():
        t, name = key[1:].split("_")
        s[int(t), idx[name]] = val
    return s


def rs(f, s, t=0):
    return float(f.rho(s, t)), bool(f.sat(s, t))


MU = above(0, 0)                       # mu: s[0] > 0, rho = s[0]
S1 = np.array([3, 1, 2, 5, 4], float).reshape(-1, 1)
O0 = [(0.0, 0.0)]                      # obstacle at origin, c = 1, so rho = (min distance) - 1
SPEC_C1 = collision_avoidance_spec(O0, 1.0, 2)

CASES = [
    # ---------------- collision avoidance: safe / unsafe / boundary ----------------
    dict(id="col_safe", group="Collision", spec="G[0,2] d>c, obstacle (0,0), c=1",
         expected=4.0, sat=True, derivation="d=5,10,15 -> min 5 -> 5-1",
         fn=lambda: rs(SPEC_C1, traj([(3, 4), (6, 8), (9, 12)]))),
    dict(id="col_unsafe", group="Collision", spec="same",
         expected=-0.5, sat=False, derivation="d=5,0.5,10 -> min 0.5 -> 0.5-1",
         fn=lambda: rs(SPEC_C1, traj([(3, 4), (0.3, 0.4), (6, 8)]))),
    dict(id="col_boundary", group="Collision", spec="same",
         expected=0.0, sat=False, derivation="d=5,1,.. -> min 1 -> 1-1 = 0 ; rho=0 is NOT satisfied (strict >0)",
         fn=lambda: rs(SPEC_C1, traj([(3, 4), (1, 0), (6, 8)]))),
    dict(id="col_two_obstacles", group="Collision", spec="G[0,0] d>c, obstacles (0,0),(10,0), c=1",
         expected=1.0, sat=True, derivation="point (8,0): d=8 and 2 -> nearest is 2 -> 2-1",
         fn=lambda: rs(collision_avoidance_spec([(0, 0), (10, 0)], 1.0, 0), traj([(8, 0)]))),
    dict(id="col_window_t0", group="Collision", spec="G[0,1] d>c at t=0 on 3-step path",
         expected=4.0, sat=True, derivation="window = steps 0,1: d=5,10 -> min 5 -> 4 (violation at step 2 is outside)",
         fn=lambda: rs(collision_avoidance_spec(O0, 1.0, 1), traj([(3, 4), (6, 8), (0.3, 0.4)]), 0)),
    dict(id="col_window_t1", group="Collision", spec="G[0,1] d>c at t=1 on same path",
         expected=-0.5, sat=False, derivation="window = steps 1,2: d=10,0.5 -> min 0.5 -> -0.5",
         fn=lambda: rs(collision_avoidance_spec(O0, 1.0, 1), traj([(3, 4), (6, 8), (0.3, 0.4)]), 1)),
    dict(id="col_ignores_v_theta", group="Collision", spec="safe case with v=99, theta=3.14",
         expected=4.0, sat=True, derivation="only s[0],s[1] (x,y) enter the spec; v and theta must not matter",
         fn=lambda: rs(SPEC_C1, traj([(3, 4), (6, 8), (9, 12)], v=99.0, th=3.14))),

    # ---------------- cartpole spec: G[0,19] (|theta|<12) and (|x|<2.4) ----------------
    dict(id="cp_safe", group="Cartpole", spec="G[0,19] |theta|<12 & |x|<2.4",
         expected=0.4, sat=True,
         derivation="step7 theta=9 -> 3 ; step12 x=2.0 -> 0.4 ; all others min(12,2.4)=2.4 -> overall 0.4",
         fn=lambda: rs(cartpole_spec(), cart(t7_theta=9, t12_x=2.0))),
    dict(id="cp_unsafe", group="Cartpole", spec="same",
         expected=-1.0, sat=False, derivation="step15 theta=-13 -> 12-13 = -1 (smallest margin)",
         fn=lambda: rs(cartpole_spec(), cart(t15_theta=-13))),
    dict(id="cp_boundary", group="Cartpole", spec="same",
         expected=0.0, sat=False, derivation="step10 theta=12 exactly -> 12-12 = 0 -> not satisfied",
         fn=lambda: rs(cartpole_spec(), cart(t10_theta=12))),
    dict(id="cp_wrong_index", group="Cartpole", spec="paper-PRINTED index 3 for the angle, unsafe case",
         expected=2.4, sat=True,
         derivation="index 3 is thetadot=0 in [x,xdot,theta,thetadot] -> angle ignored -> min(12,2.4)=2.4: WRONGLY safe",
         fn=lambda: rs(Always(0, 19, abs_below(3, 12) & abs_below(0, 2.4)), cart(t15_theta=-13))),

    # ---------------- operators on 1-D signal s=[3,1,2,5,4], mu: s>0 ----------------
    dict(id="op_pred", group="Operators", spec="mu at t=1", expected=1.0, sat=True,
         derivation="rho(mu)=s[1]=1", fn=lambda: rs(MU, S1, 1)),
    dict(id="op_always", group="Operators", spec="G[0,2] mu at t=0", expected=1.0, sat=True,
         derivation="min(3,1,2)", fn=lambda: rs(Always(0, 2, MU), S1)),
    dict(id="op_always_offset", group="Operators", spec="G[1,2] mu at t=0", expected=1.0, sat=True,
         derivation="window steps 1,2: min(1,2)", fn=lambda: rs(Always(1, 2, MU), S1)),
    dict(id="op_eventually", group="Operators", spec="F[0,2] mu at t=0", expected=3.0, sat=True,
         derivation="max(3,1,2)", fn=lambda: rs(Eventually(0, 2, MU), S1)),
    dict(id="op_eventually_offset", group="Operators", spec="F[1,3] mu at t=0", expected=5.0, sat=True,
         derivation="window steps 1..3: max(1,2,5)", fn=lambda: rs(Eventually(1, 3, MU), S1)),
    dict(id="op_and", group="Operators", spec="(s0>0) AND (s1>0), s=(2,-1)", expected=-1.0, sat=False,
         derivation="min(2,-1)", fn=lambda: rs(And(above(0, 0), above(1, 0)), np.array([[2.0, -1.0]]))),
    dict(id="op_or", group="Operators", spec="(s0>0) OR (s1>0), s=(2,-1)", expected=2.0, sat=True,
         derivation="max(2,-1)", fn=lambda: rs(Or(above(0, 0), above(1, 0)), np.array([[2.0, -1.0]]))),
    dict(id="op_not", group="Operators", spec="NOT (s0>0), s=(2,-1)", expected=-2.0, sat=False,
         derivation="-(2)", fn=lambda: rs(Not(above(0, 0)), np.array([[2.0, -1.0]]))),
    dict(id="op_until_true", group="Operators", spec="(s>0) U[0,3] (s>4), s=[1,2,5,0]", expected=1.0, sat=True,
         derivation="u=2: min(rho_q=5-4=1, guard=min(1,2,5)=1)=1 ; u=0:-3, u=1:-2, u=3:-4 -> max 1",
         fn=lambda: rs(Until(0, 3, above(0, 0), above(0, 4)), np.array([1, 2, 5, 0.]).reshape(-1, 1))),
    dict(id="op_until_q_never", group="Operators", spec="same formula, s=[1,2,1,0]", expected=-2.0, sat=False,
         derivation="rho_q = -3,-2,-3,-4 ; guards 1,1,1,0 -> min(.,.) = -3,-2,-3,-4 -> max -2",
         fn=lambda: rs(Until(0, 3, above(0, 0), above(0, 4)), np.array([1, 2, 1, 0.]).reshape(-1, 1))),
    dict(id="op_until_p_broken", group="Operators", spec="same formula, s=[1,-2,5,0]", expected=-2.0, sat=False,
         derivation="q holds at u=2 (margin 1) but guard min(1,-2,5)=-2 -> -2 ; others -3,-6,-4 -> max -2",
         fn=lambda: rs(Until(0, 3, above(0, 0), above(0, 4)), np.array([1, -2, 5, 0.]).reshape(-1, 1))),
    dict(id="op_nested", group="Operators", spec="G[0,1] F[0,1] mu on [1,-1,2,-3,4], t=0", expected=1.0, sat=True,
         derivation="F(0)=max(1,-1)=1 ; F(1)=max(-1,2)=2 ; G -> min(1,2)=1",
         fn=lambda: rs(Always(0, 1, Eventually(0, 1, MU)), np.array([1, -1, 2, -3, 4.]).reshape(-1, 1))),
    dict(id="op_dual_a", group="Operators", spec="NOT G[0,2] mu on [3,1,2]", expected=-1.0, sat=False,
         derivation="-(min(3,1,2)) = -1", fn=lambda: rs(Not(Always(0, 2, MU)), np.array([3, 1, 2.]).reshape(-1, 1))),
    dict(id="op_dual_b", group="Operators", spec="F[0,2] NOT mu on [3,1,2]", expected=-1.0, sat=False,
         derivation="max(-3,-1,-2) = -1 (equals the line above: duality)",
         fn=lambda: rs(Eventually(0, 2, Not(MU)), np.array([3, 1, 2.]).reshape(-1, 1))),
]


def run_case(c):
    rho, sat = c["fn"]()
    diff = abs(rho - c["expected"])
    ok = diff <= TOL and sat == c["sat"]
    return dict(id=c["id"], group=c["group"], spec=c["spec"], expected=c["expected"], computed=rho,
                diff=diff, sat_expected=c["sat"], sat_computed=sat, derivation=c["derivation"], ok=ok)


# ----------------------------------------------------------------------------
# Specification / units / horizon / state-ordering checks against the paper
# ----------------------------------------------------------------------------
def param_checks():
    sig = inspect.signature(cartpole_spec).parameters
    col_h = collision_avoidance_spec([(0, 0)], 1.0, 5).horizon
    rows = [
        ("Cartpole max pole angle (degrees)", "12", "Sec. 2", sig["max_angle"].default, 12.0),
        ("Cartpole max cart position (units)", "2.4", "Sec. 2", sig["max_pos"].default, 2.4),
        ("Cartpole horizon: G[0,19] -> 20 samples", "19", "Sec. 2", cartpole_spec().horizon, 19),
        ("Cartpole state order [x,xdot,theta,thetadot]: x index", "0", "Sec. 2", sig["x_idx"].default, 0),
        ("Cartpole state order: theta index (paper prints 3)", "2", "Sec. 2 (index 3 = thetadot)", sig["theta_idx"].default, 2),
        ("Collision spec horizon H", "5", "Sec. 7.3", col_h, 5),
        ("Collision state order s=[x y v theta]: position idx", "(0,1)", "Sec. 7.1", tuple(inspect.signature(collision_avoidance_spec).parameters["pos"].default), (0, 1)),
    ]
    return [dict(item=a, paper=b, source=c, code=d, ok=(d == e)) for a, b, c, d, e in rows]


# Call-site constants (not stored in stl.py): use these when running the experiments.
PAPER_CALLSITE = [("F1Tenth safety distance c", "0.3 m", "Sec. 7.3"),
                  ("Race-car safety distance c", "5.4 m", "Sec. 7.3"),
                  ("Prediction horizon / history length", "H = h = 5 steps", "Sec. 7.3-7.4"),
                  ("Units", "positions and c in meters; rho has the same unit as c", "Sec. 7.1-7.3")]


def fmt(x):
    return f"{x:.4g}" if isinstance(x, float) else str(x)


def main(out="results"):
    os.makedirs(out, exist_ok=True)
    rows = [run_case(c) for c in CASES]
    md = ["| # | group | case | specification | expected (hand) | computed | sat expected / computed | pass | hand derivation |",
          "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | {r['group']} | {r['id']} | {r['spec']} | {fmt(r['expected'])} | {r['computed']:.6g} | "
                  f"{r['sat_expected']} / {r['sat_computed']} | {'PASS' if r['ok'] else 'FAIL'} | {r['derivation']} |")
    pc = param_checks()
    md += ["", "| spec parameter / ordering | paper | source | code | match |", "|---|---|---|---|---|"]
    md += [f"| {p['item']} | {p['paper']} | {p['source']} | {p['code']} | {'OK' if p['ok'] else 'MISMATCH'} |" for p in pc]
    md += ["", "Set at the call site (not stored in stl.py):", ""]
    md += [f"- {a}: {b} ({c})" for a, b, c in PAPER_CALLSITE]
    text = "\n".join(md)
    print(text)
    with open(os.path.join(out, "stl_expected_vs_computed.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out, "stl_expected_vs_computed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "case", "specification", "expected_hand", "computed", "abs_diff",
                    "sat_expected", "sat_computed", "pass", "derivation"])
        for r in rows:
            w.writerow([r["group"], r["id"], r["spec"], r["expected"], r["computed"], r["diff"],
                        r["sat_expected"], r["sat_computed"], r["ok"], r["derivation"]])
    n_ok = sum(r["ok"] for r in rows) + sum(p["ok"] for p in pc)
    n = len(rows) + len(pc)
    print(f"\n{n_ok}/{n} checks passed")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    sys.exit(main())