"""Stage V0 self-test for the diagnostic env  [campaign spec Part 2 / 11.1 item 3].

Gate: **``ant_diag.py`` defaults must pass the golden test before any deployment**
(Prohibition 5). Nothing in the campaign may run until this prints ALL PASS.

Four groups:

  V0.1 GOLDEN EQUIVALENCE — with no ``ANT_PCR_*`` env var set, ``ant_diag.AntEnv``
       and ``ant_pcr_v1.AntEnv`` (the frozen reference copy of the deployed
       ant.py) are driven from the same seed through an identical fixed 400-step
       action sequence spanning 2 episode resets; obs / reward / done streams and
       the internal ``_d`` / ``_clock`` must agree to 1e-12.
  V0.2 KNOB TESTS — each knob does exactly its one thing and nothing else:
       freeze => pcr_payload constant; mask=off => pcr_load == 0; dcap =>
       pcr_loadmax <= cap; severity override => d scales proportionally.
  V0.3 INFO-KEY INVARIANTS — pcr_d_applied[t] == pcr_d_next[t-1] (the identity
       the ProbeShim's feed-forward cancellation relies on), pcr_sat_frac matches
       a recomputation, pcr_clock ticks once per step.
  V0.4 CATEGORY-C LITMUS (spec §9.2 — every redesign must re-pass these):
       (a) N=1 vanishing: with one leg the neighbour-sum is empty => d == 0 for
           all t no matter how the payload drifts;
       (b) frozen-partner persistence: with CONSTANT commanded torque, d still
           tracks A(t) => a genuine non-stationarity, not a co-learning artefact.

Run (run machine, needs gym+mujoco):
    python -m harl.envs.mamujoco.diag.test_ant_diag
Optional: --out <path>  also writes the full report to a debug file.

Note: the knob tests re-import ``ant_diag`` under different env vars
(``importlib.reload``), because the knobs are module-level constants read once at
import — exactly how they behave in a real run, where each arm is a fresh process.
"""

import argparse
import importlib
import os
import sys

import numpy as np

from harl.envs.mamujoco.diag import ant_pcr_v1
from harl.envs.mamujoco.diag import ant_diag
from harl.envs.mamujoco.diag.report_io import DebugReport

TOL = 1e-12
_RESET_AT = (150, 300)          # forced resets => the run spans 2 episode resets
_N_STEPS = 400


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _fixed_actions(n=_N_STEPS, seed=12345):
    """The identical fixed action sequence both envs are driven with."""
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, 8))


def _drive(env, actions, seed=0):
    """Drive ``env`` through ``actions``, resetting at _RESET_AT and on done.
    Returns the full recorded stream."""
    env.seed(seed)
    env.reset()
    rec = {k: [] for k in ("obs", "rew", "done", "d", "clock", "payload")}
    for t, a in enumerate(actions):
        if t in _RESET_AT:
            env.reset()
        ob, rew, done, info = env.step(a)
        rec["obs"].append(np.asarray(ob, dtype=np.float64).copy())
        rec["rew"].append(float(rew))
        rec["done"].append(bool(done))
        rec["d"].append(np.asarray(env._d, dtype=np.float64).copy())
        rec["clock"].append(int(env._clock))
        rec["payload"].append(float(info["pcr_payload"]))
        if done:
            env.reset()
    return {k: np.asarray(v) for k, v in rec.items()}


def _reload_diag(**envvars):
    """Re-import ant_diag with the given ANT_PCR_* env vars set; returns the module.
    Restores the previous environment afterwards via the returned callable."""
    saved = {k: os.environ.get(k) for k in
             ("ANT_PCR_SEVERITY", "ANT_PCR_FREEZE_A", "ANT_PCR_MASK",
              "ANT_PCR_DCAP", "ANT_PCR_ORACLE", "ANT_PCR_CORACLE")}
    for k in saved:
        os.environ.pop(k, None)
    for k, v in envvars.items():
        os.environ[k] = str(v)
    mod = importlib.reload(ant_diag)

    def _restore():
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(ant_diag)

    return mod, _restore


def _run_knob(rep, envvars, steps=250, seed=0, const_action=None):
    """Drive a freshly-reloaded ant_diag under ``envvars``; return per-step info."""
    mod, restore = _reload_diag(**envvars)
    try:
        env = mod.AntEnv()
        env.seed(seed)
        env.reset()
        rng = np.random.default_rng(7)
        out = {k: [] for k in ("payload", "load", "loadmax", "d_next", "sat", "clock")}
        for t in range(steps):
            a = const_action if const_action is not None else rng.uniform(-1, 1, 8)
            _, _, done, info = env.step(a)
            out["payload"].append(info["pcr_payload"])
            out["load"].append(info["pcr_load"])
            out["loadmax"].append(info["pcr_loadmax"])
            out["d_next"].append(info["pcr_d_next"].copy())
            out["sat"].append(info["pcr_sat_frac"])
            out["clock"].append(info["pcr_clock"])
            if done:
                env.reset()
        return {k: np.asarray(v) for k, v in out.items()}
    finally:
        restore()


# --------------------------------------------------------------------------
# V0.1 golden equivalence
# --------------------------------------------------------------------------
def test_golden(rep):
    rep.h2("V0.1 — golden equivalence (no env vars set)")
    for k in ("ANT_PCR_SEVERITY", "ANT_PCR_FREEZE_A", "ANT_PCR_MASK",
              "ANT_PCR_DCAP", "ANT_PCR_ORACLE", "ANT_PCR_CORACLE"):
        if os.environ.get(k) is not None:
            rep.line(f"  !! {k}={os.environ[k]} is set — the golden test requires a "
                     f"clean environment. Unset it and re-run.")
            return False
    acts = _fixed_actions()
    ref = _drive(ant_pcr_v1.AntEnv(), acts)
    new = _drive(importlib.reload(ant_diag).AntEnv(), acts)

    ok = True
    for key in ("obs", "rew", "d", "payload"):
        err = float(np.max(np.abs(ref[key] - new[key])))
        good = err <= TOL
        ok = ok and good
        rep.line(f"  max|Δ{key:<8}| = {err:.3e}   {'OK' if good else 'FAIL'}")
    for key in ("done", "clock"):
        good = bool(np.all(ref[key] == new[key]))
        ok = ok and good
        rep.line(f"  {key:<12} identical: {good}   {'OK' if good else 'FAIL'}")
    rep.line(f"  ({_N_STEPS} steps, forced resets at {_RESET_AT}, "
             f"{int(ref['done'].sum())} natural terminations)")
    rep.verdict("V0.1 GOLDEN EQUIVALENCE", ok)
    return ok


# --------------------------------------------------------------------------
# V0.2 knob tests
# --------------------------------------------------------------------------
def test_knobs(rep):
    rep.h2("V0.2 — knob tests (each knob does exactly its one thing)")
    ok = True

    # freeze => payload constant at the requested value
    for a_val in (0.0, 0.5, 1.0):
        r = _run_knob(rep, {"ANT_PCR_FREEZE_A": a_val})
        good = bool(np.all(np.abs(r["payload"] - a_val) <= TOL))
        ok = ok and good
        rep.line(f"  FREEZE_A={a_val:<4} -> payload const {a_val}: {good}  "
                 f"(range [{r['payload'].min():.3f}, {r['payload'].max():.3f}])"
                 f"   {'OK' if good else 'FAIL'}")

    # mask=off => no coupling at all => pcr_load identically 0
    r = _run_knob(rep, {"ANT_PCR_MASK": "off"})
    good = bool(np.all(r["load"] == 0.0))
    ok = ok and good
    rep.line(f"  MASK=off   -> pcr_load == 0: {good} (max {r['load'].max():.3e})"
             f"   {'OK' if good else 'FAIL'}")

    # mask=hip => only the hip channel couples => ankle entries of d are 0
    r = _run_knob(rep, {"ANT_PCR_MASK": "hip", "ANT_PCR_FREEZE_A": 1.0})
    d = r["d_next"]
    good = bool(np.all(d[:, 1::2] == 0.0)) and float(np.abs(d[:, 0::2]).max()) > 1e-6
    ok = ok and good
    rep.line(f"  MASK=hip   -> ankle d == 0 & hip d != 0: {good}  "
             f"(|d_hip|max={np.abs(d[:, 0::2]).max():.3f}, "
             f"|d_ank|max={np.abs(d[:, 1::2]).max():.3e})   {'OK' if good else 'FAIL'}")

    # mask=ankle => only the ankle channel couples => hip entries of d are 0
    r = _run_knob(rep, {"ANT_PCR_MASK": "ankle", "ANT_PCR_FREEZE_A": 1.0})
    d = r["d_next"]
    good = bool(np.all(d[:, 0::2] == 0.0)) and float(np.abs(d[:, 1::2]).max()) > 1e-6
    ok = ok and good
    rep.line(f"  MASK=ankle -> hip d == 0 & ankle d != 0: {good}  "
             f"(|d_hip|max={np.abs(d[:, 0::2]).max():.3e}, "
             f"|d_ank|max={np.abs(d[:, 1::2]).max():.3f})   {'OK' if good else 'FAIL'}")

    # dcap => |d| clipped
    cap = 0.1
    r = _run_knob(rep, {"ANT_PCR_DCAP": cap, "ANT_PCR_FREEZE_A": 1.0})
    good = bool(np.all(r["loadmax"] <= cap + TOL))
    ok = ok and good
    rep.line(f"  DCAP={cap}   -> pcr_loadmax <= cap: {good} "
             f"(max {r['loadmax'].max():.4f})   {'OK' if good else 'FAIL'}")

    # severity override => d scales proportionally (same actions & seed => same
    # commanded torques only if the trajectory is identical, so compare the FIRST
    # step, where both envs are in the identical reset state: d1 = (1-rho)*A*sigma*s).
    d_first = {}
    for sev in (0.45, 0.9):
        r = _run_knob(rep, {"ANT_PCR_SEVERITY": sev, "ANT_PCR_FREEZE_A": 1.0}, steps=1)
        d_first[sev] = r["d_next"][0]
    ratio = np.abs(d_first[0.9]) / np.maximum(np.abs(d_first[0.45]), 1e-12)
    good = bool(np.all(np.abs(ratio - 2.0) < 1e-6))
    ok = ok and good
    rep.line(f"  SEVERITY 0.45 vs 0.9 -> d ratio == 2 on step 1: {good} "
             f"(ratio range [{ratio.min():.6f}, {ratio.max():.6f}])"
             f"   {'OK' if good else 'FAIL'}")

    rep.verdict("V0.2 KNOB TESTS", ok)
    return ok


# --------------------------------------------------------------------------
# V0.3 info-key invariants
# --------------------------------------------------------------------------
def test_info_keys(rep):
    rep.h2("V0.3 — info-key invariants (the ProbeShim contract)")
    mod, restore = _reload_diag(ANT_PCR_FREEZE_A=1.0)
    ok = True
    try:
        env = mod.AntEnv()
        env.seed(3)
        env.reset()
        rng = np.random.default_rng(11)
        prev_next = np.zeros(8)
        prev_clock = None
        max_link_err = 0.0
        max_sat_err = 0.0
        clock_ok = True
        n_reset = 0
        for t in range(300):
            a = rng.uniform(-1, 1, 8)
            tau = np.clip(a, -1, 1)
            _, _, done, info = env.step(a)
            # (i) pcr_d_applied[t] == pcr_d_next[t-1]  (0 at an episode start)
            max_link_err = max(max_link_err,
                               float(np.max(np.abs(info["pcr_d_applied"] - prev_next))))
            # (ii) pcr_sat_frac matches a recomputation from d_applied
            sat = float(np.mean(np.abs(tau + info["pcr_d_applied"]) > 1.0))
            max_sat_err = max(max_sat_err, abs(sat - info["pcr_sat_frac"]))
            # (iii) the clock ticks exactly once per step
            if prev_clock is not None and info["pcr_clock"] != prev_clock + 1:
                clock_ok = False
            prev_clock = info["pcr_clock"]
            prev_next = info["pcr_d_next"].copy()
            if done:
                env.reset()
                prev_next = np.zeros(8)   # reset_model zeroes d
                n_reset += 1
        good = max_link_err <= TOL
        ok = ok and good
        rep.line(f"  max|pcr_d_applied[t] - pcr_d_next[t-1]| = {max_link_err:.3e}"
                 f"   {'OK' if good else 'FAIL'}")
        good = max_sat_err <= TOL
        ok = ok and good
        rep.line(f"  max|pcr_sat_frac - recomputed| = {max_sat_err:.3e}"
                 f"   {'OK' if good else 'FAIL'}")
        ok = ok and clock_ok
        rep.line(f"  pcr_clock ticks once per step: {clock_ok}"
                 f"   {'OK' if clock_ok else 'FAIL'}")
        # (iv) info arrays must be COPIES: reset_model zeroes self._d in place, so a
        #      handed-out reference would be silently mutated.
        _, _, _, info = env.step(np.zeros(8))
        held = info["pcr_d_next"]
        before = held.copy()
        env.reset()
        good = bool(np.array_equal(held, before))
        ok = ok and good
        rep.line(f"  info arrays survive a reset (are copies): {good}"
                 f"   {'OK' if good else 'FAIL'}")
        rep.line(f"  ({n_reset} resets exercised)")
    finally:
        restore()
    rep.verdict("V0.3 INFO-KEY INVARIANTS", ok)
    return ok


# --------------------------------------------------------------------------
# V0.4 category-C litmus asserts
# --------------------------------------------------------------------------
def _pcr_update_rule(tau, d, A, severity, rho=0.8, n_legs=4):
    """The ant_diag update rule, re-implemented here for the N=1 litmus at an
    arbitrary leg count. ``tau``/``d`` are length 2*n_legs (hip, ankle per leg)."""
    hip, ank = tau[0::2], tau[1::2]
    s = np.empty_like(tau)
    s[0::2] = hip.sum() - hip
    s[1::2] = ank.sum() - ank
    return rho * d + (1.0 - rho) * (A * severity * s)


def test_litmus_n1(rep):
    """(a) N=1 vanishing: one leg => the neighbour-sum is empty => d == 0 always,
    no matter how the payload drifts. This is what makes PCR category C."""
    rep.h2("V0.4a — category-C litmus: N=1 vanishing")
    rng = np.random.default_rng(2)
    sev = 0.9
    worst = {}
    for n_legs in (1, 4):
        d = np.zeros(2 * n_legs)
        peak = 0.0
        for t in range(4000):
            A = _payload_ref(t)
            tau = rng.uniform(-1, 1, 2 * n_legs)
            d = _pcr_update_rule(tau, d, A, sev, n_legs=n_legs)
            peak = max(peak, float(np.max(np.abs(d))))
        worst[n_legs] = peak
    good = worst[1] == 0.0 and worst[4] > 1e-3
    rep.line(f"  N=1 legs: max|d| over 4000 drifting steps = {worst[1]:.3e} "
             f"(must be exactly 0)")
    rep.line(f"  N=4 legs: max|d| over 4000 drifting steps = {worst[4]:.3f} "
             f"(must be > 0 — the effect exists)")
    rep.verdict("V0.4a N=1 VANISHING", good)
    return good


def _payload_ref(clock, P=40000, B=0.2):
    ph = (clock % P) / P
    x = ph / B if ph < B else (1.0 - ph) / (1.0 - B)
    return x * x * (3.0 - 2.0 * x)


def test_litmus_frozen_partners(rep):
    """(b) Frozen-partner persistence: with CONSTANT commanded torque (partners
    frozen — no co-learning at all) d must still track A(t). Run on the REAL env:
    for each phase, pin the clock and hold a constant action long enough for the
    leaky accumulator to converge (rho=0.8 => ~5-step time constant)."""
    rep.h2("V0.4b — category-C litmus: frozen-partner persistence")
    mod, restore = _reload_diag()
    ok = True
    try:
        env = mod.AntEnv()
        env.seed(5)
        const_a = np.array([0.6, -0.3, 0.6, -0.3, 0.6, -0.3, 0.6, -0.3])
        phases = np.linspace(0, 40000, 21, dtype=int)[:-1]
        A_true, d_obs = [], []
        for ph in phases:
            env.reset()
            env._clock = int(ph)
            last = None
            for _ in range(60):          # >> the 5-step time constant
                _, _, _, info = env.step(const_a)   # done ignored: we probe the
                last = info                          # recursion, not locomotion
            A_true.append(_payload_ref(int(ph)))
            d_obs.append(float(np.mean(np.abs(last["pcr_d_next"]))))
        A_true = np.asarray(A_true)
        d_obs = np.asarray(d_obs)
        corr = float(np.corrcoef(A_true, d_obs)[0, 1])
        spans = float(d_obs.max() - d_obs.min())
        good = corr > 0.99 and spans > 1e-3
        ok = good
        rep.line(f"  constant tau, clock swept over one period:")
        rep.line(f"    corr(A(t), mean|d|) = {corr:.4f}   (must be > 0.99)")
        rep.line(f"    mean|d| range = [{d_obs.min():.4f}, {d_obs.max():.4f}] "
                 f"(span {spans:.4f}, must be > 0)")
        rep.line(f"  => the driver keeps moving with FROZEN partners: a genuine "
                 f"non-stationarity, not a co-learning artefact.")
    finally:
        restore()
    rep.verdict("V0.4b FROZEN-PARTNER PERSISTENCE", ok)
    return ok


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="V0 self-test for the PCR diag env.")
    ap.add_argument("--out", default=None,
                    help="also write the full report to this debug file "
                         "(default: ./diag_out/v0/v0_ant_diag.md)")
    args = ap.parse_args(argv)
    out = args.out or os.path.join("diag_out", "v0", "v0_ant_diag.md")

    rep = DebugReport(out, title="V0 — ant_diag self-test",
                      subtitle="golden equivalence, knob isolation, info-key "
                               "invariants, category-C litmus")
    results = {
        "V0.1 golden equivalence": test_golden(rep),
        "V0.2 knob tests": test_knobs(rep),
        "V0.3 info-key invariants": test_info_keys(rep),
        "V0.4a N=1 vanishing": test_litmus_n1(rep),
        "V0.4b frozen-partner persistence": test_litmus_frozen_partners(rep),
    }
    ok = all(results.values())
    rep.h2("SUMMARY")
    for k, v in results.items():
        rep.line(f"  {'PASS' if v else 'FAIL'}  {k}")
    rep.verdict(
        "V0 — ant_diag.py may be deployed over gym/envs/mujoco/ant.py" if ok
        else "V0 — DO NOT DEPLOY (Prohibition 5: defaults must pass the golden test)",
        ok,
    )
    rep.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
