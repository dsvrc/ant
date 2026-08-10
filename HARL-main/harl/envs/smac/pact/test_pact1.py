"""Arithmetic certificate for SMAC PACT-1. Pure numpy: no pysc2, no StarCraft II.

    python -m harl.envs.smac.pact.test_pact1     # expect: ALL SMAC PACT-1 TESTS PASSED

Ordered by what it would cost to get wrong:

  T1  the basis reproduces the LEGACY uniform load at theta=(1/2,1/2)
      -> the hardened env strictly CONTAINS the old one; every prior measurement
         (B0, the blind collapse) still applies and no baseline is invalidated.
  T2  category-C survives for EVERY theta: no self-loading, and N=1 -> zero load.
  T3  the compensator is an EXACT inverse when s_hat == s (permutation, zero cost).
  T4  RLS PREDICTS ell from intermittent, quantised shot readings. Prediction --
      not the decomposition of beta -- is the criterion, because the compensator
      consumes ell_hat = beta_hat . psi and never beta_hat itself.
  T4b the regressor's conditioning, which is what decides whether beta can also be
      DECOMPOSED. When both unit types engage at a constant rate psi is nearly a
      fixed vector and only the projection is identifiable -- prediction still works,
      the split does not. Measure this on the real env before claiming otherwise.
  T5  RLS TRACKS a drifting beta*, and beats a non-forgetting estimator.
  T6  the quantisation floor: with K attackable enemies, ell can only ever be read
      to +/- 0.5/(K-1) -- so the estimator's error floor is a property of the
      channel, not a tuning failure. Reported per K.
  T7  confidence rises from 1/(1+r) as the covariance shrinks (the trust prior).
"""

import numpy as np

from harl.envs.smac.pact.pact1_core import (
    MIXNORM, R, THETA_LEGACY, AgentRLS, ell_from_shift, legacy_load, predict_ell,
    shift_from_ell, theta_anchors, theta_at, type_split,
)

N = 8            # 3s5z
DENOM = float(N - 1)
RHO = 0.85
TYPES = np.array([0, 0, 0, 1, 1, 1, 1, 1])       # 3 stalkers, 5 zealots


def t1_contains_legacy(rng):
    for _ in range(500):
        exert = rng.random(N)
        same, cross = type_split(exert, TYPES, DENOM)
        mixed = MIXNORM * (THETA_LEGACY[0] * same + THETA_LEGACY[1] * cross)
        assert np.allclose(mixed, legacy_load(exert, DENOM), atol=1e-12)
    print("T1  theta=(1/2,1/2) reproduces the legacy uniform load EXACTLY      OK")
    print("    => the hardened env CONTAINS the old one; no baseline is invalidated.")


def t2_category_c(rng):
    # (a) no self-loading: bumping ONLY agent i's exertion must not change its own load
    for i in range(N):
        e0 = rng.random(N)
        e1 = e0.copy()
        e1[i] += 0.7
        for th in ([1, 0], [0, 1], [0.3, 0.7], THETA_LEGACY):
            th = np.asarray(th, float)
            s0, c0 = type_split(e0, TYPES, DENOM)
            s1, c1 = type_split(e1, TYPES, DENOM)
            l0 = MIXNORM * (th[0] * s0[i] + th[1] * c0[i])
            l1 = MIXNORM * (th[0] * s1[i] + th[1] * c1[i])
            assert abs(l0 - l1) < 1e-12, (i, th, l0, l1)
    # (b) N=1: the sum over others is empty, so the load is 0 for ANY theta
    for th in ([1, 0], [0, 1], [0.5, 0.5]):
        s, c = type_split(np.array([1.0]), np.array([0]), 1.0)
        assert abs(MIXNORM * (th[0] * s[0] + th[1] * c[0])) < 1e-12
    print("T2  no self-loading, and N=1 gives zero load, for EVERY theta       OK")


def t3_inverse_is_exact():
    """The channel is a permutation, so pre-shifting by s cancels it at zero cost."""
    for k in range(2, 9):
        for ell in np.linspace(0.0, 1.0, 21):
            s = shift_from_ell(ell, k)
            for pos in range(k):
                pre = (pos - s) % k          # compensator
                landed = (pre + s) % k       # channel
                assert landed == pos, (k, ell, pos, s, landed)
    print("T3  pre-shift o deflection == identity for every (K, ell, target)   OK")


def _psi_stream(T, rng, static=False):
    """Simulate the shared bus: exertion -> basis -> leak -> psi.

    Engagement is deliberately NOT iid across types. Stalkers are ranged and zealots
    are melee, so the two groups are in contact at different times, and they die at
    different rates -- and THAT independent variation is what makes the two-channel
    split identifiable at all.

    ``static=True`` reproduces the degenerate case (both types engaged at a constant
    rate): after the leak, psi is a near-constant vector, the regressor spans one
    direction, and only the PROJECTION beta.psi is identifiable, not beta itself.
    T4b measures exactly that.
    """
    x = np.zeros((R, N))
    alive = np.ones(N)
    p = np.array([0.85, 0.85])
    out = []
    for t in range(T):
        if not static:
            p = np.clip(p + rng.normal(0, 0.05, 2), 0.10, 1.0)   # independent contact
            if t and t % 900 == 0:                                # attrition
                cand = np.where(alive > 0)[0]
                if cand.size > 3:
                    alive[rng.choice(cand)] = 0.0
        pr = np.where(TYPES == 0, p[0], p[1])
        exert = (rng.random(N) < pr).astype(float) * alive
        same, cross = type_split(exert, TYPES, DENOM)
        x = RHO * x + (1.0 - RHO) * np.stack([same, cross], 0)
        out.append(MIXNORM * x.copy())
    return out


def t4_rls_static(rng):
    """The estimator's job is to PREDICT ell, not to decompose beta.

    The compensator consumes ell_hat = beta_hat . psi and never beta_hat itself, so
    prediction is the criterion. Parameter error is reported alongside because it is
    the quantity that degrades first when the regressor is ill-conditioned (T4b).
    """
    beta = np.array([0.55, 0.20])                # beta* = c*theta
    rls = AgentRLS(R, mu=0.999, p0=1.0)
    K = 5
    pe, ee = [], []
    for t, psi in enumerate(_psi_stream(6000, rng)):
        p = psi[:, 0]
        ell = float(np.dot(beta, p))
        if rng.random() < 0.4:                   # fires ~60% of steps: INTERMITTENT
            continue
        rls.update(p, ell_from_shift(shift_from_ell(ell, K), K))   # QUANTISED
        if t > 3000:
            pe.append((predict_ell(rls.beta, p) - ell) ** 2)
            ee.append(ell ** 2)
    rmse, rms = float(np.sqrt(np.mean(pe))), float(np.sqrt(np.mean(ee)))
    perr = float(np.linalg.norm(rls.beta - beta))
    print(f"T4  prediction RMSE {rmse:.4f} vs ell RMS {rms:.4f} "
          f"({rmse / rms:.1%} of signal); |beta_hat-beta*| = {perr:.3f}   OK")
    assert rmse < 0.30 * rms, (rmse, rms)
    assert perr < 0.30, (rls.beta, beta, perr)


def t4b_conditioning(rng):
    """WHY the decomposition is the fragile part -- measure it, do not assume it."""
    print("T4b regressor conditioning decides whether beta can be DECOMPOSED:")
    print(f"    {'engagement':>22}{'cond(E[psi psi^T])':>22}")
    conds = {}
    for label, static in (("varying (realistic)", False), ("constant (degenerate)", True)):
        M = np.array([psi[:, 0] for psi in _psi_stream(6000, rng, static=static)])
        w = np.linalg.eigvalsh(M.T @ M / len(M))
        conds[label] = float(w[-1] / max(w[0], 1e-18))
        print(f"    {label:>22}{conds[label]:>22.1f}")
    assert conds["varying (realistic)"] < conds["constant (degenerate)"]
    print("    -> when both types engage at a constant rate psi is nearly a fixed")
    print("       vector, so only the PROJECTION beta.psi is identifiable. Measure")
    print("       this on the real env before claiming beta itself is tracked.  OK")


def t5_rls_tracks_drift(rng):
    a0, a1 = np.array([0.62, 0.10]), np.array([0.10, 0.62])
    live = AgentRLS(R, mu=0.99, p0=1.0)
    frozen = AgentRLS(R, mu=1.0, p0=1.0)
    K, period = 5, 2000
    el, ef = [], []
    for t, psi in enumerate(_psi_stream(9000, rng)):
        w = 0.5 - 0.5 * np.cos(2 * np.pi * (t % period) / period)
        beta = (1 - w) * a0 + w * a1
        p = psi[:, 0]
        ell = float(np.dot(beta, p))
        y = ell_from_shift(shift_from_ell(ell, K), K)
        live.update(p, y)
        frozen.update(p, y)
        if t > 3000:                              # judged on PREDICTION, as in T4
            el.append((predict_ell(live.beta, p) - ell) ** 2)
            ef.append((predict_ell(frozen.beta, p) - ell) ** 2)
    ml, mf = float(np.sqrt(np.mean(el))), float(np.sqrt(np.mean(ef)))
    assert ml < mf, (ml, mf)
    print(f"T5  forgetting TRACKS the drift: predict RMSE {ml:.4f} vs {mf:.4f} "
          f"frozen  OK")
    print(f"    -> the {ml:.4f} floor is the theorem: a drifting parameter cannot be")
    print("       tracked to zero error.")


def t6_quantisation_floor():
    print("T6  the channel's own resolution limit (NOT a tuning failure):")
    print(f"    {'K enemies':>11}{'ell step':>11}{'max read err':>14}")
    for k in (2, 3, 5, 8):
        print(f"    {k:>11}{1.0/(k-1):>11.3f}{0.5/(k-1):>14.3f}")
    print("    -> with few targets in range a unit simply cannot resolve its own")
    print("       liability finely; this sets the estimator's error floor.      OK")


def t7_confidence(rng):
    rls = AgentRLS(R, mu=0.999, p0=1.0)
    c0 = rls.confidence()
    assert abs(c0 - 1.0 / (1.0 + R)) < 1e-9, c0
    beta = np.array([0.5, 0.2])
    for psi in _psi_stream(1500, rng):
        p = psi[:, 0]
        rls.update(p, float(np.dot(beta, p)))
    c1 = rls.confidence()
    assert c1 > 0.9, c1
    print(f"T7  confidence {c0:.3f} (cold) -> {c1:.3f} (converged)              OK")
    print("    -> reliance ramps itself in; no hand-set warmup.")


def t8_theta_bounded():
    a, b = theta_anchors(1, radius=0.35, conc=0.9, r=R)
    for th in (a, b):
        assert abs(th.sum() - 1.0) < 1e-9, th
        assert np.all(th >= 0.0), th
        assert np.max(np.abs(th - THETA_LEGACY)) <= 0.35 + 1e-9, th
    mid = theta_at(0, 8000, a, b)
    assert abs(mid.sum() - 1.0) < 1e-9
    a0, b0 = theta_anchors(1, radius=0.0, conc=0.9, r=R)
    assert np.allclose(a0, THETA_LEGACY) and np.allclose(b0, THETA_LEGACY)
    print("T8  theta stays on the simplex and within radius of legacy; r=0 ==   OK")
    print("    the legacy env exactly.")


def t9_trace_confidence_disarms(rng):
    """*** THE BUG THAT KILLED THE 3s5z RUN, reproduced in 20 lines of numpy. ***

    RLS with forgetting divides P by mu on EVERY update, so any direction the
    regressor does not excite grows without bound.  On SMAC the regressor is
    near-degenerate by construction (T4b), so tr(P) -- and therefore the ORIGINAL
    confidence() -- decays monotonically even while the prediction beta_hat.psi is
    perfect.  get_agent_action arms the compensator only when conf >= conf_thresh
    (0.5 by default), so a long enough run DISARMS a working compensator and PACT-1
    silently degenerates to blind-plus-dead-features.

    confidence_pred() is keyed to psi^T P psi -- the variance of the quantity the
    re-aim actually consumes -- so it does not decay with the unexcited direction.
    """
    beta = np.array([0.55, 0.20])
    tr_rls = AgentRLS(R, mu=0.995, p0=1.0)
    pr_rls = AgentRLS(R, mu=0.995, p0=1.0)
    last_psi = None
    for psi in _psi_stream(8000, rng, static=True):   # the degenerate SMAC regime
        p = psi[:, 0]
        y = float(np.dot(beta, p))
        tr_rls.update(p, y)
        pr_rls.update(p, y)
        last_psi = p
    c_trace = tr_rls.confidence()
    c_pred = pr_rls.confidence_pred(last_psi)
    err = abs(predict_ell(pr_rls.beta, last_psi) - float(np.dot(beta, last_psi)))
    print(f"T9  degenerate regressor, 8000 updates: prediction error {err:.4f}")
    print(f"    trace confidence {c_trace:.3f}  <-- decays, DISARMS the compensator")
    print(f"    pred  confidence {c_pred:.3f}  <-- tracks what the re-aim uses  OK")
    assert err < 0.05, err                       # the prediction is fine ...
    assert c_trace < 0.5, c_trace                # ... yet the old gate has closed
    assert c_pred > 0.85, c_pred                 # ... and the new one has not
    print("    -> a gate on tr(P) closes on an estimator that is predicting")
    print("       perfectly. That is why pact1_conf_mode defaults to 'pred'.")


def t10_null_warmup_poisons(rng):
    """*** WHY THE ESTIMATOR IS FROZEN AT SEVERITY 0 (pact1_warmup_freeze). ***

    During a severity-0 warmup every deflection reading is y == 0 with psi != 0.
    That is the channel being switched OFF, not evidence about beta*.  Fed to RLS it
    still shrinks/inflates P, so the confidence readout wanders -- and the appended
    obs block stops being constant, which is exactly the warmup confound that makes
    the PACT arm no longer input-equivalent to blind on a byte-identical task.
    Frozen, P is untouched: conf is exactly 1/(1+r) forever and the block is
    constant.
    """
    live = AgentRLS(R, mu=0.995, p0=1.0)
    frozen = AgentRLS(R, mu=0.995, p0=1.0)
    cold = 1.0 / (1.0 + R)
    swing_live = 0.0
    swing_frozen = 0.0
    last_psi = None
    for psi in _psi_stream(8000, rng, static=True):
        p = psi[:, 0]
        live.update(p, 0.0)                       # sigma == 0 -> every reading is 0
        last_psi = p                              # `frozen` is simply never updated
        swing_live = max(swing_live, abs(live.confidence() - cold),
                         abs(live.confidence_pred(p) - cold))
        swing_frozen = max(swing_frozen, abs(frozen.confidence() - cold),
                           abs(frozen.confidence_pred(p) - cold))
    print(f"T10 severity-0 warmup, 8000 null readings (y == 0, psi != 0):")
    print(f"    unfrozen: confidence wandered up to {swing_live:.3f} off the "
          f"{cold:.3f} cold prior")
    print(f"    frozen:   confidence moved {swing_frozen:.1e} -- pinned exactly  OK")
    assert swing_live > 0.05, swing_live          # the null data DID move the gate
    assert swing_frozen < 1e-12, swing_frozen     # frozen: constant for every psi
    print("    -> frozen, BOTH confidence forms sit exactly on the cold prior for")
    print("       any psi, so the appended block is constant and the arm is")
    print("       input-equivalent to blind for the whole warmup.")


def main():
    print("=" * 72)
    print("SMAC PACT-1 ARITHMETIC CERTIFICATE (pure numpy, no StarCraft II)")
    print("=" * 72)
    rng = np.random.default_rng(0)
    t1_contains_legacy(rng)
    t2_category_c(rng)
    t3_inverse_is_exact()
    t4_rls_static(rng)
    t4b_conditioning(rng)
    t5_rls_tracks_drift(rng)
    t6_quantisation_floor()
    t7_confidence(rng)
    t8_theta_bounded()
    t9_trace_confidence_disarms(rng)
    t10_null_warmup_poisons(rng)
    print("=" * 72)
    print("ALL SMAC PACT-1 TESTS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
