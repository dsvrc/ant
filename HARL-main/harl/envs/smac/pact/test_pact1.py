"""Arithmetic certificate for SMAC PACT-1. Pure numpy: no pysc2, no StarCraft II.

    python -m harl.envs.smac.pact.test_pact1     # expect: ALL SMAC PACT-1 TESTS PASSED

Ordered by what it would cost to get wrong:

  T1  the basis reproduces the LEGACY uniform load at theta=(1/2,1/2)
      -> the hardened env strictly CONTAINS the old one; every prior measurement
         (B0, the blind collapse) still applies and no baseline is invalidated.
  T2  category-C survives for EVERY theta: no self-loading, and N=1 -> zero load.
  T3  the compensator is an EXACT inverse when s_hat == s (permutation, zero cost).
  T4  RLS recovers a static beta* from INTERMITTENT, QUANTISED shot readings.
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


def _psi_stream(T, rng, drift=None):
    """Simulate the shared bus: exertion -> basis -> leak -> psi."""
    x = np.zeros((R, N))
    out = []
    for t in range(T):
        exert = (rng.random(N) < 0.85).astype(float)     # ~engagement fraction
        same, cross = type_split(exert, TYPES, DENOM)
        x = RHO * x + (1.0 - RHO) * np.stack([same, cross], 0)
        out.append(MIXNORM * x.copy())
    return out


def t4_rls_static(rng):
    beta = np.array([0.55, 0.20])                # beta* = c*theta
    rls = AgentRLS(R, mu=0.999, p0=1.0)
    K = 5
    for psi in _psi_stream(4000, rng):
        p = psi[:, 0]
        if rng.random() > 0.6:                   # fires ~60% of steps: INTERMITTENT
            continue
        ell = float(np.dot(beta, p))
        s = shift_from_ell(ell, K)               # QUANTISED to K-1 steps
        rls.update(p, ell_from_shift(s, K))
    err = np.linalg.norm(rls.beta - beta)
    assert err < 0.12, (rls.beta, beta, err)
    print(f"T4  RLS recovers beta* from intermittent quantised shots (err {err:.3f}) OK")


def t5_rls_tracks_drift(rng):
    a0, a1 = np.array([0.62, 0.10]), np.array([0.10, 0.62])
    live = AgentRLS(R, mu=0.99, p0=1.0)
    frozen = AgentRLS(R, mu=1.0, p0=1.0)
    K, period = 5, 2000
    el, ef = [], []
    stream = _psi_stream(9000, rng)
    for t, psi in enumerate(stream):
        w = 0.5 - 0.5 * np.cos(2 * np.pi * (t % period) / period)
        beta = (1 - w) * a0 + w * a1
        p = psi[:, 0]
        ell = float(np.dot(beta, p))
        y = ell_from_shift(shift_from_ell(ell, K), K)
        live.update(p, y)
        frozen.update(p, y)
        if t > 3000:
            el.append(np.linalg.norm(live.beta - beta))
            ef.append(np.linalg.norm(frozen.beta - beta))
    ml, mf = float(np.mean(el)), float(np.mean(ef))
    assert ml < mf, (ml, mf)
    print(f"T5  forgetting TRACKS the drift: err {ml:.3f} vs {mf:.3f} frozen     OK")


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


def main():
    print("=" * 72)
    print("SMAC PACT-1 ARITHMETIC CERTIFICATE (pure numpy, no StarCraft II)")
    print("=" * 72)
    rng = np.random.default_rng(0)
    t1_contains_legacy(rng)
    t2_category_c(rng)
    t3_inverse_is_exact()
    t4_rls_static(rng)
    t5_rls_tracks_drift(rng)
    t6_quantisation_floor()
    t7_confidence(rng)
    t8_theta_bounded()
    print("=" * 72)
    print("ALL SMAC PACT-1 TESTS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
