"""Arithmetic certificate for PACT-1. Pure numpy: no mujoco, no torch, no gym.

    python -m harl.envs.mamujoco.pact.test_pact1     # expect: ALL PACT-1 TESTS PASSED

Checks, in order of how much they would cost if wrong:

  T1  the basis reproduces ant.py's LEGACY coupling at theta=(1/2,1/2,0)
      -- i.e. the hardened env strictly CONTAINS the old one, so B0=5328 and
      sigma*=0.5 still apply and no baseline needs retraining.
  T2  every basis matrix is zero-diagonal in the leg block (category-C at EVERY
      theta: N=1 => empty sum => d == 0 for any payload).
  T3  FLOOR PROPERTY: g=0 gives u == clip(a) for ANY beta_hat, however wrong.
  T4  RLS recovers a STATIC beta* from clean data.
  T5  RLS TRACKS a drifting beta*, and does so better than a frozen estimate.
  T6  the anchored predictor beats the pure-arithmetic one under parameter error
      (the reason the sensor is worth having).
  T7  sensor noise degrades the estimate gracefully -- no divergence.
"""

import numpy as np

from harl.envs.mamujoco.pact.pact1_mujoco import (
    AgentRLS, ant_basis, basis_waveforms, compensate, predict_load, trust_from_w,
    _MIXNORM,
)

RHO = 0.8
N_LEGS = 4
N = 2 * N_LEGS
R = 3


def _legacy_coupling(tau):
    """ant.py BEFORE the hardening: s[0::2]=hip.sum()-hip ; s[1::2]=ank.sum()-ank."""
    tau = np.asarray(tau, dtype=np.float64)
    s = np.empty_like(tau)
    hip, ank = tau[0::2], tau[1::2]
    s[0::2] = hip.sum() - hip
    s[1::2] = ank.sum() - ank
    return s


def _new_coupling(tau, th, B):
    """W(theta) @ tau via the basis."""
    return (np.asarray(th, dtype=np.float64)[:, None] * basis_waveforms(tau, B)).sum(0)


def t1_contains_legacy(B, rng):
    for _ in range(200):
        tau = rng.uniform(-1, 1, size=N)
        got = _new_coupling(tau, [0.5, 0.5, 0.0], B)
        assert np.allclose(got, _legacy_coupling(tau), atol=1e-12), (got, tau)
    print("T1  theta=(1/2,1/2,0) reproduces the legacy coupling EXACTLY        OK")
    print("    => the hardened env CONTAINS the old one; B0 and sigma* carry over.")


def t2_zero_diagonal(B):
    for k, b in enumerate(B):
        for l in range(N_LEGS):
            blk = b[2 * l:2 * l + 2, 2 * l:2 * l + 2]
            assert np.all(np.abs(blk) < 1e-12), f"B[{k}] self-block at leg {l}"
    # and the N=1 litmus test: one leg acting alone feels nothing
    solo = np.zeros(N)
    solo[0] = solo[1] = 1.0                      # only leg 0 pushes
    for th in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0.3, 0.3, 0.4]):
        assert abs(_new_coupling(solo, th, B)[0]) < 1e-12
        assert abs(_new_coupling(solo, th, B)[1]) < 1e-12
    print("T2  zero-diagonal for every theta; a lone leg loads nothing         OK")


def t3_floor_property(B, rng):
    offs = np.cumsum([0] + [2] * N_LEGS)
    for _ in range(200):
        a = rng.uniform(-1, 1, size=(N_LEGS, 2))
        d_hat = rng.normal(0, 50.0, size=N)      # deliberately absurd estimate
        u = compensate(a, np.zeros(N_LEGS), d_hat, offs)
        assert np.allclose(u, np.clip(a, -1, 1)), "g=0 is NOT the blind policy"
    g = trust_from_w(np.full(N_LEGS, -50.0), np.zeros(N_LEGS), 0.0)
    assert np.all(g < 1e-12), g
    print("T3  FLOOR: g=0 => u == clip(a) for ANY beta_hat, however wrong      OK")


def t4_rls_static(B, rng):
    beta = np.array([0.30, 0.10, 0.05])
    rls = AgentRLS(R, mu=0.9995, p0=10.0)
    for _ in range(4000):
        tau = np.clip(rng.normal(0, 0.4, size=N), -1, 1)
        waves = basis_waveforms(tau, B)
        y = (beta[:, None] * waves).sum(0)
        rls.update(waves[:, 0:2].T, y[0:2])       # agent 0 sees only its 2 joints
    err = np.linalg.norm(rls.beta - beta)
    assert err < 1e-3, (rls.beta, beta, err)
    print(f"T4  RLS recovers a static beta* from its OWN 2 joints (err {err:.2e}) OK")


def t5_rls_tracks_drift(B, rng):
    T, period = 12000, 3000
    a0, a1 = np.array([0.40, 0.05, 0.05]), np.array([0.05, 0.40, 0.05])
    live = AgentRLS(R, mu=0.995, p0=10.0)
    frozen = AgentRLS(R, mu=1.0, p0=10.0)          # mu=1 => never forgets
    el, ef = [], []
    for t in range(T):
        w = 0.5 - 0.5 * np.cos(2 * np.pi * (t % period) / period)
        beta = (1 - w) * a0 + w * a1
        tau = np.clip(rng.normal(0, 0.4, size=N), -1, 1)
        waves = basis_waveforms(tau, B)
        y = (beta[:, None] * waves).sum(0)
        live.update(waves[:, 0:2].T, y[0:2])
        frozen.update(waves[:, 0:2].T, y[0:2])
        if t > T // 3:
            el.append(np.linalg.norm(live.beta - beta))
            ef.append(np.linalg.norm(frozen.beta - beta))
    ml, mf = float(np.mean(el)), float(np.mean(ef))
    assert ml < mf, (ml, mf)
    print(f"T5  forgetting TRACKS the drift: err {ml:.4f} vs {mf:.4f} frozen    OK")
    print(f"    -> the {ml:.4f} floor is the theorem: you cannot track drift to 0.")


def t6_anchor_beats_arithmetic(B, rng):
    """Under parameter error, anchoring on the sensor beats rebuilding from scratch."""
    beta = np.array([0.30, 0.10, 0.05])
    bad = beta * 1.35                              # 35% parameter error
    d = np.zeros(N)
    x2 = np.zeros(N)                               # pure-arithmetic accumulator
    ea, ex = [], []
    for t in range(3000):
        tau = np.clip(rng.normal(0, 0.4, size=N), -1, 1)
        waves = basis_waveforms(tau, B)
        d_next = RHO * d + (1 - RHO) * (beta[:, None] * waves).sum(0)
        anchored = predict_load(d, waves, bad, RHO)          # anchor = the SENSOR
        x2 = predict_load(x2, waves, bad, RHO)               # anchor = its own guess
        if t > 200:
            ea.append(np.mean(np.abs(anchored - d_next)))
            ex.append(np.mean(np.abs(x2 - d_next)))
        d = d_next
    ma, mx = float(np.mean(ea)), float(np.mean(ex))
    assert ma < mx, (ma, mx)
    print(f"T6  anchored predictor beats arithmetic under 35% param error:")
    print(f"    {ma:.5f} vs {mx:.5f}  ({mx/ma:.1f}x)  -- errors do not compound  OK")


def t7_noise_graceful(B, rng):
    beta = np.array([0.30, 0.10, 0.05])
    print("T7  sensor noise degrades the estimate gracefully (no divergence)")
    print(f"    {'noise':>10}{'beta err':>12}")
    errs = []
    for sig in (0.0, 0.005, 0.02, 0.05):
        rls = AgentRLS(R, mu=0.999, p0=10.0)
        for _ in range(6000):
            tau = np.clip(rng.normal(0, 0.4, size=N), -1, 1)
            waves = basis_waveforms(tau, B)
            y = (beta[:, None] * waves).sum(0) + rng.normal(0, sig / (1 - RHO), size=N)
            rls.update(waves[:, 0:2].T, y[0:2])
        err = float(np.linalg.norm(rls.beta - beta))
        print(f"    {sig:>10.3f}{err:>12.4f}")
        assert np.isfinite(err) and err < 1.0, (sig, err)
        errs.append(err)
    # the only claim worth asserting: it never diverges, and the noisiest sensor is
    # worse than the clean one. Strict monotonicity across a 4-point grid is a
    # sampling artifact, not a property, so it is reported and not asserted.
    assert errs[-1] > errs[0] - 1e-9, errs
    print("                                                                    OK")


def main():
    print("=" * 72)
    print("PACT-1 ARITHMETIC CERTIFICATE (pure numpy)")
    print("=" * 72)
    rng = np.random.default_rng(0)
    B = ant_basis(N_LEGS, _MIXNORM)
    t1_contains_legacy(B, rng)
    t2_zero_diagonal(B)
    t3_floor_property(B, rng)
    t4_rls_static(B, rng)
    t5_rls_tracks_drift(B, rng)
    t6_anchor_beats_arithmetic(B, rng)
    t7_noise_graceful(B, rng)
    print("=" * 72)
    print("ALL PACT-1 TESTS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
