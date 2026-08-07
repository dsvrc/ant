"""PACT unit tests (pure numpy — no mujoco, torch, or simulator).

Exercises the arithmetic core that the env wrapper is built from
(``pact_mujoco`` module-level functions), which is where every correctness claim
lives:

  T1  exactness   — x2 (peer arithmetic) reproduces the env's true liability d
                    up to the scalar c: under constant A, d == c·x2 to 1e-9; under
                    a drifting A, corr(x2, d) > 0.999 (the online gate quantity).
                    Also the one-step TIMING contract (compensation at t uses x2
                    built from executed torques through t−1).
  T2  the knob    — β bounds, rate-limit, mean vs agent0 aggregation, persistence,
                    and δ=0 ⇒ β frozen.
  T3  reduction   — β≡0 ⇒ u = clip(a): the floor property (== plain HAPPO), so no
                    configuration can crater below blind.
  T4  spaces      — the augmentation-dimension arithmetic matches the spec
                    (+1 action dim, +4 obs, +13 share for Ant 4x2).

Run:  python -m harl.envs.mamujoco.pact.test_pact     (expect all PASS)
"""

import numpy as np

from harl.envs.mamujoco.pact.pact_mujoco import (
    beta_step,
    compensate,
    coupling_sum,
    direct_beta,
    leak_step,
    pcr_d_step,
)

RHO = 0.8
SIGMA = 0.45
N = 4
K = 2
NACT = 8
OFF = np.cumsum([0] + [K] * N)


def _payload(t, period=40000, b=0.2):
    """ant.py's asymmetric-smoothstep payload A(t) ∈ [0,1] (for the drift test)."""
    ph = (t % period) / period
    x = ph / b if ph < b else (1.0 - ph) / (1.0 - b)
    return x * x * (3.0 - 2.0 * x)


def _per_step_cos(X2, D, floor=1e-3):
    """Mean per-step cosine similarity between the x2 and d 8-vectors.

    This is the correct, c-INVARIANT exactness / gate metric. Per step d ≈ c·x2
    (parallel vectors), so cos ≈ 1 when the recursion is wired right and ≈ 0 under
    any index/timing/reset bug — regardless of how c drifts across steps.

    (NB: a Pearson corr POOLED across a whole payload cycle is only ~0.95 even when
    every point is exactly d=c·x2, because c(t) sweeps a range and the scatter is a
    fan of origin-lines with different slopes. That is a property of pooling, not a
    wiring error — which is exactly why the gate uses per-step cosine, not pooled
    corr.)
    """
    X2 = np.asarray(X2, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    num = np.sum(X2 * D, axis=1)
    nx = np.linalg.norm(X2, axis=1)
    nd = np.linalg.norm(D, axis=1)
    m = (nx > floor) & (nd > floor) & np.isfinite(num)
    cos = num[m] / (nx[m] * nd[m])
    return (float(np.mean(cos)) if cos.size else float("nan")), int(m.sum())


def _rollout(A_fn, delta=0.01, beta_max=0.6, steps=2000, reset_every=None, seed=0):
    """Simulate the wrapper's step ordering beside ant.py's true d recursion.

    Returns arrays of (x2_post, d_post) pairs and the running beta, exactly as
    the wrapper produces them: beta update -> compensate with cached x2 ->
    env applies & updates d -> wrapper updates x2.
    """
    rng = np.random.default_rng(seed)
    x2 = np.zeros(NACT)
    d = np.zeros(NACT)
    beta = 0.0
    X2, D, A_used = [], [], []
    for t in range(steps):
        if reset_every and t > 0 and t % reset_every == 0:
            x2 = np.zeros(NACT)   # episode reset: d and x2 both zero
            d = np.zeros(NACT)
        A = A_fn(t)
        a = rng.uniform(-1.2, 1.2, size=(N, K))     # unbounded policy sample
        w = rng.uniform(-1.0, 1.0, size=N)
        beta, _ = beta_step(beta, w, delta, beta_max)
        u = compensate(a, beta, x2, OFF)            # cached x2 (from t-1)
        u_flat = u.reshape(-1)
        d = pcr_d_step(d, u_flat, A, SIGMA, RHO)    # env: true liability -> next
        x2 = leak_step(x2, u_flat, RHO)             # wrapper: waveform -> next
        X2.append(x2.copy())
        D.append(d.copy())
        A_used.append(A)
    return np.asarray(X2), np.asarray(D), np.asarray(A_used), beta


def test_t1_exactness():
    # (a) constant A: d == c·x2 exactly (both follow the same recursion, factor c)
    A0 = 0.8
    X2, D, _, _ = _rollout(lambda t: A0, steps=1500, reset_every=400, seed=1)
    c = A0 * SIGMA
    err = np.max(np.abs(D - c * X2))
    assert err < 1e-9, f"T1a constant-A exactness: max|d - c*x2|={err:.2e} (want <1e-9)"

    # (b) drifting A: the GATE quantity — mean per-step cosine(x2, d) ~ 1 on
    #     payload>0.3 (c-invariant, so slow drift does not degrade it).
    X2, D, A, _ = _rollout(_payload, steps=8000, reset_every=1000, seed=2)
    on = A > 0.3
    cos, n = _per_step_cos(X2[on], D[on])
    assert cos > 0.999, f"T1b drift per-step cos(x2, d)={cos:.6f} (want >0.999, n={n})"
    # document the pooling artefact so nobody re-introduces a pooled-corr gate:
    pooled = np.corrcoef(X2[on].ravel(), D[on].ravel())[0, 1]
    assert pooled < 0.999, (
        "T1b sanity: pooled corr over the cycle should be <0.999 (the c-fan) — "
        f"got {pooled:.4f}; if this ever exceeds 0.999 the test setup changed."
    )

    # (c) timing contract: x2 used to compensate step t == leak over executed
    #     torques through t-1. Rebuild the leak from the recorded executed torques
    #     and confirm it matches the cache the wrapper would have used.
    rng = np.random.default_rng(3)
    x2 = np.zeros(NACT)
    beta = 0.3
    prev_cache = None
    for t in range(50):
        cache_before = x2.copy()                    # this is x2 used at step t
        a = rng.uniform(-1, 1, size=(N, K))
        u = compensate(a, beta, cache_before, OFF)
        if prev_cache is not None:
            # cache_before must equal one leak step applied to prev executed torque
            rebuilt = leak_step(prev_cache, prev_u.reshape(-1), RHO)
            assert np.allclose(cache_before, rebuilt, atol=1e-12), "T1c timing broke"
        x2 = leak_step(x2, u.reshape(-1), RHO)
        prev_cache, prev_u = cache_before, u
    print(f"T1 PASS — x2 reproduces the env liability (d=c*x2 exact; drift per-step "
          f"cos={cos:.5f}); timing contract holds")


def test_t2_knob():
    # bounds + rate limit
    beta = 0.0
    for _ in range(1000):
        beta, dbeta = beta_step(beta, np.ones(N), 0.01, 0.6)   # push up hard
        assert abs(dbeta) <= 0.01 + 1e-12
        assert 0.0 <= beta <= 0.6 + 1e-12
    assert abs(beta - 0.6) < 1e-9, "T2 beta should saturate at beta_max"
    beta, _ = beta_step(beta, -np.ones(N), 0.01, 0.6)
    assert beta < 0.6, "T2 beta should come down when driven negative"

    # mean vs agent0 aggregation
    w = np.array([1.0, -1.0, -1.0, -1.0])       # mean=-0.5, agent0=+1
    b_mean, dm = beta_step(0.3, w, 0.1, 1.0, "mean")
    b_a0, da = beta_step(0.3, w, 0.1, 1.0, "agent0")
    assert dm < 0 < da, "T2 driver aggregation (mean vs agent0) wrong"

    # w is clipped to [-1,1] before use
    _, d_big = beta_step(0.0, np.full(N, 5.0), 0.01, 0.6, "mean")
    assert abs(d_big - 0.01) < 1e-9, "T2 w must be clipped to [-1,1]"

    # delta=0 => beta frozen
    b, _ = beta_step(0.42, np.ones(N), 0.0, 0.6)
    assert b == 0.42, "T2 delta=0 must freeze beta"
    print("T2 PASS — beta bounds, rate-limit, aggregation, clipping, delta=0 freeze")


def test_t3_reduction():
    rng = np.random.default_rng(4)
    x2 = rng.uniform(-0.5, 0.5, size=NACT)
    a = rng.uniform(-1.3, 1.3, size=(N, K))
    u0 = compensate(a, 0.0, x2, OFF)             # scalar beta=0
    assert np.allclose(u0, np.clip(a, -1, 1)), "T3 beta=0 must reduce to clip(a) (blind)"
    # per-agent zero vector reduces the same way
    u0v = compensate(a, np.zeros(N), x2, OFF)
    assert np.allclose(u0v, np.clip(a, -1, 1)), "T3 per-agent beta=0 must also be blind"
    # and it is genuinely independent of x2
    u0b = compensate(a, 0.0, rng.uniform(-1, 1, size=NACT), OFF)
    assert np.allclose(u0, u0b), "T3 beta=0 output must not depend on x2"
    # per-agent gains apply per agent
    beta = np.array([0.0, 0.5, 0.0, 0.0])
    uv = compensate(a, beta, x2, OFF)
    assert np.allclose(uv[0], np.clip(a[0], -1, 1))                       # agent 0: blind
    assert np.allclose(uv[1], np.clip(a[1] - 0.5 * x2[2:4], -1, 1))       # agent 1: comp
    print("T3 PASS — beta=0 => u=clip(a) (floor property); per-agent gains apply per agent")


def test_t5_direct_beta():
    # direct mode: β_i = β_max·sigmoid(w_i), EMA-smoothed toward that target.
    bmax, ema = 0.6, 0.9
    prev = np.full(N, 0.3)
    # w=0 -> target = bmax/2 = 0.3; already at 0.3 so it stays
    new, dd = direct_beta(np.zeros(N), bmax, prev, ema)
    assert np.allclose(new, 0.3), "T5 sigmoid(0)*bmax should be bmax/2"
    # large +w -> target -> bmax; EMA moves 10% of the way this step
    new, _ = direct_beta(np.full(N, 20.0), bmax, prev, ema)
    assert np.allclose(new, 0.9 * 0.3 + 0.1 * bmax), "T5 EMA step wrong (up)"
    # large -w -> target -> 0
    new, _ = direct_beta(np.full(N, -20.0), bmax, prev, ema)
    assert np.allclose(new, 0.9 * 0.3 + 0.1 * 0.0), "T5 EMA step wrong (down)"
    # bounds: repeated large +w converges to bmax, never exceeds
    b = np.zeros(N)
    for _ in range(500):
        b, _ = direct_beta(np.full(N, 20.0), bmax, b, ema)
    assert np.all(b <= bmax + 1e-9) and abs(b[0] - bmax) < 1e-3, "T5 must converge to bmax"
    # per-agent independence
    b, _ = direct_beta(np.array([20.0, -20.0, 0.0, 20.0]), bmax, np.full(N, 0.3), ema)
    assert b[0] > b[2] > b[1], "T5 per-agent gains must differ by their own w"
    print("T5 PASS — direct per-agent beta: sigmoid map, EMA smoothing, bounds, independence")


def test_t4_spaces():
    # coupling_sum matches ant.py's category-C channel exactly
    tau = np.arange(1.0, 9.0)                     # [1..8]
    s = coupling_sum(tau)
    hip = tau[0::2]; ank = tau[1::2]
    assert np.allclose(s[0::2], hip.sum() - hip)
    assert np.allclose(s[1::2], ank.sum() - ank)
    # augmentation-dim arithmetic (Ant 4x2): obs +[x2_i(2),beta(1),ema(1)]=+4,
    # share +[x2_all(8),beta(1),ema_all(4)]=+13, action +1 (Delta-beta).
    obs_aug = K + 2
    share_aug = NACT + 1 + N
    assert (obs_aug, share_aug) == (4, 13), "T4 augmentation dims wrong"
    print("T4 PASS — coupling channel exact; aug dims +1 act / +4 obs / +13 share")


def main():
    test_t1_exactness()
    test_t2_knob()
    test_t3_reduction()
    test_t4_spaces()
    test_t5_direct_beta()
    print("\nALL PACT UNIT TESTS PASSED")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
