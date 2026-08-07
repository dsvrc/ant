"""O-MAX Stage-V0 unit tests (pure numpy; no mujoco).

Validates the ONE load-bearing claim of the ceiling ladder: the hardwired
compensation makes the training env byte-equivalent to the stationary Ant. If this
holds, rung O1 (HAPPO trained through comp_beta=1) is literally HAPPO on the
stationary channel, and its return IS the reachable ceiling.

  * T1 (exact cancellation): simulating ant.py's d-dynamics AND the wrapper's
        cache-then-compensate, the DELIVERED torque equals clip(a) to 1e-12 on the
        unsaturated set — the disturbance is gone.
  * T2 (cache timing / reset): the cached d used to compensate action t equals the
        env's actual d at step t (no off-by-one), and it resets to 0 at an episode
        boundary (ant.py zeroes _d on reset).
  * T3 (β and identity): comp_beta=0 is a bitwise no-op; β<1 leaves exactly
        (1−β)·d of residual disturbance (so the O0 β-grid is interpretable).
  * T4 (saturation leak): when the compensated command rails (|a−d|>1), the
        residual delivered error equals the clipped part, never more than |d|.

Run:  python -m harl.envs.mamujoco.omax.test_omax
Expect: "T1..T4 PASS" and "V0 PASS".
"""

import numpy as np

RHO = 0.8
SIGMA = 0.45
N_ACT = 8


def _ant_step(d, tau_delivered_prev, a_received, A):
    """One ant.py transition, returning (delivered, d_next).

    Mirrors ant.py exactly: tau = clip(a_received); delivered = clip(tau + d);
    then d_next = rho*d + (1-rho)*A*sigma*s, s_i = Σ_{j≠i} tau_j per joint type.
    """
    tau = np.clip(a_received, -1.0, 1.0)
    delivered = np.clip(tau + d, -1.0, 1.0)
    hip, ank = tau[0::2], tau[1::2]
    s = np.empty_like(tau)
    s[0::2] = hip.sum() - hip
    s[1::2] = ank.sum() - ank
    d_next = RHO * d + (1.0 - RHO) * (A * SIGMA * s)
    return delivered, d_next


def _run(a_seq, beta, dones=None, amp_ok=True):
    """Full loop: the wrapper caches d and compensates u = a − β·d_cache, ant
    applies it. Returns (delivered_seq, a_clip_seq, d_at_step_seq)."""
    T = a_seq.shape[0]
    dones = np.zeros(T) if dones is None else dones
    d_env = np.zeros(N_ACT)          # ant's true d at the CURRENT step
    d_cache = np.zeros(N_ACT)        # wrapper's cache (= pcr_d_next from last step)
    delivered_seq, aclip_seq, d_step_seq = [], [], []
    A = 1.0                          # frozen peak payload (hardest); c = A·σ
    for t in range(T):
        # wrapper: compensate the policy's action with the cached d
        u = a_seq[t] - beta * d_cache
        d_step_seq.append(d_env.copy())
        delivered, d_next = _ant_step(d_env, None, u, A)
        delivered_seq.append(delivered)
        aclip_seq.append(np.clip(a_seq[t], -1.0, 1.0))
        # wrapper caches pcr_d_next for the NEXT action
        d_cache = d_next.copy()
        d_env = d_next.copy()
        if dones[t] > 0.5:           # episode boundary: ant zeroes _d, wrapper too
            d_env = np.zeros(N_ACT)
            d_cache = np.zeros(N_ACT)
    return np.array(delivered_seq), np.array(aclip_seq), np.array(d_step_seq)


def test_t1_exact_cancellation():
    rng = np.random.default_rng(0)
    # small actions so |a − d| never rails (the unsaturated set A2 holds on)
    a = np.clip(0.25 * rng.standard_normal((400, N_ACT)), -0.3, 0.3)
    delivered, aclip, dstep = _run(a, beta=1.0)
    gap = float(np.max(np.abs(delivered - aclip)))
    live = float(np.abs(dstep).max())        # the loop really was disturbed
    ok = gap < 1e-12 and live > 0.02
    print(f"T1 [cancel]  max|delivered − clip(a)| = {gap:.2e}  "
          f"(|d|max seen = {live:.3f})  -> {'OK' if ok else 'FAIL'}")
    return ok


def test_t2_cache_timing_reset():
    rng = np.random.default_rng(1)
    a = np.clip(0.25 * rng.standard_normal((300, N_ACT)), -0.3, 0.3)
    dones = np.zeros(300)
    dones[[57, 133, 210]] = 1.0
    delivered, aclip, dstep = _run(a, beta=1.0, dones=dones)
    # exact cancellation must survive the resets (no off-by-one at boundaries)
    gap = float(np.max(np.abs(delivered - aclip)))
    # the step right AFTER each reset must see d = 0
    after = [dstep[i + 1] for i in [57, 133, 210]]
    reset_ok = all(float(np.abs(x).max()) < 1e-12 for x in after)
    ok = gap < 1e-12 and reset_ok
    print(f"T2 [timing]  cancel-through-resets gap = {gap:.2e}; d==0 after reset "
          f"= {reset_ok}  -> {'OK' if ok else 'FAIL'}")
    return ok


def test_t3_beta_identity():
    rng = np.random.default_rng(2)
    a = np.clip(0.25 * rng.standard_normal((300, N_ACT)), -0.3, 0.3)
    # comp_beta = 0 is a bitwise no-op: delivered == clip(a + d) (the blind env)
    deliv0, aclip, dstep = _run(a, beta=0.0)
    blind = np.clip(aclip + dstep, -1.0, 1.0)
    identity = float(np.max(np.abs(deliv0 - blind))) < 1e-12
    # beta = 0.5 leaves exactly (1−beta)·d residual on the unsaturated set
    deliv_h, aclip_h, dstep_h = _run(a, beta=0.5)
    resid = deliv_h - aclip_h
    expect = np.clip(aclip_h + 0.5 * dstep_h, -1, 1) - aclip_h  # (1−β)d, clipped
    half_ok = float(np.max(np.abs(resid - expect))) < 1e-12
    ok = identity and half_ok
    print(f"T3 [beta]    β=0 is blind-env identity = {identity}; β=0.5 leaves "
          f"(1−β)d residual = {half_ok}  -> {'OK' if ok else 'FAIL'}")
    return ok


def test_t4_saturation_leak():
    rng = np.random.default_rng(3)
    # large actions so a − d rails: residual must equal the clipped leak, ≤ |d|
    a = np.clip(0.9 * rng.standard_normal((300, N_ACT)), -1.0, 1.0)
    delivered, aclip, dstep = _run(a, beta=1.0)
    resid = np.abs(delivered - aclip)
    bounded = float(np.max(resid - (np.abs(dstep) + 1e-9))) <= 0.0
    railed = float(np.max(resid)) > 1e-6       # some leak actually occurred
    ok = bounded and railed
    print(f"T4 [sat]     max residual = {float(np.max(resid)):.3f}  ≤ |d| everywhere "
          f"= {bounded} (railing happened = {railed})  -> {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [
        test_t1_exact_cancellation(),
        test_t2_cache_timing_reset(),
        test_t3_beta_identity(),
        test_t4_saturation_leak(),
    ]
    print("V0 %s" % ("PASS" if all(results) else "FAIL"))
