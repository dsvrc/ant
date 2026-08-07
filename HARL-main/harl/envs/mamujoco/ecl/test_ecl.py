"""ECL v2 Stage-V0 unit tests (pure numpy, NO simulator/torch/mujoco).

Runs in seconds; verifies the *core* v2 logic before a slow HASAC run:

  * T1' identifier: the clip-aware fit (C1) recovers the hidden `c` on a
        RAIL-RIDING stream (|τ| pushed onto ±1 a large fraction of the time) and
        on a TROT stream (near sum-zero hips), where the legacy ratio estimator
        demonstrably fails; and lock_gain HOLDs on a degenerate (no-coupling,
        no-clip) stream.
  * T2' envelope: the range-normalized chart ε̃ (C2.2) is monotone in `c`, pins
        to [0,1] over a cycle, and persists across episode resets; and the
        identifier's per-window fit is deterministic (retag idempotency, C3.1).

Run:  python -m harl.envs.mamujoco.ecl.test_ecl
Expect: "T1' PASS", "T2' PASS".
"""

import numpy as np

from harl.envs.mamujoco.ecl.ecl_identifier import ECLIdentifier
from harl.envs.mamujoco.ecl.envelope_adapter import EnvelopeAdapter

N = 4
RHO = 0.8
OBS_DIM = 27
QVEL = [17, 19, 21, 23]        # this env's hip qvel indices (after the -2 shift)
QVEL_ANK = [18, 20, 22, 24]    # ankle = hip + 1
HIP, ANK = 0, 1


class _MockBuf:
    """Minimal stand-in exposing exactly what the identifier reads."""

    def __init__(self, actions, obs, next_obs, dones, nt, raw_readout):
        self.actions = actions
        self.obs = obs
        self.next_obs = next_obs
        self.dones = dones
        self.n_rollout_threads = nt
        self.buffer_size = actions[0].shape[0]
        self.cur_size = self.buffer_size
        self.idx = 0                      # full buffer -> window = whole thing, in order
        self.raw_readout = raw_readout    # (bsz, N, 2)


def _leak_sum(x, W):
    """x2_i = leak_ρ(Σ_{j≠i} x_j), per (N, W, nt)."""
    S = x.sum(axis=0)[None] - x
    x2 = np.zeros_like(S)
    for k in range(1, W):
        x2[:, k, :] = RHO * x2[:, k - 1, :] + (1.0 - RHO) * S[:, k - 1, :]
    return x2


def _gen_buffer(c, W, nt, rng, rail=False, trot=False, degenerate=False):
    """W steps × nt threads of synthetic PCR data at a fixed severity c, through
    the EXACT env channel delivered = clip(τ + c·x2, ±1)."""
    t = np.arange(W)
    omega = 0.3
    hip = np.zeros((N, W, nt))
    ank = np.zeros((N, W, nt))
    for th in range(nt):
        base = rng.uniform(0, 2 * np.pi)
        for i in range(N):
            phi = base + i * np.pi / 2
            amp = 0.9 if (rail or trot) else 0.35     # high enough that delivered clips
            hip[i, :, th] = amp * np.sin(omega * t + phi) + 0.06 * rng.standard_normal(W)
            ank[i, :, th] = 0.6 * np.sin(omega * t + phi + 0.5) + 0.05 * rng.standard_normal(W)
    # sum-zero projection FIRST (trot: Σ hips ≈ 0 ⇒ S_i ≈ −τ_i), then rail-clip — so
    # the censoring survives (subtracting the mean after clipping would un-clip it).
    if trot:
        hip = hip - hip.mean(axis=0, keepdims=True) + 0.03 * rng.standard_normal(hip.shape)
        ank = ank - ank.mean(axis=0, keepdims=True) + 0.03 * rng.standard_normal(ank.shape)
    if rail or trot:
        hip = np.clip(1.4 * hip, -1.0, 1.0)          # ride the rails a large fraction of steps
        ank = np.clip(1.4 * ank, -1.0, 1.0)
    if degenerate:
        # fully-collinear, no-clip stream: identical slow legs ⇒ x2_i ≈ 3·x1_i,
        # so z(c) ∝ x1 for every c ⇒ SSE flat in c ⇒ c UNIDENTIFIABLE (must HOLD).
        slow = (0.2 * np.sin(0.01 * t))[None, :, None]
        hip = np.broadcast_to(slow, (N, W, nt)).copy()
        ank = np.broadcast_to(slow, (N, W, nt)).copy()

    x2h = _leak_sum(hip, W)
    x2a = _leak_sum(ank, W)
    # wandering, sign-flipping plant gain g (ratiometric-invariance stressor)
    y_h = np.zeros((N, W, nt))
    y_a = np.zeros((N, W, nt))
    for i in range(N):
        gh = (1.0 + 0.3 * np.sin(0.01 * t + i))[:, None]
        if i % 2 == 1:
            gh = -gh
        dh = np.clip(hip[i] + c * x2h[i], -1.0, 1.0)
        da = np.clip(ank[i] + c * x2a[i], -1.0, 1.0)
        y_h[i] = gh * dh + 0.02 * rng.standard_normal((W, nt))
        y_a[i] = gh * da + 0.02 * rng.standard_normal((W, nt))

    bsz = W * nt
    actions = [np.zeros((bsz, 2)) for _ in range(N)]
    obs = [np.zeros((bsz, OBS_DIM)) for _ in range(N)]
    nxt = [np.zeros((bsz, OBS_DIM)) for _ in range(N)]
    raw = np.zeros((bsz, N, 2))
    for i in range(N):
        actions[i][:, HIP] = hip[i].reshape(-1)
        actions[i][:, ANK] = ank[i].reshape(-1)
        raw[:, i, 0] = y_h[i].reshape(-1)
        raw[:, i, 1] = y_a[i].reshape(-1)
        nxt[i][:, QVEL[i]] = y_h[i].reshape(-1)          # let the scan find offset 0
        nxt[i][:, QVEL_ANK[i]] = y_a[i].reshape(-1)
    dones = np.zeros((bsz, 1))
    return _MockBuf(actions, obs, nxt, dones, nt, raw)


def _cfg(W):
    return {"W": W, "rho": RHO, "smooth_windows": 0.01, "c_max_init": 0.01,
            "readout_qvel_idx": QVEL, "readout_qvel_idx_ankle": QVEL_ANK,
            "use_ankle_channel": True, "hip_action_idx": HIP, "ankle_action_idx": ANK,
            "identifier_mode": "clipfit", "grid_dc": 0.025, "c_grid_max": 1.2,
            "x2_margin": 25, "c_clip": 1.5, "lock_min_gain": 0.01}


def _sweep(regime, W=600, nt=8):
    rng = np.random.default_rng(0)
    c_true = np.linspace(0.0, 0.9, 10)
    c_clip, c_ratio = [], []
    for c in c_true:
        idf = ECLIdentifier(N, _cfg(W))
        buf = _gen_buffer(c, W, nt, rng, **regime)
        c_clip.append(idf.refresh(buf))
        c_ratio.append(idf.c_now_ratio)
    return c_true, np.array(c_clip), np.array(c_ratio)


def test_identifier():
    ok = True
    for name, regime in [("rail-riding", {"rail": True}), ("trot", {"trot": True})]:
        c_true, c_hat, c_rat = _sweep(regime)
        corr_c = np.corrcoef(c_true, c_hat)[0, 1]
        bias = np.max(np.abs(c_hat - c_true)) / 0.9
        corr_r = np.corrcoef(c_true, c_rat)[0, 1]
        scale_r = (c_rat[-1] / max(c_true[-1], 1e-9))
        clip_pass = (corr_c > 0.95) and (bias < 0.05)
        ratio_fails = (corr_r < 0.5) or (scale_r < 0.33) or (scale_r > 3.0)
        print(f"T1' [{name}]  clipfit corr={corr_c:.3f} bias={bias:.3f} | "
              f"ratio corr={corr_r:.3f} scale={scale_r:.2f}  "
              f"-> clipfit {'OK' if clip_pass else 'FAIL'}, "
              f"ratio {'fails(expected)' if ratio_fails else 'DID NOT FAIL'}")
        print("   c_true:", np.round(c_true, 2))
        print("   c_clip:", np.round(c_hat, 3))
        ok = ok and clip_pass and ratio_fails

    # lock_gain HOLD on a degenerate (no-coupling, no-clip) stream
    idf = ECLIdentifier(N, _cfg(600))
    buf = _gen_buffer(0.5, 600, 8, np.random.default_rng(1), degenerate=True)
    idf.refresh(buf)
    hold = idf.lock_gain < idf.lock_min_gain and idf.c_now == 0.0
    print(f"T1' [degenerate] lock_gain={idf.lock_gain:.4f} c_now={idf.c_now:.4f} "
          f"-> {'HOLD(OK)' if hold else 'FAIL'}")
    ok = ok and hold
    print("   -> T1' %s\n" % ("PASS" if ok else "FAIL"))
    return ok


def test_envelope():
    T = 60000
    rng = np.random.default_rng(1)
    env = EnvelopeAdapter(N, {"gain_halflife": 200, "power_halflife": 600,
                              "center_halflife": 200, "range_norm": True,
                              "q_fast_hl": 50, "q_slow_hl": 8000})
    t = np.arange(T)
    ph = (t % 20000) / 20000.0
    x = np.where(ph < 0.2, ph / 0.2, (1 - ph) / 0.8)
    c_traj = (x * x * (3 - 2 * x)) * 1.2
    omega = 0.3
    ell = np.zeros(N)
    eps_hist = np.zeros(T)
    phase = rng.uniform(0, 2 * np.pi)
    for k in range(T):
        hip = 0.2 * np.sin(omega * k + phase + np.arange(N) * np.pi / 2) + 0.06 * rng.standard_normal(N)
        S = hip.sum() - hip
        ell = RHO * ell + (1 - RHO) * (c_traj[k] * S)
        y = (1.0 + 0.3 * np.sin(0.01 * k + np.arange(N)))
        y[1::2] *= -1
        y = y * np.clip(hip + ell, -1, 1) + 0.02 * rng.standard_normal(N)
        eps_hist[k] = env.update(hip, y).mean()
        if k % 1000 == 999:
            env.reset_episode()                      # persistence across resets
    warm = T // 2
    corr = np.corrcoef(c_traj[warm:], eps_hist[warm:])[0, 1]
    in_range = (eps_hist[warm:].min() >= -1e-6) and (eps_hist[warm:].max() <= 1.0 + 1e-6)
    spans = (eps_hist[warm:].max() - eps_hist[warm:].min()) > 0.4
    ok = corr > 0.6 and in_range and spans

    # retag idempotency: same window -> same clipfit estimate
    idf = ECLIdentifier(N, _cfg(600))
    buf = _gen_buffer(0.4, 600, 8, np.random.default_rng(3), rail=True)
    a, _ = idf.fit_window(buf, 0, 500)
    b, _ = idf.fit_window(buf, 0, 500)
    idempotent = abs(a - b) < 1e-9
    print(f"T2' envelope corr(ε̃,c)={corr:.3f} range∈[0,1]={in_range} "
          f"span>{0.4}={spans}; retag idempotent={idempotent} ({a:.4f}=={b:.4f})")
    ok = ok and idempotent
    print("   -> T2' %s\n" % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    print("=== ECL v2 Stage-V0 sanity (pure numpy, no simulator) ===\n")
    ok1 = test_identifier()
    ok2 = test_envelope()
    print("=== %s ===" % ("ALL PASS — core v2 logic works, safe to launch smoke run"
                          if (ok1 and ok2) else "FAILURE — fix before training"))
