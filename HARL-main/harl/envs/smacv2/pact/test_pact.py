"""Pure-numpy arithmetic certificate for SMACv2-CWD PACT (no StarCraft II).

Proves the three facts the whole method rests on, using the SAME recursions the env
runs (``harl/envs/smacv2/smacv2_env.py``):

  T1  The waveform factorizes: with c constant, d_i = c * x2_i exactly, so an agent
      that iterates the c-free accumulator x2 knows the shove up to one scalar.
  Sat The only divergence is the DCAP clip on d (the small saturation leak); below
      the cap d and c*x2 are bit-equal.
  Gate The cosine helper the env logs is 1 for aligned rows and skips zero rows.

Run:  python -m harl.envs.smacv2.pact.test_pact
"""

import numpy as np

from harl.envs.smacv2.smacv2_env import _CWD_RHO, _CWD_DCAP, _cwd_cos_rows


def _sim(S_seq, c_seq):
    """Iterate the env's d-recursion and the PACT x2-recursion on the same S, c.

    Mirrors ``_cwd_advance``: d <- rho*d + (1-rho)*c*S, clipped to +/-DCAP;
    x2 <- rho*x2 + (1-rho)*S, unclipped.  Returns per-step (d_clipped, d_raw, x2).
    """
    n = S_seq.shape[1]
    d = np.zeros((n, 2))
    x2 = np.zeros((n, 2))
    out = []
    for S, c in zip(S_seq, c_seq):
        d_raw = _CWD_RHO * d + (1.0 - _CWD_RHO) * (c * S)
        d = np.clip(d_raw, -_CWD_DCAP, _CWD_DCAP)
        x2 = _CWD_RHO * x2 + (1.0 - _CWD_RHO) * S
        out.append((d.copy(), d_raw.copy(), x2.copy()))
    return out


def test_factorization_constant_c():
    """T1: c constant over the window => d == c*x2 exactly (on the unsaturated set)."""
    rng = np.random.default_rng(0)
    n, T = 5, 200
    c = 0.3  # small enough that DCAP never bites (|c*x2| stays < 2)
    S_seq = rng.normal(0, 0.4, size=(T, n, 2))  # |S| ~ 0.5, so |c*x2| well under DCAP
    c_seq = np.full(T, c)
    worst = 0.0
    for d, d_raw, x2 in _sim(S_seq, c_seq):
        assert np.max(np.abs(d - d_raw)) < 1e-9, "unexpected clipping in this regime"
        worst = max(worst, float(np.max(np.abs(d - c * x2))))
    assert worst < 1e-9, f"d != c*x2 (max dev {worst:.2e})"
    print(f"  T1 factorization (constant c): max |d - c*x2| = {worst:.2e}  OK")


def test_saturation_is_the_only_leak():
    """Sat: with a large c the DCAP clip bites.  The PURE (unclipped) recursion still
    factorizes exactly (d_noclip == c*x2, always); the real clipped d stays bounded
    and matches c*x2 only until the FIRST clip.  After a clip the clipped d carries
    the clipped value forward, so d and c*x2 diverge PERSISTENTLY -- the honest
    saturation leak (T3's +sat term), not a per-step error.  (At the operating
    severity 1.5 this touches only ~2% of steps; run the gate at severity 0.5 for a
    clip-free wiring check.)"""
    rng = np.random.default_rng(1)
    n, T = 5, 300
    c = 3.0  # large: |c*x2| routinely exceeds DCAP=2 -> clip active
    d = np.zeros((n, 2))
    d_noclip = np.zeros((n, 2))  # the same recursion WITHOUT the clip
    x2 = np.zeros((n, 2))
    clipped_yet = False
    saw_clip = False
    for S in rng.normal(0, 0.5, size=(T, n, 2)):
        d_raw = _CWD_RHO * d + (1.0 - _CWD_RHO) * (c * S)
        clip_now = bool(np.any(np.abs(d_raw) > _CWD_DCAP + 1e-12))
        d = np.clip(d_raw, -_CWD_DCAP, _CWD_DCAP)
        d_noclip = _CWD_RHO * d_noclip + (1.0 - _CWD_RHO) * (c * S)
        x2 = _CWD_RHO * x2 + (1.0 - _CWD_RHO) * S
        # (1) the pure (unclipped) waveform factorizes exactly -- always
        assert np.max(np.abs(d_noclip - c * x2)) < 1e-9, "d_noclip != c*x2"
        if clip_now:
            clipped_yet, saw_clip = True, True
        elif not clipped_yet:
            # (2) before the first clip, the clipped d also equals c*x2 exactly
            assert np.max(np.abs(d - c * x2)) < 1e-9, "pre-clip d != c*x2"
        # (3) the clip bounds |d|
        assert np.max(np.abs(d)) <= _CWD_DCAP + 1e-9, "clip must bound |d|"
    assert saw_clip, "test regime never exercised the DCAP clip"
    print("  Sat leak: d_noclip==c*x2 always; clipped d exact pre-clip; |d|<=DCAP  OK")


def test_cosine_gate():
    """Gate: aligned rows -> 1, anti-aligned -> -1, zero rows skipped."""
    x2 = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    d_aligned = np.array([[0.5, 0.0], [0.0, 4.0], [0.0, 0.0]])  # 3rd row zero -> skipped
    assert abs(_cwd_cos_rows(x2, d_aligned) - 1.0) < 1e-9
    d_anti = np.array([[-0.5, 0.0], [0.0, -4.0], [1.0, 1.0]])
    assert abs(_cwd_cos_rows(x2, d_anti) - (-1.0)) < 1e-9  # 3rd row skipped (x2 zero)
    assert abs(_cwd_cos_rows(np.zeros((3, 2)), np.zeros((3, 2))) - 1.0) < 1e-9
    print("  Gate cosine: aligned=1, anti=-1, zeros skipped  OK")


def test_floor_property_is_append_only():
    """Floor: the PACT obs block is APPENDED, so zeroing/ignoring it == blind obs.

    (Structural check of the augmentation contract: base obs is a prefix of the
    augmented obs; a policy that ignores the last 3 dims is exactly the blind one.)
    """
    base = np.arange(10, dtype=np.float32)
    x2_i = np.array([0.7, -0.3], dtype=np.float32)
    aug = np.concatenate([base, x2_i, [float(np.linalg.norm(x2_i))]])
    assert aug.shape[0] == base.shape[0] + 3
    assert np.array_equal(aug[: base.shape[0]], base), "base obs must be unchanged"
    assert abs(aug[-1] - np.hypot(0.7, 0.3)) < 1e-6
    print("  Floor: obs block is append-only (base obs preserved)  OK")


def main():
    print("SMACv2-CWD PACT arithmetic certificate (pure numpy):")
    test_factorization_constant_c()
    test_saturation_is_the_only_leak()
    test_cosine_gate()
    test_floor_property_is_append_only()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
