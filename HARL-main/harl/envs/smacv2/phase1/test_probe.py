"""Pure-arithmetic unit tests for the SMACv2-CWD Phase-1 probe.

No StarCraft II binary is launched: the discrete re-aim controller is exercised
against a tiny fake inner env (supplying only the availability mask), and its
choices are checked against an independent brute-force argmin.  The driver ramp,
freeze, continuous-shift, and leaky-integrator steady state are checked against
their closed forms.

Run on the run machine (needs the ``smacv2`` python package importable — it is not
instantiated):

    python -m harl.envs.smacv2.phase1.test_probe
"""

import numpy as np

from harl.envs.smacv2.phase1.probe_env import SMACv2ProbeEnv, _DIR, STOP
from harl.envs.smacv2.smacv2_env import (
    _cwd_driver,
    _CWD_P,
    _CWD_RHO,
    _CWD_DCAP,
)

M_DEFAULT = 2.0


class _FakeInner:
    """Minimal stand-in for the underlying StarCraft2Env: just the avail mask."""

    def __init__(self, avail):
        self._avail = avail

    def get_avail_actions(self):
        return self._avail


def _make_probe(n_agents, cwd_d, avail, beta, move_amount=M_DEFAULT, comp_mode="discrete"):
    p = object.__new__(SMACv2ProbeEnv)  # skip __init__ (no args/SC2 needed)
    p.n_agents = n_agents
    p.env = _FakeInner(avail)
    p._cwd_d = np.asarray(cwd_d, dtype=np.float64)
    p._cwd_move_amount = float(move_amount)
    p.comp_beta = float(beta)
    p.comp_mode = comp_mode
    return p


def _brute_reaim(a, di, M, beta, avail_i):
    """Independent argmin: the least-post-shove-residual available discrete action."""
    v_star = M * _DIR[a]
    best, best_cost = a, None
    for cand in (1, 2, 3, 4, 5):
        if cand >= len(avail_i) or avail_i[cand] == 0:
            continue
        delivered = np.zeros(2) if cand == STOP else M * _DIR[cand] + beta * di
        cost = float(np.sum((delivered - v_star) ** 2))
        if best_cost is None or cost < best_cost - 1e-9:
            best_cost, best = cost, cand
    return best


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_beta0_is_identity():
    """beta=0 -> the controller reproduces the blind action for every move."""
    print("test_beta0_is_identity")
    all_avail = [np.ones(10, dtype=int) for _ in range(4)]
    d = np.array([[1.3, -0.7], [-2.0, 2.0], [0.1, 0.9], [-1.1, -1.4]])
    acts = np.array([[2], [3], [4], [5]])
    p = _make_probe(4, d, all_avail, beta=0.0)
    out = np.asarray(p._discrete_reaim(acts)).reshape(-1)
    check("beta=0 leaves every cardinal move unchanged", np.array_equal(out, [2, 3, 4, 5]))


def test_exact_diagonal_cancellation():
    """a=EAST, d=(M,M), beta=1 -> pick SOUTH; delivered == intended (residual 0)."""
    print("test_exact_diagonal_cancellation")
    M = M_DEFAULT
    d = np.array([[M, M]])
    p = _make_probe(1, d, [np.ones(10, dtype=int)], beta=1.0)
    out = int(np.asarray(p._discrete_reaim(np.array([[4]]))).reshape(-1)[0])
    delivered = M * _DIR[out] + 1.0 * d[0]      # SOUTH gives (0,-M)+(M,M)=(M,0)=EAST intent
    check("chose SOUTH (3)", out == 3)
    check("residual ~ 0", np.allclose(delivered, M * _DIR[4], atol=1e-9))


def test_matches_bruteforce_random():
    """Real method == independent brute force over 4000 random cases."""
    print("test_matches_bruteforce_random")
    rng = np.random.default_rng(0)
    mism = 0
    for _ in range(4000):
        a = int(rng.choice([2, 3, 4, 5]))
        di = rng.uniform(-_CWD_DCAP, _CWD_DCAP, size=2)
        beta = float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]))
        avail_i = np.ones(10, dtype=int)
        # randomly mask 0-2 of the candidate moves (never mask the commanded one)
        for cand in rng.choice([1, 2, 3, 4, 5], size=int(rng.integers(0, 3)), replace=False):
            if cand != a:
                avail_i[cand] = 0
        p = _make_probe(1, di[None, :], [avail_i], beta=beta)
        got = int(np.asarray(p._discrete_reaim(np.array([[a]]))).reshape(-1)[0])
        want = _brute_reaim(a, di, M_DEFAULT, beta, avail_i)
        mism += got != want
    check("real re-aim matches brute-force on all 4000 cases", mism == 0)


def test_availability_respected():
    """A masked-out optimum is never chosen."""
    print("test_availability_respected")
    M = M_DEFAULT
    d = np.array([[M, M]])                     # optimum would be SOUTH (3)
    avail = [np.ones(10, dtype=int)]
    avail[0][3] = 0                            # mask SOUTH
    p = _make_probe(1, d, avail, beta=1.0)
    out = int(np.asarray(p._discrete_reaim(np.array([[4]]))).reshape(-1)[0])
    check("did not pick masked SOUTH", out != 3)
    check("picked an available action", avail[0][out] == 1)


def test_non_move_actions_untouched():
    """no-op(0), STOP(1), and attack(>=6) are never rewritten."""
    print("test_non_move_actions_untouched")
    d = np.array([[2.0, 2.0]] * 4)
    avail = [np.ones(12, dtype=int) for _ in range(4)]
    acts = np.array([[0], [1], [6], [9]])
    p = _make_probe(4, d, avail, beta=1.0)
    out = np.asarray(p._discrete_reaim(acts)).reshape(-1)
    check("no-op/stop/attack unchanged", np.array_equal(out, [0, 1, 6, 9]))


def test_continuous_shift():
    """continuous mode delivers (1-beta)*d; other modes deliver full d."""
    print("test_continuous_shift")
    d = np.array([[1.0, 2.0]])
    p = _make_probe(1, d, [np.ones(10, dtype=int)], beta=0.6, comp_mode="continuous")
    check("continuous shift = (1-beta)*d", np.allclose(p._cwd_delivered_shift(0), 0.4 * d[0]))
    p.comp_mode = "discrete"
    check("discrete shift = full d", np.allclose(p._cwd_delivered_shift(0), d[0]))
    p.comp_mode = "none"
    check("none shift = full d", np.allclose(p._cwd_delivered_shift(0), d[0]))


def test_driver_shape_and_freeze():
    """A(t): convex ramp in [0,1], peak 1.0 just before the drop; freeze pins it."""
    print("test_driver_shape_and_freeze")
    c = 0.85
    peak = _cwd_driver(int(c * _CWD_P) - 1)
    trough = _cwd_driver(0)
    mid_ramp = _cwd_driver(int(0.4 * _CWD_P))
    vals = [_cwd_driver(t) for t in range(_CWD_P)]
    check("A in [0,1]", all(-1e-9 <= v <= 1 + 1e-9 for v in vals))
    check("A(0) ~ 0 (trough)", abs(trough) < 1e-6)
    check("A near peak ~ 1", peak > 0.99)
    check("convex early (A(0.4) < 0.4)", mid_ramp < 0.4)  # (0.4/0.85)^2 = 0.22
    # freeze pins the value regardless of the clock
    fp = object.__new__(SMACv2ProbeEnv)
    fp.cwd_freeze = 0.7
    fp._cwd_clock = 12345
    check("freeze pins A", abs(fp._cwd_driver_value() - 0.7) < 1e-9)
    fp.cwd_freeze = None
    fp._cwd_clock = 100
    check("no freeze follows the ramp", abs(fp._cwd_driver_value() - _cwd_driver(100)) < 1e-9)


def test_leak_steady_state():
    """d <- rho*d + (1-rho)*(A*sigma*S) converges to A*sigma*S, then clips at DCAP."""
    print("test_leak_steady_state")
    A, sigma = 1.0, 0.5
    S = np.array([0.8, -0.3])
    d = np.zeros(2)
    for _ in range(200):
        d = _CWD_RHO * d + (1.0 - _CWD_RHO) * (A * sigma * S)
    check("steady state = A*sigma*S", np.allclose(d, A * sigma * S, atol=1e-6))
    # a large drive pins the shove at the per-axis cap
    Sbig = np.array([100.0, -100.0])
    dbig = np.zeros(2)
    for _ in range(200):
        dbig = _CWD_RHO * dbig + (1.0 - _CWD_RHO) * (A * 5.0 * Sbig)
        np.clip(dbig, -_CWD_DCAP, _CWD_DCAP, out=dbig)
    check("clips at DCAP", np.allclose(np.abs(dbig), _CWD_DCAP))


def main():
    tests = [
        test_beta0_is_identity,
        test_exact_diagonal_cancellation,
        test_matches_bruteforce_random,
        test_availability_respected,
        test_non_move_actions_untouched,
        test_continuous_shift,
        test_driver_shape_and_freeze,
        test_leak_steady_state,
    ]
    for t in tests:
        t()
    print("\nAll Phase-1 probe arithmetic tests PASSED.")


if __name__ == "__main__":
    main()
