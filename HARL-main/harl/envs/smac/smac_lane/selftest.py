"""SMAC-LANE conformance suite -- offline, no StarCraft II, no torch.

The REAL ``LaneMixin`` over a stand-in host whose ``step`` does what
StarCraft2Env.step does with actions: it asks ``get_agent_action`` for every
unit's command, then plays them.  So the mixin's own path -- lane channels and
rotation before the tick, the rotated move command, sensing and RLS after -- is
what runs.

Run::

    python -m harl.envs.smac.smac_lane.selftest
"""

import contextlib
import io
import sys

import numpy as np

from .coupling import MOVE_VEC, LaneCoupling, rotate
from .layer import NS_KEYS, LaneMixin, rotated_target

FAILS = []
N, PERIOD = 3, 25000
PEAK, DRY = PERIOD // 4, PERIOD // 2 + 100
SPEED = 4.13 * 8.0 / 22.4


def check(name, cond, detail=""):
    ok = bool(cond)
    print("  [%s] %-60s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)
    return ok


class _P(object):
    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)


class _U(object):
    def __init__(self, tag, x, y):
        self.tag, self.health, self.pos = tag, 160.0, _P(x, y)


class _Host(object):
    """Enough of StarCraft2Env for the mixin: units with positions, the move
    amount, availability, and a step that routes through get_agent_action."""

    def __init__(self, args, **kwargs):
        self.n_agents, self.n_enemies, self.episode_limit = N, 5, 250
        self._move_amount = 2
        self.agents = {i: _U(i, 10.0 + 1.5 * i, 10.0) for i in range(N)}

    def get_avail_agent_actions(self, a_id):
        return [0, 1] + [1] * 4 + [1] * self.n_enemies

    def get_unit_by_id(self, a_id):
        return self.agents[a_id]

    def get_agent_action(self, a_id, action):          # the stock path
        u = self.agents[a_id]
        if 2 <= int(action) <= 5:
            return ("move",) + rotated_target(u.pos.x, u.pos.y, action, 0.0,
                                              self._move_amount)
        return ("other", int(action))

    def step(self, actions):
        acts = [int(np.asarray(a).reshape(-1)[0]) for a in actions]
        self.cmds = [self.get_agent_action(i, a) for i, a in enumerate(acts)]
        for i, c in enumerate(self.cmds):
            if c[0] == "move":
                u = self.agents[i]
                v = np.array([c[1] - u.pos.x, c[2] - u.pos.y])
                v = v / max(np.linalg.norm(v), 1e-9) * SPEED
                u.pos = _P(u.pos.x + v[0], u.pos.y + v[1])
        return (None, None, None, None, [{} for _ in range(N)], None)

    def reset(self):
        return None


class _Env(LaneMixin, _Host):
    def _ns_move_command(self, unit, tx, ty):
        return ("move", tx, ty)


def env(**kw):
    args = dict(ns_severity=3.0, ns_phase0=PEAK, ns_period=PERIOD)
    args.update(kw)
    with contextlib.redirect_stdout(io.StringIO()):
        return _Env(args)


def squad_orders(rng, t):
    """A loose squad moving together: a shared direction that changes every ~12
    steps, 70% of units follow it, the rest pick a direction or hold."""
    group = [2, 4, 3, 5][(t // 12) % 4]
    return [group if rng.rand() < 0.7 else int(rng.choice([1, 2, 3, 4, 5]))
            for _ in range(N)]


def run(e, steps, seed, stock_twin=None):
    rng = np.random.RandomState(seed)
    rec = []
    for t in range(steps):
        acts = squad_orders(rng, t)
        if stock_twin is not None:
            for i in range(N):
                stock_twin.agents[i].pos = _P(e.agents[i].pos.x, e.agents[i].pos.y)
            stock = [_Host.get_agent_action(stock_twin, i, a) for i, a in enumerate(acts)]
        e.step(acts)
        rec.append((list(e.cmds), stock if stock_twin is not None else None,
                    e.ns_phi.copy(), e.ns_d.copy(), e.ns_moving.copy()))
        # keep the squad together: pull stragglers back
        cx = np.mean([e.agents[i].pos.x for i in range(N)])
        cy = np.mean([e.agents[i].pos.y for i in range(N)])
        for i in range(N):
            p = e.agents[i].pos
            if np.hypot(p.x - cx, p.y - cy) > 4.0:
                e.agents[i].pos = _P(0.7 * p.x + 0.3 * cx, 0.7 * p.y + 0.3 * cy)
    return rec


def same_cmds(a, b):
    return all(x[0] == y[0] and np.allclose(x[1:], y[1:], atol=1e-12)
               for x, y in zip(a, b))


def t_structure():
    print("structure  (P-3.1, P-3.2, the inverse)")
    c = LaneCoupling(N)
    rng = np.random.RandomState(0)
    worst = 0.0
    for _ in range(200):
        pos = rng.uniform(0, 6, (N, 2))
        head = MOVE_VEC[rng.randint(0, 4, N)] * (rng.rand(N) < 0.8)[:, None]
        al = (rng.rand(N) < 0.9).astype(float)
        x, s = c.channels(pos, head, al)
        xb, sb = c.channels_bruteforce(pos, head, al)
        worst = max(worst, float(np.abs(x - xb).max()), float(np.abs(s - sb).max()))
    check("vectorised_lane_channels_equal_brute_force", worst < 1e-12, "%.1e" % worst)
    x, s = c.channels(np.array([[0, 0], [30, 30], [-30, 30]], float),
                      MOVE_VEC[[0, 0, 0]], np.ones(N))
    check("a_unit_with_no_teammate_near_its_lane_reads_nearly_zero",
          float(x[0].sum()) < 0.01, "far teammates load %.4f" % float(x[0].sum()))
    x, s = c.channels(np.array([[0, 0], [0, 1.0], [0, -1.0]], float),
                      np.array([[0, 1.0], [0, 0], [0, 0]]), np.ones(N))
    check("front_teammate_loads_front_behind_teammate_loads_nothing",
          x[0, 0] > 0.5 and x[0, 1] == 0.0 and x[1].sum() == 0.0,
          "front %.3f flank %.3f, idle units read zero" % (x[0, 0], x[0, 1]))
    tx, ty = rotated_target(1.0, 2.0, 2, np.pi / 2, 2.0)
    check("move_orders_rotate_about_the_unit", abs(tx - (-1.0)) < 1e-12
          and abs(ty - 2.0) < 1e-12, "north by +90 deg -> west: (%.1f, %.1f)" % (tx, ty))
    v = MOVE_VEC.copy()
    check("rotation_has_an_exact_inverse",
          np.allclose(rotate(rotate(v, np.full(4, 1.1)), np.full(4, -1.1)), v, atol=1e-12))


def t_identities():
    print("identities through the real step  (NS-2.1, NS-2.5, I.2, II.6, P-7.1)")
    for label, kw in (("sigma_zero", dict(ns_severity=0.0)),
                      ("placebo", dict(ns_phase0=DRY))):
        e = env(**kw)
        twin = _Host({})
        rec = run(e, 400, 1, stock_twin=twin)
        ok = all(same_cmds(cmds, stock) for cmds, stock, _, _, _ in rec)
        check("%s_every_executed_order_is_the_chosen_order" % label, ok,
              "400 steps of squad movement")
    e = env(ns_severity=5.0)
    for i in range(1, N):
        e.agents[i].health = 0.0                           # no living teammate
    e.agents[1].pos = _P(e.agents[0].pos.x, e.agents[0].pos.y + 0.5)   # dead, ahead
    e.step([2, 2, 2])
    check("a_unit_with_no_living_teammate_never_swerves",
          float(np.abs(e.ns_phi).max()) == 0.0 and float(e.ns_d.max()) == 0.0,
          "sigma = 5, a dead teammate half a unit ahead loads nothing")

    e = env(ns_severity=3.0, ns_pact=1, ns_oracle=1)
    twin = _Host({})
    rec = run(e, 400, 2, stock_twin=twin)
    live = sum(int(np.any(d > 0)) for _, _, _, d, _ in rec)
    valve = [bool(np.any(d[mv] > e.ns_corr_clip)) for _, _, _, d, mv in rec]
    ok = all(same_cmds(cmds, stock) for (cmds, stock, _, _, _), v in zip(rec, valve)
             if not v)
    check("oracle_compensation_cancels_the_swerve_exactly_below_the_valve",
          ok and live > 50,
          "%d live steps; executed == chosen on every step the quarter-turn valve "
          "did not bind (it bound on %d)" % (live, int(sum(valve))))

    eb = env(ns_severity=3.0)
    ep = env(ns_severity=3.0, ns_pact=1, ns_g_fixed=0.0)
    rb, rp = run(eb, 300, 3), run(ep, 300, 3)
    check("pactoff_is_identical_to_blind_order_for_order",
          all(same_cmds(a[0], b[0]) for a, b in zip(rb, rp)))


def t_estimation():
    print("estimation and compensation  (P-1.2, P-2.1, II.4, II.6)")
    eb = env(ns_severity=3.0, ns_mu=0.99)
    ep = env(ns_severity=3.0, ns_mu=0.99, ns_pact=1)
    rb, rp = run(eb, 3000, 4), run(ep, 3000, 4)
    info = ep.ns_info()
    check("rls_identifies_the_drifting_gain",
          info["ns_beta_cos"] > 0.99 and info["ns_beta_relerr"] < 0.1,
          "beta_cos %.4f, relerr %.4f" % (info["ns_beta_cos"], info["ns_beta_relerr"]))

    def mean_abs(rec, skip=500):
        v = [np.abs(phi[mv]).mean() for _, _, phi, _, mv in rec[skip:] if mv.any()]
        return float(np.degrees(np.mean(v)))

    b, p = mean_abs(rb), mean_abs(rp)
    check("compensation_removes_most_of_the_executed_swerve",
          p < 0.25 * b and b > 1.0,
          "mean executed swerve: blind %.2f deg, pact %.2f deg (%.1fx less)"
          % (b, p, b / max(p, 1e-9)))
    check("ns_info_emits_exactly_the_declared_keys",
          list(info.keys()) == NS_KEYS, "%d keys" % len(info))
    e = env()
    run(e, 50, 5)
    rows, clock = e.ns_rows_total, e.ns_clock
    e.reset()
    check("reset_clears_the_lane_memory_but_not_the_clock_or_estimator",
          float(np.abs(e.ns_Q).max()) == 0.0 and e.ns_clock == clock
          and e.ns_rows_total == rows)


def main():
    print("=" * 90)
    print("SMAC-LANE conformance suite -- offline, no StarCraft II")
    print("=" * 90)
    t_structure()
    t_identities()
    t_estimation()
    print("=" * 90)
    if FAILS:
        print("FAILED %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("ALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
