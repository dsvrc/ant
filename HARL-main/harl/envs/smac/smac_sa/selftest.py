"""SMAC-SA conformance suite -- offline, no StarCraft II, no torch.

Every check drives the REAL ``SurfaceMixin`` over a unit-holding stand-in host
whose ``step`` reproduces StarCraft2Env's order (tick, then update_units), so the
mixin's own ``step`` -- the slot lottery, then sensing and prediction -- is what
runs.

Run::

    python -m harl.envs.smac.smac_sa.selftest
"""

import contextlib
import io
import sys

import numpy as np

from ..smac_ns import toy
from ..smac_ns.coupling import step_damage
from ..smac_ns.pact1_core import AgentRLS
from .coupling import SlotCoupling
from .driver import FormationDriver, HEX_LINE_LOSS
from .layer import NS_KEYS, STOP, SurfaceMixin

FAILS = []
MAP, NA, NE = "3s5z", 8, 8
PEAK = 15000 // 4                                # the driver's maximum, A = 1
DRY = 15000 // 2 + 10                            # inside the placebo


def check(name, cond, detail=""):
    ok = bool(cond)
    print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)
    return ok


class _Host(toy.BattleHost):
    def step(self, actions):
        self.executed = [int(a) for a in actions]
        toy.tick(self, self.executed, self.toy_ally_dmg, None, None)
        toy.update_units(self)
        return (None, None, None, None, [{} for _ in range(self.n_agents)], None)

    def reset(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]


class _Env(SurfaceMixin, _Host):
    pass


def env(**kw):
    args = dict(map_name=MAP, ns_severity=2.0, ns_augment=1, ns_phase0=PEAK)
    args.update(kw)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        e = _Env(args)
    e.toy_ally_dmg = step_damage(e.coupling.ally_names, 8)
    e.agents = toy.fresh(e.coupling.ally_names, 0)
    e.enemies = toy.fresh(e.coupling.enemy_names, 100)
    return e, buf.getvalue()


def _respawn(e, salt):
    if not any(u.health > 0 for u in e.enemies.values()):
        e.enemies = toy.fresh(e.coupling.enemy_names, 1000 + salt)
        e._ns_reset_state()


def t_operator():
    print("operator  (NS-1.1, NS-1.2, P-3.1, P-3.2)")
    c = SlotCoupling(MAP, NA, NE)
    rng = np.random.RandomState(0)
    worst_u = worst_p = 0.0
    for _ in range(300):
        t = rng.randint(-1, NE, NA)
        al = (rng.rand(NA) < 0.9).astype(float)
        f = (rng.rand(NA) < 0.85).astype(float)
        dd = rng.randint(0, 4, (NA, NE)).astype(float)      # integers: many ties
        for d in (None, dd):
            worst_u = max(worst_u, float(np.max(np.abs(
                c.loading(t, al, f, d) - c.loading_bruteforce(t, al, f, d)))))
            worst_p = max(worst_p, float(np.max(np.abs(
                c.psi(t, al, f, d) - c.psi_bruteforce(t, al, f, d)))))
    check("vectorised_loading_and_basis_equal_brute_force",
          worst_u < 1e-12 and worst_p < 1e-12,
          "loading %.1e, basis %.1e (with and without geometry, ties included)"
          % (worst_u, worst_p))

    # slots go to the nearest: on one crowded target, loading rises with distance
    t = np.full(NA, 3)
    d = np.tile(np.arange(NA, dtype=float)[:, None], (1, NE))
    u = c.loading(t, np.ones(NA), np.ones(NA), d)
    mel = np.where(c.melee)[0]
    check("nearest_melee_attacker_takes_a_slot_and_is_never_loaded",
          float(u[mel[0]]) == 0.0 and np.all(np.diff(u[mel]) > 0.0),
          "zealot loadings by distance %s" % np.round(u[mel], 3))
    d2 = np.zeros((NA, NE))
    u2 = c.loading(t, np.ones(NA), np.ones(NA), d2)
    check("ties_go_to_the_lower_index_deterministically",
          np.allclose(u2, u) and float(u2[mel[0]]) == 0.0)
    W = c.W()
    check("operator_zero_diagonal_and_melee_only",
          np.all(np.diag(W) == 0.0) and np.all(W[~c.melee, :] == 0.0)
          and np.all(W[:, ~c.melee] == 0.0),
          "ranged rows and columns are exactly zero")
    check("capacities_are_published_geometry",
          abs(c.cap0[3, 3] - np.pi * 1.0 / 0.5) < 1e-12
          and abs(c.cap0[3, 0] - np.pi * 1.125 / 0.5) < 1e-12,
          "zealot ring on a Zealot %.3f, on a Stalker %.3f" % (c.cap0[3, 3], c.cap0[3, 0]))
    t = np.full(NA, 2)
    u = c.loading(t, np.ones(NA), np.ones(NA))
    check("ranged_attackers_read_exactly_zero_loading",
          np.all(u[~c.melee] == 0.0) and np.all(u[c.melee] > 0.0),
          "all eight on one target: ranged u = %s" % u[~c.melee])


def t_signature():
    print("signature  (I.2 -- category C, identity at zero, the exact reduction)")
    c = SlotCoupling(MAP, NA, NE)
    # a lone MELEE attacker inside the real fleet: one Zealot on enemy 0, every
    # other unit (melee or not) on other targets.  (roster() falls back to
    # Marines for a one-unit fleet, so a 1-agent coupling is not a melee test.)
    z = int(np.where(c.melee)[0][0])
    tl = np.arange(NA) % NE
    tl[z] = 0
    tl[(tl == 0) & (np.arange(NA) != z)] = 1
    u1 = c.loading(tl, np.ones(NA), np.ones(NA))
    p1 = float(c.p_block(c.excess(u1[z], 1e-3)))
    check("a_lone_melee_attacker_is_never_blocked_at_any_severity",
          bool(c.melee[z]) and float(u1[z]) == 0.0 and p1 == 0.0,
          "Zealot alone on its target: u = %.1f, p_block = %.1f at g = 1e-3"
          % (float(u1[z]), p1))
    rng = np.random.RandomState(1)
    t = rng.randint(0, NE, NA)
    u = c.loading(t, np.ones(NA), np.ones(NA))
    check("identity_at_zero_is_exact",
          float(np.max(np.abs(c.excess(u, np.ones(NA))))) == 0.0,
          "excess == 0 exactly at g == 1")
    uu = np.concatenate([[0.0], rng.uniform(0, 5, 500)])
    gg = rng.uniform(0.05, 1.0, uu.size)
    lhs = c.perf(uu / gg) / c.perf(uu) - 1.0
    rhs = c.excess(uu, gg)
    check("reduction_is_exact_in_share_units",
          float(np.max(np.abs(lhs - rhs) / np.maximum(1.0, np.abs(lhs)))) < 1e-9,
          "f(u/g)/f(u) - 1 == (g^-4 - 1) s(u)")
    # an exact model needs excitation, not a prior: targets drawn from one
    # Stalker and one Zealot so the ring actually fills, and a flat prior
    est = AgentRLS(c.r + 1, mu=1.0, p0=1e6)
    g_true = np.array([0.7, 0.55])                      # per enemy class
    pair = np.array([0, 3])                             # a Stalker and a Zealot
    for _ in range(4000):
        tt = pair[rng.randint(0, 2, NA)]
        al = np.ones(NA)
        f = (tt >= 0).astype(float)
        uu2 = c.loading(tt, al, f)
        ps = c.psi(tt, al, f)
        for i in np.where((f > 0) & c.melee)[0]:
            gc = g_true[c.cls_of[tt[i]]]
            est.update(ps[i][None, :], np.array([float(c.excess(uu2[i], gc))]))
    slope = est.beta[1:] * 1.0 / c.x_scale
    truth = g_true ** -4 - 1.0
    check("rls_recovers_the_driver_factor_per_class",
          float(np.max(np.abs(slope - truth) / truth)) < 1e-4,
          "slope %s vs g^-4 - 1 = %s" % (np.round(slope, 4), np.round(truth, 4)))
    check("one_slot_always_stays_open",
          np.all(c.g_eff(np.full(NE, 1e-3), np.full(NA, 0))[c.melee]
                 >= 1.0 / c.cap0[c.melee, 0] - 1e-15),
          "g_eff >= 1/capacity")


def t_lottery():
    print("the slot lottery  (the harm, through the real step)")
    for sigma, phase in ((0.0, PEAK), (2.0, DRY)):
        e, _ = env(ns_severity=sigma, ns_phase0=phase)
        blocked = 0
        for s in range(400):
            _respawn(e, s)
            alive = [k for k in range(NE) if e.enemies[k].health > 0]
            e.step([6 + alive[0]] * NA)
            blocked += int(e.ns_blocked_mask.sum())
        label = "sigma_zero" if sigma == 0.0 else "placebo"
        check("%s_blocks_nothing_and_executes_the_chosen_actions" % label,
              blocked == 0 and e.executed == [6 + alive[0]] * NA,
              "%d blocked in 400 full-focus steps" % blocked)

    e, _ = env(ns_severity=1.5)
    n_b = n_a = 0
    p_sum = 0.0
    stop_ok, ranged_ok = True, True
    for s in range(3000):
        _respawn(e, s)
        alive = [k for k in range(NE) if e.enemies[k].health > 0]
        acts = [6 + alive[0]] * NA
        e.step(acts)
        m = (e.ns_fired > 0) & e.coupling.melee
        n_b += int(e.ns_blocked_mask[m].sum())
        n_a += int(m.sum())
        p_sum += float(e.ns_p[m].sum())
        stop_ok &= all((e.executed[i] == STOP) == bool(e.ns_blocked_mask[i])
                       for i in range(NA))
        ranged_ok &= not bool(e.ns_blocked_mask[~e.coupling.melee].any())
    se = np.sqrt(max(p_sum, 1.0))
    check("blocked_orders_are_executed_as_stop_and_nothing_else", stop_ok)
    check("ranged_attackers_are_never_blocked", ranged_ok)
    check("block_frequency_matches_its_probability",
          abs(n_b - p_sum) < 4.0 * se and n_b > 0,
          "blocked %d vs expected %.0f (+-%.0f) over %d melee orders"
          % (n_b, p_sum, se, n_a))

    # the chosen action is what the estimator sees; the engine saw STOP
    e, _ = env(ns_severity=3.0)
    alive = [k for k in range(NE) if e.enemies[k].health > 0]
    e.step([6 + alive[0]] * NA)
    check("estimator_reads_the_intended_target_even_when_blocked",
          np.all(e.ns_target == alive[0]) and e.ns_blocked_mask.any(),
          "%d blocked, all targets recorded as enemy %d"
          % (int(e.ns_blocked_mask.sum()), alive[0]))


def t_nearest_in_step():
    print("slot order inside the real step")
    e, _ = env(ns_severity=3.0, ns_slot_order="nearest")
    # zealots lined up at increasing distance from enemy 0
    for rank, i in enumerate(np.where(e.coupling.melee)[0]):
        e.agents[i].pos = toy.Pos(0.0, 1.0 + rank)
    for k in range(NE):
        e.enemies[k].pos = toy.Pos(0.0, 0.0 if k == 0 else 50.0)
    mel = np.where(e.coupling.melee)[0]
    hits = np.zeros(NA)
    for s in range(2000):
        for k in range(NE):
            if e.enemies[k].health <= 0:
                e.enemies[k] = toy.Unit(5000 + s * 10 + k, 100.0, 50.0, e.enemies[k].pos)
        e.step([6] * NA)
        hits += e.ns_blocked_mask
    rates = hits[mel] / 2000.0
    check("blocking_falls_on_the_farthest_attackers",
          rates[0] == 0.0 and np.all(np.diff(rates) >= -0.02) and rates[-1] > 0.5,
          "block rate by distance rank %s" % np.round(rates, 3))


def t_contract():
    print("contract with the runner and the policy")
    e, out = env()
    e.step([1] * NA)
    info = e.ns_info()
    check("ns_info_emits_exactly_the_declared_keys",
          list(info.keys()) == NS_KEYS, "%d keys" % len(info))
    e.ns_cost[:, 6:] = np.arange(NE)
    first = e.reset()
    check("first_observation_of_an_episode_carries_no_stale_tail",
          all(float(np.max(np.abs(o[1:]))) == 0.0 for o in first))
    e0, _ = env(ns_severity=0.0)
    for s in range(300):
        _respawn(e0, s)
        alive = [k for k in range(NE) if e0.enemies[k].health > 0]
        e0.step([6 + alive[0]] * NA)
    tail = e0.ns_cost[:, 6:]
    check("sigma_zero_leaves_the_tail_exactly_flat",
          float(np.max(np.abs(tail - tail[:, :1]))) == 0.0,
          "beta_hat stays 0 when nothing is ever blocked")
    e2, _ = env()
    check("default_period_is_100_episode_limits",
          e2.driver.period == 100 * e2.episode_limit)
    _, out150 = env(ns_period=150)
    check("a_short_period_is_flagged_not_trackable", "NOT-TRACKABLE" in out150)
    d = FormationDriver()
    check("anchor_is_the_hex_line_formation", d.loss == HEX_LINE_LOSS == 1.0 / 3.0,
          "L = 1/3: two of six neighbour positions")


def main():
    print("=" * 84)
    print("SMAC-SA conformance suite -- offline, no StarCraft II")
    print("=" * 84)
    t_operator()
    t_signature()
    t_lottery()
    t_nearest_in_step()
    t_contract()
    print("=" * 84)
    if FAILS:
        print("FAILED %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("ALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
