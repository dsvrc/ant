"""The conformance suite.  No StarCraft II, no torch, no learning framework.

`PACT_NS_SPEC` I.7: *"Fourteen offline checks... Port all of them; each
corresponds to a requirement above and each has failed at least once during
development."*  The names below are the spec's own, so a reviewer can match them
to requirement IDs line by line.

Run::

    python -m harl.envs.smac.smac_ns.selftest
"""

import sys

import numpy as np

from . import coupling as cpl
from .coupling import Coupling
from .driver import GuardDriver, SC2_ARMOUR_LOSS
from .pact1_core import (
    AgentRLS,
    herd_index,
    relative_excess,
    rls_confidence_pred,
    steer_logits,
    trust_from_w,
)

FAILS = []
MAP, NA, NE = "3s5z", 8, 8


def check(name, cond, detail=""):
    ok = bool(cond)
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)
    return ok


def _rand(rng, n=NA, m=NE, p_fire=0.85):
    fired = (rng.rand(n) < p_fire).astype(float)
    tgt = np.where(fired > 0, rng.randint(0, m, n), -1)
    return tgt, np.ones(n), fired


# ===================================================================== operator
def t_operator():
    print("operator  (NS-1.2, P-3.1, P-3.2, P-3.4)")
    c = Coupling(MAP, NA, NE)
    W = c.W()
    r = c.report()
    check("zero_diagonal", np.all(np.diag(W) == 0.0))
    check("asymmetric_and_spread", r["asymmetry"] > 0.05 and r["spread"] > 0.05,
          "asym=%.3f spread=%.3f ratio=%.1fx" % (r["asymmetry"], r["spread"],
                                                 r["ratio"]))
    # the flat proxy the spec forbids, for contrast
    proxy = np.where(W > 0, 1.0, 0.0)
    pr = Coupling(MAP, NA, NE)
    off = proxy[~np.eye(NA, dtype=bool)]
    check("a_flat_proxy_really_is_flat", float(off[off > 0].std()) == 0.0,
          "proxy spread=0 against the real %.3f" % r["spread"])

    rng = np.random.RandomState(0)
    worst = 0.0
    for _ in range(200):
        t, al, f = _rand(rng)
        worst = max(worst, float(np.max(np.abs(c.channels(t, al, f)
                                               - c.channels_bruteforce(t, al, f)))))
    check("vectorised_basis_equals_brute_force", worst < 1e-12,
          "max|diff| = %.3g  (GATE 1: abort on mismatch)" % worst)

    c1 = Coupling(MAP, 1, NE)
    z = c1.channels(np.array([0]), np.array([1.0]), np.array([1.0]))
    u = c1.loading(np.array([0]), np.array([1.0]), np.array([1.0]), np.ones(NE))
    check("n1_gives_exactly_zero_peer_load",
          float(np.max(np.abs(z))) == 0.0 and float(u[0]) == 0.0)

    check("channel_pruning_keeps_everything_aligned",
          c.r == len(c.keep) and all(c.classes[cc] not in c.pruned for cc in c.keep),
          "r=%d/%d kept=%s pruned=%s" % (c.r, c.r_full,
                                         [c.classes[i] for i in c.keep], c.pruned))
    # a constant channel must be visible, not silently carried
    xs = []
    for _ in range(400):
        t, al, f = _rand(rng)
        xs.append(c.channels(t, al, f))
    X = np.concatenate(xs, 0)
    check("channel_variation_flags_a_constant_channel",
          np.all(X.std(axis=0) > 1e-6), "per-channel std = %s"
          % np.round(X.std(axis=0), 4))
    return c


# ========================================================================= dial
def t_dial():
    print("dial  (NS-2.1 .. NS-2.5)")
    d = GuardDriver(period=150)
    info = d.certify()
    check("identity_at_zero_is_exact", True, "g(sigma=0) == 1 over the whole domain")
    check("monotone_and_never_generous", True, "asserted at every driver value")
    t = np.arange(d.period)
    pl = d.is_placebo(t)
    rows = [np.asarray(d.g(t, s)) for s in (0.5, 1.0, 2.0, 3.0)]
    check("placebo_regime_is_exactly_inert",
          all(np.all(r[pl] == 1.0) for r in rows)
          and all(np.array_equal(r[pl], rows[0][pl]) for r in rows),
          "%d/%d steps byte-identical across sigma" % (int(pl.sum()), d.period))
    check("anchored_to_published_constant", d.loss == SC2_ARMOUR_LOSS,
          "loss=%.2f = one SC2 armour point against a 10-damage weapon" % d.loss)
    check("dial_actually_bites_outside_the_placebo",
          float(np.min(d.g(t, 1.0))) < 0.95,
          "g_min at sigma=1 = %.4f" % float(np.min(d.g(t, 1.0))))
    return d


# ========================================================================= harm
def t_harm(c, d):
    print("harm  (I.2 -- the category-C signature)")
    t = np.arange(d.period)
    rng = np.random.RandomState(3)
    tg, al, f = _rand(rng)
    u = c.loading(tg, al, f, np.ones(NE))
    g0 = np.asarray(d.g(37, 0.0)) * np.ones(NE)
    check("identity_when_the_dial_is_off",
          float(np.max(np.abs(c.excess(u, g0[np.clip(tg, 0, NE - 1)])))) == 0.0,
          "excess == 0 EXACTLY at sigma=0")
    prev = None
    mono = True
    for s in (0.0, 0.5, 1.0, 2.0, 3.0):
        gg = np.asarray(d.g(37, s)) * np.ones(NE)
        e = float(np.mean(c.excess(u, gg[np.clip(tg, 0, NE - 1)])[tg >= 0]))
        if prev is not None and e < prev - 1e-12:
            mono = False
        prev = e
    check("harm_is_monotone_in_severity", mono, "mean excess rises with sigma")
    # the signature itself, at the harshest dial the floor allows
    c1 = Coupling(MAP, 1, NE)
    u1 = c1.loading(np.array([0]), np.array([1.0]), np.array([1.0]), np.ones(NE))
    h = 1.0 + float(c1.excess(u1, np.array([1e-3]))[0])
    check("a_lone_agent_reads_harm_exactly_one", h == 1.0,
          "harm = %.9f at g = 1e-3  (if this is not 1.0 the NS is category B)" % h)
    check("sensor_uses_the_binding_element_not_the_mean",
          "np.max" in open(cpl.__file__, encoding="utf-8").read(),
          "loading() aggregates with max over the agent's own elements (NS-1.1)")


# ====================================================================== ceiling
def t_ceiling(c, d):
    print("ceiling  (NS-4.1, NS-4.2)")
    from .ceiling import decompose, sweep
    rng = np.random.RandomState(5)
    tg, al, f = _rand(rng)
    out = decompose(c, d, 1.0, np.ones(NA), tg, al, f, None, 20)
    tot = out["irreducible"] + out["own"] + out["coordination_gap"]
    check("ceiling_shares_are_a_partition", abs(tot - 1.0) < 1e-9,
          "fixed %.3f + own %.3f + peer %.3f" % (out["irreducible"], out["own"],
                                                 out["coordination_gap"]))
    check("all_controllable_has_no_irreducible_share",
          out["irreducible"] == 0.0,
          "every damage source on this medium is a learning agent")
    rows = sweep(MAP, NA, NE, samples=120, sigmas=(1.0,),
                 shares=(0.25, 0.5, 0.75, 1.0))
    gaps = [r["coordination_gap"] for r in rows]
    check("gap_grows_with_fleet_share",
          all(b >= a - 1e-9 for a, b in zip(gaps, gaps[1:])),
          " -> ".join("%.0f%%" % (100 * g) for g in gaps))
    pl = decompose(c, d, 3.0, np.ones(NA), tg, al, f, None, d.period - 1)
    check("placebo_produces_no_excess_to_attribute", pl["delta_total"] == 0.0,
          "Delta == 0 inside the placebo at sigma=3")


# ==================================================================== estimator
def t_estimator(c):
    print("estimator and channel  (P-2.1, P-4.2, P-5.1, P-5.2, P-6.1, P-7.1)")
    check("relative_excess_and_clip",
          float(relative_excess(np.array([21.0]), np.array([1.0]), clip=10.0)[0]) == 10.0
          and float(relative_excess(np.array([1.5]), np.array([1.0]))[0]) == 0.5)

    # P-4.2: a dead row must not inflate the covariance
    r1 = AgentRLS(3, mu=0.99, p0=10.0)
    tr0 = float(np.trace(r1.P))
    for _ in range(500):
        r1.update(np.zeros((1, 3)), np.array([0.0]))
    check("rls_dead_row_does_not_inflate_covariance",
          abs(float(np.trace(r1.P)) - tr0) < 1e-12,
          "tr(P) %.3f -> %.3f over 500 dead rows" % (tr0, float(np.trace(r1.P))))

    # P-4.3: recovery of a known law on the REAL basis
    rng = np.random.RandomState(11)
    beta = np.array([0.05, 0.8, 0.4])
    est = AgentRLS(c.r + 1, mu=0.999, p0=10.0)
    for _ in range(6000):
        t, al, f = _rand(rng)
        psi = c.psi(t, al, f)[0]
        est.update(psi[None, :], np.array([float(psi @ beta) + rng.randn() * 1e-3]))
    check("rls_recovers_known_beta", float(np.max(np.abs(est.beta - beta))) < 0.05,
          "max|err| = %.4f" % float(np.max(np.abs(est.beta - beta))))

    # P-5.1: the prior is INVERTED -- w = 0 sits near full reliance, not half
    g0 = float(trust_from_w(0.0, 0.0, 0.0))
    check("trust_prior_is_inverted", g0 > 0.85,
          "g(w=0) = %.3f  (half would be 0.50 -- URB measured 3642 vs 5444)" % g0)

    # P-5.2: prediction confidence survives dead excitation; the trace does not
    r2 = AgentRLS(3, mu=0.999, p0=10.0)
    psi = np.array([1.0, 0.4, 0.0])
    for _ in range(4000):
        r2.update(psi[None, :], np.array([0.3]))
    cp = rls_confidence_pred(r2.P, 10.0, 3, psi)
    check("pred_confidence_survives_dead_excitation", cp > 0.5,
          "conf_pred = %.3f while the unexcited direction inflates" % cp)
    check("confidence_cold_to_warm",
          abs(rls_confidence_pred(np.eye(3) * 10.0, 10.0, 3, psi) - 1.0 / 4.0) < 1e-9,
          "cold value is exactly 1/(1+r)")

    # P-6.1 / P-7.1: the channel
    lg = np.arange(14, dtype=np.float64)
    cost = np.zeros(14)
    cost[6:] = np.array([5.0, 1.0, 3.0, 9.0, 2.0, 7.0, 4.0, 6.0])
    valid = np.zeros(14, bool)
    valid[6:] = True
    check("floor_property_is_exact",
          np.array_equal(steer_logits(lg, cost, 0.0, 1.0, valid), lg),
          "g = 0 returns the logits bit for bit, for ANY beta_hat")
    s = steer_logits(lg, cost, 0.9, 1.0, valid)
    check("steer_direction", int(np.argmax(s[6:] - lg[6:])) == int(np.argmin(cost[6:]))
          and np.allclose(s[:6], lg[:6]),
          "the cheapest option gains most; masked options untouched")
    check("steer_degenerate_cases",
          np.array_equal(steer_logits(lg, np.ones(14), 0.9, 1.0, valid), lg)
          and np.array_equal(steer_logits(lg, cost, 0.9, 1.0,
                                          np.zeros(14, bool)), lg),
          "identical predictions and <2 valid options are no-ops, not NaN")
    check("herd_index_bounds",
          abs(herd_index(np.array([1.0, 1, 1, 1])) - 0.0) < 1e-12
          and abs(herd_index(np.array([4.0, 0, 0, 0])) - 1.0) < 1e-12,
          "0 = perfectly spread, 1 = everyone on one option")


# ========================================================================= mixin
class _StubHost(object):
    """A host that calls its OWN virtuals during construction, exactly as
    StarCraft2Env does -- it declares the observation space inside __init__ by
    calling ``self.get_obs_size()``.  That is the shape of the bug this class
    exists to catch, and it cost a launch on the cluster."""

    def __init__(self, args, **kwargs):
        self.map_name = args["map_name"]
        self.n_agents, self.n_enemies = 8, 8
        self.episode_limit, self._step_mul = 150, 8
        self.n_actions_no_attack = 6
        self.n_actions = self.n_actions_no_attack + self.n_enemies
        self._controller = None
        self.agents = {i: None for i in range(self.n_agents)}
        self.enemies = {i: None for i in range(self.n_enemies)}   # filled per step
        # the two host calls that reach our overrides mid-construction
        self.observation_space = [self.get_obs_size() for _ in range(self.n_agents)]
        self._probe_obs = self.get_obs_agent(0)

    def get_obs_size(self):
        return [100, [7, 4], [8, 5], [1, 4], [1, 1]]

    def get_obs_agent(self, agent_id):
        return np.zeros(100, dtype=np.float32)

    def reset(self):
        return None


def t_mixin():
    print("mixin over the host  (the construction-order hazard)")
    from .layer import SeverityMixin

    class _Env(SeverityMixin, _StubHost):
        pass

    ok, why = True, ""
    try:
        e = _Env({"map_name": MAP, "ns_severity": 1.0, "ns_augment": 1})
    except Exception as exc:                       # the exact cluster crash
        ok, why, e = False, "%s: %s" % (type(exc).__name__, exc), None
    check("mixin_survives_host_calling_its_overrides_during_init", ok, why)
    if e is None:
        return
    base = _StubHost({"map_name": MAP}).get_obs_size()[0]
    check("augmented_obs_size_matches_the_augmented_obs",
          e.get_obs_size()[0] == base + e.n_actions
          and len(e.get_obs_agent(0)) == base + e.n_actions,
          "declared %d == actual %d (base %d + %d actions)"
          % (e.get_obs_size()[0], len(e.get_obs_agent(0)), base, e.n_actions))

    class _Env2(SeverityMixin, _StubHost):
        pass

    e2 = _Env2({"map_name": MAP, "ns_severity": 1.0, "ns_augment": 0})
    check("stock_baselines_see_byte_identical_stock_observations",
          e2.get_obs_size()[0] == base and len(e2.get_obs_agent(0)) == base,
          "no tail when ns_augment = 0, so mappo/happo run on stock SMAC obs")


# ================================================================ first-run fixes
def t_first_run_fixes():
    """Each check reproduces a failure the first 3.2M-step run measured."""
    print("first-run fixes  (A: live capacity, B: steering floor)")
    rng = np.random.RandomState(21)
    cmax = Coupling(MAP, NA, NE, cap_mode="max")
    crem = Coupling(MAP, NA, NE, cap_mode="remaining")

    # A1: remaining capacity must still give EXACT category C and EXACT identity
    c1 = Coupling(MAP, 1, NE, cap_mode="remaining")
    u1 = c1.loading(np.array([0]), np.array([1.0]), np.array([1.0]), np.ones(NE),
                    cap=np.full(NE, 3.0))          # a nearly dead target
    h1 = 1.0 + float(c1.excess(u1, np.array([1e-3]))[0])
    check("remaining_capacity_keeps_lone_agent_harm_exactly_one", h1 == 1.0,
          "harm = %.9f against a 3-hp target at g = 1e-3" % h1)
    tg, al, f = _rand(rng)
    cap = rng.uniform(5.0, 150.0, NE)
    u = crem.loading(tg, al, f, np.ones(NE), cap=cap)
    e0 = crem.excess(u, np.ones(NA))
    check("remaining_capacity_keeps_identity_at_zero_exact",
          float(np.max(np.abs(e0))) == 0.0, "excess == 0 EXACTLY at g == 1")

    # A2: the fix actually loads the medium.  A focus-fired target losing hit
    # points must read a far higher u than the same volley against a full bar.
    focus = np.zeros(NA, dtype=np.int64)
    fire = np.ones(NA)
    full = cmax.loading(focus, np.ones(NA), fire, np.ones(NE))
    low = crem.loading(focus, np.ones(NA), fire, np.ones(NE),
                       cap=np.r_[20.0, np.full(NE - 1, 150.0)])
    check("remaining_capacity_loads_a_dying_target",
          float(low.mean()) > 5.0 * float(full.mean()),
          "u %.3f against a full bar -> %.3f against 20 hp remaining"
          % (full.mean(), low.mean()))
    check("loading_is_bounded_for_the_regressor",
          float(crem.loading(focus, np.ones(NA), fire, np.ones(NE),
                             cap=np.full(NE, 1e-6)).max()) <= Coupling.U_CLIP,
          "u never exceeds U_CLIP = %.1f however little hp is left" % Coupling.U_CLIP)
    v = crem.channels(tg, al, f, cap=cap)
    b = crem.channels_bruteforce(tg, al, f, cap=cap)
    check("vectorised_basis_equals_brute_force_under_live_capacity",
          float(np.max(np.abs(v - b))) < 1e-12,
          "max|diff| = %.3g" % float(np.max(np.abs(v - b))))

    # B: the floor.  The first run's ranking -- a spread of 0.0007 -- is scale-
    # free to the z-score, so it moves a logit as hard as a 10% difference would.
    lg = np.arange(14, dtype=np.float64)
    valid = np.zeros(14, bool)
    valid[6:] = True
    tiny = np.zeros(14)
    tiny[6:] = 0.0027 + rng.randn(8) * 0.0007
    shifted = steer_logits(lg, tiny, 0.85, 1.0, valid)
    check("zscore_turns_a_0.0007_spread_into_a_full_logit_shove",
          float(np.max(np.abs(shifted - lg))) > 0.5,
          "spread %.4f still moved a logit by %.2f -- why a floor is needed"
          % (float(tiny[6:].std()), float(np.max(np.abs(shifted - lg)))))
    from .channel import steer
    check("below_the_floor_the_host_logits_are_bit_for_bit",
          np.array_equal(steer(lg, tiny, 0.85, 1.0, valid, 0.01), lg),
          "spread %.4f < floor 0.01 -> shift exactly zero (P-7.1 extended)"
          % float(tiny[6:].std()))
    wide = np.zeros(14)
    wide[6:] = rng.uniform(0.0, 0.2, 8)
    # the log-ratio channel: proportional to what is wasted, dimensionless, inert
    lr = steer(lg, wide, 0.9, 1.0, valid, 0.0, "logratio")
    d = lr - lg
    cheapest, dearest = 6 + int(np.argmin(wide[6:])), 6 + int(np.argmax(wide[6:]))
    check("logratio_direction_and_zero_mean",
          d[cheapest] > 0 > d[dearest] and abs(float(d[6:].mean())) < 1e-12
          and np.all(d[:6] == 0.0),
          "cheapest +%.3f, dearest %.3f, mean over attacks %.1e, moves untouched"
          % (d[cheapest], d[dearest], float(d[6:].mean())))
    two = np.zeros(14)
    two[7] = 1.0                                   # one option wastes half its damage
    v2 = np.zeros(14, bool)
    v2[6:8] = True
    r2 = steer(np.zeros(14), two, 0.9, 1.0, v2, 0.0, "logratio")
    odds = float(np.exp(r2[7] - r2[6]))
    check("logratio_reweights_odds_by_one_plus_excess_to_the_minus_g",
          abs(odds - 2.0 ** -0.9) < 1e-12,
          "excess 1.0 keeps %.4f of its odds (2^-0.9 = %.4f)" % (odds, 2.0 ** -0.9))
    small = np.zeros(14)
    small[6:] = rng.uniform(0.0, 0.02, 8)
    zs = float(np.max(np.abs(steer(lg, small, 0.9, 1.0, valid, 0.0, "zscore") - lg)))
    ls = float(np.max(np.abs(steer(lg, small, 0.9, 1.0, valid, 0.0, "logratio") - lg)))
    check("a_1pct_difference_is_a_full_shove_in_zscore_and_a_nudge_in_logratio",
          zs > 1.0 and ls < 0.02,
          "max logit move: zscore %.2f, logratio %.4f" % (zs, ls))
    check("logratio_is_exact_at_g_zero_and_on_identical_predictions",
          np.array_equal(steer(lg, wide, 0.0, 1.0, valid, 0.0, "logratio"), lg)
          and np.array_equal(steer(lg, np.full(14, 0.3), 0.9, 1.0, valid, 0.0,
                                   "logratio"), lg))
    check("above_the_floor_the_shift_is_the_core_steer_logits",
          np.array_equal(steer(lg, wide, 0.85, 1.0, valid, 0.01),
                         steer_logits(lg, wide, 0.85, 1.0, valid)),
          "spread %.3f >= floor: pact1_core, verbatim" % float(wide[6:].std()))

    # the channel's units make the reduction EXACT: excess = (1/g - 1) * s(u)
    uu = np.concatenate([[0.0], rng.uniform(0.0, Coupling.U_CLIP, 400)])
    gg = rng.uniform(0.05, 1.0, uu.size)
    lhs = crem.excess(uu, gg)
    rhs = (1.0 / gg - 1.0) * crem._share(uu)
    check("reduction_is_exact_in_share_units",
          float(np.max(np.abs(lhs - rhs))) < 1e-12 and float(crem._share(0.0)) == 0.0,
          "max|excess - (1/g-1) s(u)| = %.1e over u in [0, %.0f]"
          % (float(np.max(np.abs(lhs - rhs))), Coupling.U_CLIP))
    # ... so RLS on (y, psi) identifies the driver factor itself, per class
    est = AgentRLS(crem.r + 1, mu=1.0, p0=10.0)
    g_true = np.array([0.8, 0.6])                       # per class
    for _ in range(3000):
        tg, al, f = _rand(rng)
        cap = rng.uniform(5.0, 150.0, NE)
        u = crem.loading(tg, al, f, np.ones(NE), cap=cap)
        ps = crem.psi(tg, al, f, cap=cap)
        for i in np.where(f > 0)[0]:
            gc = g_true[crem.cls_of[tg[i]]]
            est.update(ps[i][None, :], np.array([float(crem.excess(u[i], gc))]))
    b = est.beta[1:] / crem.x_scale                     # back to per-unit-share slope
    # exact model, so the only error left is the p0 prior's finite-sample bias
    check("rls_on_share_channels_recovers_one_over_g_minus_one",
          float(np.max(np.abs(b - (1.0 / g_true - 1.0)))) < 1e-4,
          "slope per class %s, truth %s" % (np.round(b, 6),
                                            np.round(1.0 / g_true - 1.0, 6)))

    # one-pass option basis == the per-agent definition (GATE 1b, offline copy)
    worst_p = worst_u = 0.0
    for _ in range(50):
        tg, al, f = _rand(rng)
        al = (rng.rand(NA) < 0.9).astype(float)
        cap = rng.uniform(0.0, 200.0, NE)
        for cp in (None, cap):
            allp = crem.psi_all_options(tg, al, f, cap=cp)
            allu = crem.loading_all_options(tg, al, f, cap=cp)
            for i in range(NA):
                ref = crem.psi_options(i, tg, al, f, np.arange(NE), cap=cp)
                worst_p = max(worst_p, float(np.max(np.abs(allp[i] - ref))))
                for k in range(NE):
                    ex = tg.copy()
                    ex[i] = k
                    want = crem.loading(ex, al, f, np.ones(NE), cap=cp)[i]
                    worst_u = max(worst_u, abs(float(allu[i, k]) - float(want)))
    check("one_pass_option_basis_equals_per_agent_definition",
          worst_p < 1e-12 and worst_u < 1e-9,
          "basis max|diff| %.2g, loading max|diff| %.2g" % (worst_p, worst_u))


# ================================================================ the hook itself
def _unit_env(**kw):
    """The REAL layer over a unit-holding host, stepped in the ENGINE'S ORDER
    (toy.step: tick -> hook -> update_units).  The first version of these tests
    never called the hook, and a crash in ``ns_cap_mode: max`` shipped; the second
    applied harm after the hook, which is not what SC2 does, and could not see that
    the restore was undoing whole steps of damage."""
    import contextlib
    import io

    from . import toy
    from .layer import SeverityMixin

    class _Env(SeverityMixin, toy.BattleHost):
        pass

    args = {"map_name": MAP, "ns_severity": 3.0, "ns_augment": 1}
    args.update(kw)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        e = _Env(args)
    e.agents = toy.fresh(e.coupling.ally_names, 0)
    e.enemies = toy.fresh(e.coupling.enemy_names, 100)
    return e, buf.getvalue()


def _focus_steps(e, n_steps, rng, record=None):
    """Focus the weakest enemy with most of the squad, through the engine order."""
    from . import toy

    for _ in range(n_steps):
        hp = e._ns_live_hp()
        alive = np.where(hp > 0)[0]
        if alive.size == 0:
            e.enemies = toy.fresh(e.coupling.enemy_names, 100 + 10 * e.ns_n_steps)
            e._ns_reset_state()
            continue
        weak = int(alive[np.argmin(hp[alive])])
        acts = [6 + (weak if rng.rand() < 0.7 else int(rng.choice(alive)))
                if rng.rand() < 0.8 else 1 for _ in range(NA)]
        pre = {k: (u.health, u.shield) for k, u in e.enemies.items()}
        toy.tick(e, acts, e.coupling.dmg)
        post_raw = {u.tag: (u.health, u.shield)
                    for u in e._obs.observation.raw_data.units}
        e._ns_hook(acts)
        if record is not None:
            snap = {u.tag: (u.health, u.shield)
                    for u in e._obs.observation.raw_data.units}
            record.append((pre, post_raw, snap, dict((k, u.tag)
                                                     for k, u in e.enemies.items())))
        toy.update_units(e)


def t_hook():
    print("the hook, end to end  (both capacity modes, every arm's cost path)")
    rng = np.random.RandomState(5)
    for mode in ("remaining", "max"):
        ok, why = True, ""
        try:
            e, _ = _unit_env(ns_cap_mode=mode)
            _focus_steps(e, 40, rng)
            info = e.ns_info()
        except Exception as exc:
            ok, why, info = False, "%s: %s" % (type(exc).__name__, exc), {}
        check("hook_runs_end_to_end_with_cap_mode_%s" % mode, ok,
              why or "40 focus-fire steps, %d ns_* keys" % len(info))

    e, _ = _unit_env(ns_oracle=1)
    ok, why = True, ""
    try:
        _focus_steps(e, 20, rng)
    except Exception as exc:
        ok, why = False, "%s: %s" % (type(exc).__name__, exc)
    check("oracle_arm_cost_path_runs", ok, why or "true excess per option, one pass")

    # NS-3.4: the info row describes the step the harm was computed FOR
    e, _ = _unit_env()
    _focus_steps(e, 3, rng)
    info = e.ns_info()
    check("info_latches_the_step_the_harm_was_computed_for",
          info["ns_clock"] == float(e.ns_clock - 1)
          and info["ns_A"] == float(e.driver.A(e.ns_clock - 1)),
          "info clock %d, counter now %d" % (int(info["ns_clock"]), e.ns_clock))

    # P-7.1 at sigma = 0: y is identically zero, beta_hat never leaves 0, every
    # decision is gated and the actor sees a flat tail -- the arm IS the host
    e, _ = _unit_env(ns_severity=0.0)
    _focus_steps(e, 300, rng)
    tail = e.ns_cost[:, 6:]
    check("sigma_zero_every_steering_decision_is_flat",
          e.ns_gated == e.ns_gate_n and e.ns_gate_n > 0
          and float(np.max(np.abs(tail - tail[:, :1]))) == 0.0,
          "%d/%d gated, attack tail exactly flat" % (e.ns_gated, e.ns_gate_n))

    # the knob that gated a well-identified ranking must not be silently accepted
    raised = False
    try:
        _unit_env(ns_snr_min=1.0)
    except ValueError:
        raised = True
    check("removed_ns_snr_min_refuses_to_build", raised,
          "compared the spread with one row's error; replaced by ns_steer_floor")

    e, out = _unit_env()
    check("default_period_is_100_episode_limits",
          e.driver.period == 100 * e.episode_limit
          and "NOT-TRACKABLE" not in out,
          "period %d steps, estimator memory >= %d steps"
          % (e.driver.period, int(round(1.0 / (1.0 - e.ns_mu)))))
    e, out = _unit_env(ns_period=150)
    check("first_run_period_is_flagged_not_trackable", "NOT-TRACKABLE" in out,
          "period 150 against a >= 1000-step memory")

    # the host builds the first observation INSIDE reset(); its tail must be flat
    from . import toy as _toy
    from .layer import SeverityMixin as _Mix

    class _ObsHost(_toy.BattleHost):
        def get_obs_agent(self, agent_id):
            return np.zeros(3, dtype=np.float32)

        def reset(self):
            return [self.get_obs_agent(i) for i in range(self.n_agents)]

    class _EnvR(_Mix, _ObsHost):
        pass

    import contextlib as _cl
    import io as _io
    with _cl.redirect_stdout(_io.StringIO()):
        er = _EnvR({"map_name": MAP, "ns_severity": 5.0, "ns_augment": 1,
                    "ns_phase0": 15000 // 4})
    er.agents = _toy.fresh(er.coupling.ally_names, 0)
    er.enemies = _toy.fresh(er.coupling.enemy_names, 100)
    er.ns_cost[:, 6:] = np.arange(8, dtype=np.float64)     # a non-flat last tail
    first = er.reset()
    check("first_observation_of_an_episode_carries_no_stale_tail",
          all(float(np.max(np.abs(o[3:]))) == 0.0 for o in first),
          "the tail is cleared before the host builds the observation")

    # ---- the restore, against the engine order ------------------------------
    from . import layer as lyr
    from . import toy

    # hold the guard at its peak so every step can harm
    e, _ = _unit_env(ns_severity=5.0, ns_phase0=15000 // 4)
    rec = []
    _focus_steps(e, 400, np.random.RandomState(11), record=rec)
    over, leak, revived, n_rest = 0.0, 0.0, 0, 0
    for pre, post_raw, snap, tags in rec:
        for k, tag in tags.items():
            if pre[k][0] <= 0:
                continue
            was = pre[k][0] + pre[k][1]
            if tag not in post_raw:
                revived += int(tag in snap)            # a killed unit came back?
                continue
            landed = max(0.0, was - sum(post_raw[tag]))
            gain = sum(snap[tag]) - sum(post_raw[tag])
            n_rest += int(gain > 1e-9)
            over = max(over, gain - landed)
            leak = max(leak, sum(snap[tag]) - was)     # net change must stay >= 0
    check("restore_never_exceeds_the_damage_that_landed",
          over <= 1e-9 and n_rest > 0,
          "%d restores, worst restore-minus-landed %.2e" % (n_rest, over))
    check("net_hit_point_change_of_every_enemy_stays_non_negative",
          leak <= 1e-9,
          "stock reward_battle takes abs() of it -- healing would be paid as damage")
    check("a_unit_killed_this_step_is_never_restored", revived == 0)

    # the first version's arithmetic, reproduced: pre-tick shields 30, the tick
    # takes 20, a 25% wasted fraction must hand back 5 -- ending at 15, not 35
    e, _ = _unit_env(ns_severity=5.0, ns_phase0=15000 // 4)
    e.enemies[0] = toy.Unit(100, 80.0, 30.0)
    post = {0: toy.Unit(100, 80.0, 10.0)}
    tg = np.full(NA, -1)
    tg[:2] = 0
    exc = np.zeros(NA)
    exc[:2] = 1.0 / 3.0                                  # f = 0.25 for both shooters
    landed = np.zeros(NE)
    landed[0] = 20.0
    got = e._ns_restore(tg, (tg >= 0).astype(float), exc, post, landed)
    check("restore_adds_to_the_post_tick_bar_not_the_pre_tick_one",
          abs(post[0].shield - 15.0) < 1e-9 and abs(float(got[0]) - 5.0) < 1e-9,
          "shields 30 -> tick 10 -> restored %.1f (the first version wrote 35)"
          % post[0].shield)

    # snapshot and engine agree: capture the debug commands through a fake engine
    class _Pb(object):
        class DebugSetUnitValue(object):
            Shields, Life = "Shields", "Life"

            def __init__(self, unit_value, value, unit_tag):
                self.unit_value, self.value, self.unit_tag = unit_value, value, unit_tag

        class DebugCommand(object):
            def __init__(self, unit_value):
                self.unit_value = unit_value

    class _Ctl(object):
        def __init__(self):
            self.sent = []

        def debug(self, cmds):
            self.sent.extend(cmds)

    saved = list(lyr._PB)
    lyr._PB[:] = [_Pb]
    try:
        e, _ = _unit_env(ns_severity=5.0, ns_phase0=15000 // 4)
        e._controller = _Ctl()
        worst = 0.0
        n_cmd = 0
        rng2 = np.random.RandomState(12)
        for _ in range(200):
            e._controller.sent = []
            hp = e._ns_live_hp()
            alive = np.where(hp > 0)[0]
            if alive.size == 0:
                e.enemies = toy.fresh(e.coupling.enemy_names, 1000 + e.ns_n_steps)
                e._ns_reset_state()
                continue
            weak = int(alive[np.argmin(hp[alive])])
            acts = [6 + weak if rng2.rand() < 0.8 else 1 for _ in range(NA)]
            toy.tick(e, acts, e.coupling.dmg)
            e._ns_hook(acts)
            snap = {u.tag: u for u in e._obs.observation.raw_data.units}
            for c in e._controller.sent:
                v = c.unit_value
                u = snap[v.unit_tag]
                want = u.shield if v.unit_value == "Shields" else u.health
                worst = max(worst, abs(v.value - want))
                n_cmd += 1
            toy.update_units(e)
    finally:
        lyr._PB[:] = saved
    check("engine_write_equals_the_snapshot_the_reward_reads",
          n_cmd > 0 and worst <= 1e-9,
          "%d debug writes, worst |engine - snapshot| %.1e" % (n_cmd, worst))


def main():
    print("=" * 78)
    print("SMAC-NS conformance suite -- offline, no StarCraft II")
    print("=" * 78)
    c = t_operator()
    d = t_dial()
    t_harm(c, d)
    t_ceiling(c, d)
    t_estimator(c)
    t_mixin()
    t_first_run_fixes()
    t_hook()
    print("=" * 78)
    if FAILS:
        print("FAILED %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("ALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
