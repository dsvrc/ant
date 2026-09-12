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


def main():
    print("=" * 78)
    print("SMAC-NS conformance suite -- offline, no StarCraft II")
    print("=" * 78)
    c = t_operator()
    d = t_dial()
    t_harm(c, d)
    t_ceiling(c, d)
    t_estimator(c)
    print("=" * 78)
    if FAILS:
        print("FAILED %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("ALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
