"""PART C -- the ceiling decomposition.  No policy, no training, no simulator.

`PACT_NS_SPEC` I.5: *"This is the highest-value hour in the whole project... Do it
before writing method code; committing the output first is what makes it a
prediction rather than a post-hoc explanation."*

Under the dial each agent's loading exceeds its sigma=0 counterfactual by
``Delta_i = u_i * (1 - g)``, and every unit of that excess traces to a contributor
that partitions by WHO CAN MOVE IT:

    Delta_fixed   load from participants no agent controls
    Delta_own     the agent's own unit, amplified by 1/g
    Delta_peer    other controllable agents, through W, by 1/g   <- the claim

Excess is proportional to load, so a class's share of the load IS its share of the
excess -- the attribution is exact rather than heuristic, and it is computed by
re-running the loading function with one class of contributor at a time.

WHAT SMAC's NUMBERS LOOK LIKE, AND WHY -- READ THIS BEFORE QUOTING THEM
--------------------------------------------------------------------------------
On a fully-controllable SMAC squad the decomposition is degenerate in a way URB's
is not, and pretending otherwise would be dishonest:

  * ``Delta_fixed = 0``.  Every unit pouring damage into the enemy line is a
    learning agent.  There is no background human traffic on this medium, so
    there is no irreducible share.  URB's 42.6% comes from exactly that and SMAC
    has no counterpart.
  * ``Delta_own = 0`` BY CONSTRUCTION of the sensor.  Overkill is damage wasted
    because *someone else* was already killing the target; a lone agent wastes
    nothing.  That is the same ``j != i`` that buys the category-C signature, so
    the zero is structural, not a measurement.

which leaves ``Delta_peer = 100%``.  That is a real property of this medium, not a
flattering choice -- but a 100% coordination gap is a much weaker claim than a
measured 40.8%, because nothing was at risk of coming out small.  So the number
that carries information here is the N-SCALING one (NS-4.2): hold the squad fixed
and hand a growing fraction of it to a non-learning controller, and watch the gap
fall as the irreducible share appears.  That is a prediction no competing
credit-assignment method makes, and it costs no training to test.
"""

import numpy as np

from .coupling import Coupling
from .driver import GuardDriver


def decompose(coupling, driver, sigma, controllable, targets, alive, fired,
              sens=None, t=None):
    """The four numbers of I.5, for one observed state.

    ``controllable`` is a 0/1 mask over agents: 1 = a learning agent whose
    contribution another agent could in principle coordinate with, 0 = a
    participant nobody controls, whose load is irreducible.
    """
    n = coupling.n
    ctrl = np.asarray(controllable, dtype=np.float64).reshape(n)
    alive = np.asarray(alive, dtype=np.float64).reshape(n)
    fired = np.asarray(fired, dtype=np.float64).reshape(n)
    t = 0 if t is None else int(t)
    # `sens=None` gives a scalar multiplier; broadcast it to one per element so
    # the per-element and uniform paths are handled identically downstream.
    g = np.broadcast_to(
        np.asarray(driver.g(t, sigma, sens), dtype=np.float64).reshape(-1),
        (coupling.m,)).copy()

    # total, then one contributor class at a time -- the attribution is exact
    one = np.ones(coupling.m)
    u_tot = coupling.loading(targets, alive, fired, one)
    u_fix = coupling.loading(targets, alive, fired * (1.0 - ctrl), one)
    u_peer = coupling.loading(targets, alive, fired * ctrl, one)

    # Delta is the harm RATIO minus one, evaluated on nominal loading -- the same
    # quantity the layer applies, so the ceiling and the environment agree.
    g_i = np.array([g[t] if t >= 0 else 1.0 for t in np.asarray(targets)])
    d_tot = coupling.excess(u_tot, g_i)
    keep = (np.asarray(targets) >= 0) & (alive > 0)
    if not np.any(keep) or float(np.sum(u_tot[keep])) <= 0.0:
        nan = float("nan")
        return dict(irreducible=nan, own=0.0, coordination_gap=nan,
                    decentralized_ceiling=nan, delta_total=0.0, sigma=float(sigma))
    tot = float(np.sum(u_tot[keep]))
    fix = float(np.sum(u_fix[keep]))
    peer = float(np.sum(u_peer[keep]))
    # the two classes partition the load exactly (the loading is linear in it)
    s = max(1e-12, fix + peer)
    return dict(
        irreducible=fix / s,
        own=0.0,                     # structural: the sensor excludes self (j != i)
        coordination_gap=peer / s,
        decentralized_ceiling=1.0 - fix / s,
        non_coordinating_ceiling=1.0 - fix / s - peer / s,
        delta_total=float(np.mean(d_tot[keep])),
        sigma=float(sigma),
    )


def sweep(map_name="3s5z", n_agents=8, n_enemies=8, step_mul=8, alpha=2.28,
          sigmas=(0.5, 1.0, 2.0, 3.0), shares=(0.25, 0.5, 0.75, 1.0),
          samples=400, seed=0, period=15000):
    """The I.5 table, plus NS-4.2's gap-versus-controllable-share curve.

    ``period`` only sets where the clock is sampled (uniformly over one cycle), so
    the table is the same for any period; it defaults to the layer's 100 episode
    limits so nothing reads as if the old 150-step cycle were still in use."""
    c = Coupling(map_name, n_agents, n_enemies, step_mul, alpha)
    d = GuardDriver(period=period)
    rng = np.random.RandomState(seed)
    sens = np.clip(1.0 + 0.5 * rng.randn(n_enemies), 0.4, 2.0)
    sens /= sens.mean()

    rows = []
    for sh in shares:
        k = max(1, int(round(sh * n_agents)))
        for s in sigmas:
            acc = []
            for _ in range(samples):
                ctrl = np.zeros(n_agents)
                ctrl[rng.choice(n_agents, k, replace=False)] = 1.0
                tgt = rng.randint(0, n_enemies, n_agents)
                fired = (rng.rand(n_agents) < 0.85).astype(float)
                tgt = np.where(fired > 0, tgt, -1)
                alive = np.ones(n_agents)
                tt = int(rng.randint(0, period))
                acc.append(decompose(c, d, s, ctrl, tgt, alive, fired, sens, tt))
            def m(key):
                v = np.array([a[key] for a in acc], dtype=float)
                v = v[np.isfinite(v)]
                return float(v.mean()) if v.size else float("nan")
            rows.append(dict(share=float(k) / n_agents, n_controllable=k,
                             sigma=float(s), irreducible=m("irreducible"),
                             own=0.0, coordination_gap=m("coordination_gap"),
                             delta_total=m("delta_total")))
    return rows


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="3s5z")
    ap.add_argument("--n_agents", type=int, default=8)
    ap.add_argument("--n_enemies", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=2.28)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    rows = sweep(a.map, a.n_agents, a.n_enemies, alpha=a.alpha)
    print("PART C -- the ceiling, computed BEFORE any method code (I.5 / NS-4.1)")
    print("  %-8s %-6s %-13s %-11s %-16s %s"
          % ("share", "sigma", "irreducible", "own(free)", "PEER(coord)", "Delta_tot"))
    for r in rows:
        print("  %-8.2f %-6.2f %-13s %-11s %-16s %.4f"
              % (r["share"], r["sigma"], "%.1f%%" % (100 * r["irreducible"]),
                 "0.0%", "%.1f%%" % (100 * r["coordination_gap"]),
                 r["delta_total"]))
    full = [r for r in rows if r["share"] == 1.0]
    quarter = [r for r in rows if abs(r["share"] - 0.25) < 1e-9]
    if full and quarter:
        print()
        print("  NS-4.2 N-scaling: gap %.1f%% at 25%% controllable -> %.1f%% at 100%%."
              % (100 * np.mean([r["coordination_gap"] for r in quarter]),
                 100 * np.mean([r["coordination_gap"] for r in full])))
        print("  It RISES with controllable share because irreducible load")
        print("  disappears while peer load does not -- the prediction, testable")
        print("  in minutes, that no credit-assignment baseline makes.")
    print()
    print("  NOTE: `own` is structurally 0 (the sensor excludes j == i) and at 100%")
    print("  controllable `irreducible` is 0 too (every damage source is an agent).")
    print("  Quote the N-scaling CURVE, not the 100% endpoint -- see the module")
    print("  docstring for why the endpoint carries no information.")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(dict(map=a.map, alpha=a.alpha, rows=rows), f, indent=2,
                      default=float)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
