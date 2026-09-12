"""Part C -- the ceiling decomposition and the loading distribution, with NO
training and NO simulator.

`PACT_NS_SPEC` I.5 / NS-4.1: *"Compute and commit the decomposition before any
method code exists.  If the coordination gap is small, this environment is a
poor showcase; say so and pick another."*

WHAT THE PARTITION SAYS HERE, HONESTLY
--------------------------------------------------------------------------------
Under the dial each player's swerve exceeds its sigma = 0 counterfactual by
``Delta_i = d_i``.  Every unit of that traces to a contributor:

    Delta_fixed   load from participants no agent controls   -- 0 %.  Only the
                  controlled teammates load a lane; the opposition and the GK are
                  not in the operator (they are not part of the squad's own
                  traffic, and counting them would make a lone controlled player
                  feel the dial: category B in disguise).
    Delta_own     the agent's own unit                       -- 0 %, by j != i.
    Delta_peer    the other controllable agents              -- 100 %.

So the coordination gap is 100 % BY CONSTRUCTION and the endpoint carries no
information -- the same caveat smac_ns states.  What IS informative, and what
this script reports, is:

  * the LOADING DISTRIBUTION (III.1 Q7): is the lane loaded enough for the dial
    to express itself?  A medium far below its limit cannot express a capacity
    loss however large sigma grows.
  * the N-SCALING of peer load: how the swerve at fixed sigma grows with the
    number of teammates in the half -- NS-4.2's prediction, testable in seconds.
  * the reference numbers to COMMIT: load_norm, ref, scale.

Run::

    python -m harl.envs.football.grf_ns.ceiling --out ceiling.json
"""

import argparse
import itertools
import json

import numpy as np

from .actions import N_DIRS
from .coupling import Coupling
from .driver import DialParams, PitchDriver

#: The scenario file's own spawn coordinates for the three controlled attackers
#: (gfootball/scenarios/academy_3_vs_1_with_keeper.py; the GK at (-1, 0) is not
#: controlled).  Declared structure.  ``smoke.py`` cross-checks these against the
#: engine's first reset and flags any drift.
SPAWN = {
    "academy_3_vs_1_with_keeper": np.array([[0.60, 0.00], [0.70, 0.20], [0.70, -0.20]]),
}


def spawn_for(env_name, n):
    if env_name in SPAWN and SPAWN[env_name].shape[0] == n:
        return SPAWN[env_name]
    # fall back to a declared spread across the attacking third
    xs = np.linspace(0.55, 0.75, n)
    ys = np.linspace(-0.25, 0.25, n)
    return np.column_stack([xs, ys])


def peer_load_by_n(p, driver, pos_full, sigma=1.0, samples=2048, seed=0):
    """Mean swerve d at the driver's PEAK for every subset size N of the squad,
    at the spawn geometry with uniformly random headings.  Normalised by the
    FULL squad's load_norm so the rows are comparable."""
    n_full = pos_full.shape[0]
    full = Coupling(n_full, p, driver.send)
    ln = full.load_norm(pos_full[None])
    amp_peak = float(sigma) * p.loss_at_sigma1 * 1.0
    out = []
    rng = np.random.RandomState(seed)
    for n in range(1, n_full + 1):
        vals = []
        for idx in itertools.combinations(range(n_full), n):
            pos = pos_full[list(idx)]
            c = Coupling(n, p, driver.send)
            for _ in range(max(1, samples // 8)):
                head = rng.randint(0, N_DIRS, n)
                x, _ = c.channels(pos, head)
                vals.append((x @ (amp_peak * driver.send)) / ln)
        v = np.concatenate([np.atleast_1d(a) for a in vals])
        out.append(dict(N=n, mean_d_peak=float(v.mean()), max_d_peak=float(v.max()),
                        exactly_zero_frac=float(np.mean(v == 0.0))))
    return out, ln


def loading_distribution(p, driver, pos_ref, ln, sigma=1.0, samples=4096, seed=1):
    """d at the driver's peak over the jittered reference: min / quartiles / max."""
    n = pos_ref.shape[0]
    c = Coupling(n, p, driver.send)
    rng = np.random.RandomState(seed)
    amp_peak = float(sigma) * p.loss_at_sigma1
    vals = []
    for _ in range(samples):
        pos = pos_ref + rng.randn(n, 2) * 0.1
        head = rng.randint(0, N_DIRS, n)
        recv = np.ones(n)
        recv[rng.randint(0, n)] = p.recv_ball
        x, _ = c.channels(pos, head, recv)
        vals.append((x @ (amp_peak * driver.send)) / ln)
    v = np.concatenate(vals)
    q = np.percentile(v, [0, 25, 50, 75, 100])
    return dict(sigma=float(sigma), min=float(q[0]), q25=float(q[1]), median=float(q[2]),
                q75=float(q[3]), max=float(q[4]), mean=float(v.mean()),
                frac_above_half_step=float(np.mean(v > 0.5)),
                frac_above_corr_clip=float(np.mean(v > p.corr_clip)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_name", default="academy_3_vs_1_with_keeper")
    ap.add_argument("--n_agents", type=int, default=3)
    ap.add_argument("--out", default="")
    ap.add_argument("--sigmas", default="0.5,1,2,3")
    a = ap.parse_args()

    p = DialParams()
    driver = PitchDriver(p)
    driver.certify()
    pos = spawn_for(a.env_name, a.n_agents)
    n = pos.shape[0]
    c = Coupling(n, p, driver.send)
    print(c.verify(pos[None]))

    d = np.sqrt(((pos[None] - pos[:, None]) ** 2).sum(-1))
    print("\nspawn geometry (%s):" % a.env_name)
    for i in range(n):
        print("  agent %d at (%.2f, %.2f)" % (i, pos[i, 0], pos[i, 1]))
    print("  pairwise distances: %s  (lane length scale lambda=%.2f)"
          % (np.round(d[~np.eye(n, dtype=bool)], 3).tolist(), p.kernel_lambda))

    ref, scale = c.geometric_reference(pos[None])
    ln = c.load_norm(pos[None])
    print("\nREFERENCES TO COMMIT (football.yaml: ns_load_norm):")
    print("  load_norm = %.6f     ref = %s     scale = %s"
          % (ln, np.round(ref, 4).tolist(), np.round(scale, 4).tolist()))

    recv0 = np.ones(n)
    recv0[0] = p.recv_ball                      # the ball starts at agent 0's feet
    st = c.operator_stats(pos, np.full(n, 4), recv0)
    print("\noperator W at spawn, all heading right, agent 0 on the ball: zero_diag=%s spread=%.3f ratio=%.1fx "
          "asym=%.3f (NS-1.2 wants zero diagonal, spread, asymmetry)"
          % (st["diag_max"] == 0.0, st["spread"], st["ratio"], st["asymmetry"]))

    print("\nPART C -- the partition (I.5).  By construction on this instance:")
    print("  Delta_fixed = 0 %   (no uncontrolled participant loads a lane)")
    print("  Delta_own   = 0 %   (j != i)")
    print("  Delta_peer  = 100 % -> coordination gap 100 %.  THE ENDPOINT CARRIES NO")
    print("  INFORMATION; quote the N-scaling curve below, not the endpoint.")

    rows, _ = peer_load_by_n(p, driver, pos)
    print("\nN-scaling of the peer load at sigma=1, driver peak (spawn geometry, random headings):")
    print("  %3s  %12s  %12s  %14s" % ("N", "mean d", "max d", "exactly-zero"))
    for r in rows:
        print("  %3d  %12.4f  %12.4f  %13.1f%%" % (r["N"], r["mean_d_peak"], r["max_d_peak"],
                                                  100 * r["exactly_zero_frac"]))
    assert rows[0]["max_d_peak"] == 0.0, "a lone player must read EXACTLY zero (gate 3)"

    print("\nloading distribution of d at the driver peak (jittered reference, one ball carrier):")
    dist = []
    for s in [float(x) for x in a.sigmas.split(",")]:
        r = loading_distribution(p, driver, pos, ln, sigma=s)
        dist.append(r)
        print("  sigma=%.2f%s  min=%.3f q25=%.3f med=%.3f q75=%.3f max=%.3f  "
              "P(d>0.5 step)=%.1f%%  P(d>corr_clip)=%.1f%%"
              % (s, " (beyond-physical)" if s > 1 else "", r["min"], r["q25"], r["median"],
                 r["q75"], r["max"], 100 * r["frac_above_half_step"],
                 100 * r["frac_above_corr_clip"]))

    print("\nNS-1.6 -- the driver's own table (swerve in compass steps / step at reference load):")
    for r in driver.report((1.0, 3.0)):
        print("  sigma=%.1f%s  mean_swerve=%.4f  peak=%.4f  swing=%.4f  placebo=%d/%d"
              % (r["sigma"], " (beyond-physical stress test)" if r["beyond_physical"] else "",
                 r["mean_swerve"], r["peak_swerve"], r["swing"], r["placebo_steps"],
                 r["period"]))

    out = dict(env_name=a.env_name, n_agents=n, spawn=pos.tolist(), load_norm=ln,
               ref=ref.tolist(), scale=scale.tolist(), operator=st,
               partition=dict(fixed=0.0, own=0.0, peer=1.0, note="100% by construction"),
               n_scaling=rows, loading=dist, driver=driver.report((1.0, 3.0)),
               dial=dict(period=p.period, wet_fraction=p.wet_fraction,
                         loss_at_sigma1=p.loss_at_sigma1, kernel_lambda=p.kernel_lambda,
                         rho=p.rho, recv_ball=p.recv_ball,
                         send=driver.send.tolist(), corr_clip=p.corr_clip))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("\nwrote %s -- commit it before any method run (NS-4.1)" % a.out)


if __name__ == "__main__":
    main()
