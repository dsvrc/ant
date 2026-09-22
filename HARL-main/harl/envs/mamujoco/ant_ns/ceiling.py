"""Part C -- the partition, the N-scaling and the loading distribution, with NO
training and NO simulator.

`PACT_NS_SPEC` I.5 / NS-4.1: *"Compute and commit the decomposition before any
method code exists.  If the coordination gap is small, this environment is a
poor showcase; say so and pick another."*

WHAT THE PARTITION SAYS HERE, HONESTLY
--------------------------------------------------------------------------------
Each agent's excess torque over its sigma = 0 counterfactual is ``d_i``, and
every unit of it traces to a contributor:

    Delta_fixed   load from participants no agent controls  -- 0 %.  Every joint
                  of the ant belongs to some agent; there is no uncontrolled
                  actuator on the machine.
    Delta_own     the agent's own joints                    -- 0 %, by j != i.
    Delta_peer    the other agents                          -- 100 %.

So the coordination gap is 100 % BY CONSTRUCTION and THE ENDPOINT CARRIES NO
INFORMATION -- the same caveat ``smac_ns`` and ``grf_ns`` state.  What is
informative is the CURVE, and on Ant the curve is unusually strong evidence
because it is RUNNABLE: ``r = 3`` load paths are independent of the partition, so
the identical dial and the identical basis serve

    2x4 (N=2)  ->  4x2 (N=4)  ->  8x1 (N=8)

and each of those is a real training run rather than only an offline number.
Splitting a leg's hip from its own ankle (``8x1``) moves the strongest load path
in the machine out of "solvable alone" and into "requires coordination" without
making the task lossier.  That is NS-4.2's prediction stated as an experiment.

Run::

    python -m harl.envs.mamujoco.ant_ns.ceiling --out ceiling.json
"""

import argparse
import json

import numpy as np

from .coupling import Coupling, _LoneCoupling
from .driver import DialParams, ThermalDriver
from .structure import (CLASS_NAMES, JOINT_NAMES, N_JOINTS, agent_of, describe, kernel,
                        kernel_source, partition_of, recv_vector)


def peer_load(p, driver, agent_conf, load_norm, sigma=1.0, samples=4096, seed=1,
              full=False):
    """The per-agent disturbance magnitude ``|d_i|`` at the driver's PEAK, under
    random peer torques, normalised by the reference ``load_norm``.

    ``full=True`` uses the declared reference condition (every peer at full
    torque, random sign); otherwise peers are uniform over their range, which is
    closer to what a working controller does.
    """
    c = Coupling(agent_conf, p, driver.send)
    rng = np.random.RandomState(int(seed))
    amp = float(sigma) * p.loss_at_sigma1
    vals = []
    for _ in range(int(samples)):
        tau = (rng.choice([-1.0, 1.0], size=N_JOINTS) if full
               else rng.uniform(-1.0, 1.0, N_JOINTS))
        _, x = c.project(c._settle(tau))
        vals.append(np.abs(x @ (amp * driver.send)) / load_norm)
    return np.concatenate([np.atleast_1d(v) for v in vals]), c


def coordination_share(p, agent_conf):
    """How much of the MACHINE's own coupling crosses an agent boundary.

    This is the I.5 decomposition in the only form that carries information
    here.  The trunk couples every pair of joints; the partition decides which
    of those couplings one agent sees BOTH ends of:

        own   pairs inside one agent -- that agent resolves them alone, and the
              dial never touches them (the peer mask excludes them)
        peer  pairs across agents    -- what the dial amplifies, and what no
              agent can resolve alone

    ``peer / total`` therefore rises from 0 (one agent owning the machine) to 1
    (every joint its own agent) WITHOUT the task becoming lossier: the same
    physics is simply re-labelled from "solvable alone" to "requires
    coordination".  That is the mechanism behind NS-4.2's prediction.
    """
    K = recv_vector(p.recv_spread)[:, None] * kernel(p.length_scale)
    K = K * (~np.eye(N_JOINTS, dtype=bool))
    owner = agent_of(partition_of(agent_conf))
    peer = float((K * (owner[:, None] != owner[None, :])).sum())
    total = float(K.sum())
    return peer / max(total, 1e-30), peer, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent_conf", default="4x2", help="the REFERENCE partition")
    ap.add_argument("--sigmas", default="0.5,1,2,3")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    p = DialParams()
    driver = ThermalDriver(p)
    driver.certify()
    ref_conf = a.agent_conf
    c = Coupling(ref_conf, p, driver.send)
    print(c.verify())
    print("")
    print(describe(p.length_scale))
    print("joint order (ant.xml actuators): %s" % list(JOINT_NAMES))
    print("load paths (r = %d, live %d): %s   pruned (P-3.4): %s   send (UNKNOWN "
          "to the agent) = %s"
          % (c.r, c.r_live, [CLASS_NAMES[m] for m in c.live],
             [CLASS_NAMES[m] for m in c.pruned] or "none",
             np.round(driver.send, 3).tolist()))

    ref, scale = c.geometric_reference()
    ln = c.load_norm()
    print("")
    print("REFERENCES TO COMMIT (mamujoco_ns.yaml: ns_load_norm), partition %s:" % ref_conf)
    print("  load_norm = %.6f     ref = %s     scale = %s"
          % (ln, np.round(ref, 4).tolist(), np.round(scale, 4).tolist()))

    st = c.operator_stats()
    print("")
    print("operator W (joint level): zero_diag=%s spread=%.3f ratio=%.1fx asym=%.3f "
          "links=%d   (NS-1.2 wants a zero diagonal, spread, and measurable asymmetry)"
          % (st["diag_max"] == 0.0, st["spread"], st["ratio"], st["asymmetry"],
             st["n_links"]))
    A = c.agent_operator()
    print("  aggregated to agents:")
    print(np.round(A, 3))

    print("")
    print("PART C -- the partition (I.5).  By construction on this instance:")
    print("  Delta_fixed = 0 %   (every actuator on the machine belongs to an agent)")
    print("  Delta_own   = 0 %   (j != i)")
    print("  Delta_peer  = 100 % -> coordination gap 100 %.  THE ENDPOINT CARRIES NO")
    print("  INFORMATION; quote the N-scaling curve below, not the endpoint.")

    print("")
    print("N-SCALING at sigma=1, driver peak (NS-4.2).  Every row is a runnable training")
    print("config, and the dial, the basis and r = 3 are IDENTICAL in all of them --")
    print("only the partition moves:")
    print("  %-6s %3s  %14s  %18s  %13s"
          % ("conf", "N", "|d| per joint", "coupling crossing", "exactly zero"))
    rows = []
    for conf in ("1x8", "2x4", "2x4d", "4x2", "8x1"):
        share, peer_w, total_w = coordination_share(p, conf)
        if conf == "1x8":
            lone = _LoneCoupling(p, driver.send)
            _, x1 = lone.project(lone._settle(np.ones(N_JOINTS)))
            assert float(np.abs(x1).max()) == 0.0, (
                "a lone agent must read EXACTLY zero (gate 3)")
            per_joint, mx, zero, n_ag = 0.0, 0.0, 1.0, 1
        else:
            v, cc = peer_load(p, driver, conf, ln)
            n_ag = cc.n
            #  |d| per JOINT: the torque error each ACTUATOR actually suffers.
            #  The per-AGENT magnitude is NOT comparable across partitions -- an
            #  agent owning four joints has a larger vector norm for the same
            #  per-actuator error, which reads as "more disturbed" when it is
            #  not.  Reading the per-agent number instead inverts this table and
            #  would have reported NS-4.2's prediction as refuted.
            per_joint = float(v.mean() / np.sqrt(cc.dims[0]))
            mx = float(v.max() / np.sqrt(cc.dims[0]))
            zero = float(np.mean(v == 0.0))
        rows.append(dict(agent_conf=conf, N=n_ag, d_per_joint=per_joint, d_max=mx,
                         zero_frac=zero, coupling_crossing=share,
                         peer_weight=peer_w, total_weight=total_w))
        print("  %-6s %3d  %14.4f  %17.1f%%  %12.1f%%%s"
              % (conf, n_ag, per_joint, 100 * share, 100 * zero,
                 "   <- a lone agent, structurally" if conf == "1x8" else ""))
    by_n = sorted(rows, key=lambda r: r["N"])
    #  The theoretical quantity -- the share of the machine's coupling that crosses
    #  an agent boundary -- must be non-decreasing in N.  That is the prediction.
    assert all(b["coupling_crossing"] >= a_["coupling_crossing"] - 1e-12
               for a_, b in zip(by_n, by_n[1:])), (
        "the share of the coupling crossing an agent boundary must not FALL as the "
        "partition gets finer; it did, so the partition table is wrong")
    #  The measured per-actuator disturbance must rise with it, up to the point
    #  where the share saturates.  On Ant with the machine's own operator that
    #  point is 4x2: the only pair inside a 4x2 agent is its own hip-ankle, and
    #  that path carries EXACTLY ZERO, so 100% already crosses and 8x1 can add
    #  nothing.  Saturation is a real property of this robot, not a failure --
    #  and "8x1 shows no further change" is a sharper prediction than "more
    #  agents is worse".
    upto = [r for r in by_n if r["coupling_crossing"] < 1.0 - 1e-12]
    upto += [r for r in by_n if r["coupling_crossing"] >= 1.0 - 1e-12][:1]
    assert all(b["d_per_joint"] >= a_["d_per_joint"] - 1e-12
               for a_, b in zip(upto, upto[1:])), (
        "the per-actuator disturbance must RISE with N while the coupling share is "
        "still rising; it did not, so this instance is a poor showcase for NS-4.2 "
        "and the README must say so")
    sat = [r["agent_conf"] for r in by_n if r["coupling_crossing"] >= 1.0 - 1e-12]
    if len(sat) > 1:
        print("")
        print("  SATURATED at %s: every coupled pair already crosses an agent "
              "boundary there, so %s add nothing.  On Ant this is exact -- the only "
              "pair inside a 4x2 agent is its own hip-ankle, and that load path is "
              "identically zero." % (sat[0], ", ".join(sat[1:])))

    print("")
    print("LOADING DISTRIBUTION of |d| at the peak, partition %s (fraction of the "
          "torque range):" % ref_conf)
    dist = []
    for s in [float(x) for x in a.sigmas.split(",")]:
        v, _ = peer_load(p, driver, ref_conf, ln, sigma=s)
        q = np.percentile(v, [25, 50, 75, 100])
        dist.append(dict(sigma=s, q25=float(q[0]), median=float(q[1]), q75=float(q[2]),
                         max=float(q[3]), mean=float(v.mean()),
                         frac_above_corr_clip=float(np.mean(v > p.corr_clip)),
                         frac_above_range=float(np.mean(v > 1.0))))
        print("  sigma=%.2f%s  q25=%.3f med=%.3f q75=%.3f max=%.3f  "
              "P(|d|>corr_clip)=%.1f%%  P(|d|>full range)=%.1f%%"
              % (s, " (beyond-physical)" if s > 1 else "", q[0], q[1], q[2], q[3],
                 100 * np.mean(v > p.corr_clip), 100 * np.mean(v > 1.0)))
    v_full, _ = peer_load(p, driver, ref_conf, ln, sigma=1.0, full=True)
    print("  the DECLARED reference condition (every peer at full torque, sigma=1, "
          "peak): mean |d| = %.4f of the torque range -- this is what 'L = %.2f' means"
          % (v_full.mean(), p.loss_at_sigma1))

    print("")
    print("NS-1.6 -- the driver's own table:")
    for r in driver.report((1.0, 3.0)):
        print("  sigma=%.1f%s  mean_amplitude=%.4f  peak=%.4f  swing=%.4f  placebo=%d/%d"
              % (r["sigma"],
                 " (beyond-physical stress test)" if r["beyond_physical"] else "",
                 r["mean_amplitude"], r["peak_amplitude"], r["swing"],
                 r["placebo_steps"], r["period"]))

    out = dict(agent_conf=ref_conf, load_norm=ln, ref=ref.tolist(), scale=scale.tolist(),
               operator=st, agent_operator=A.tolist(),
               partition=dict(fixed=0.0, own=0.0, peer=1.0, note="100% by construction"),
               n_scaling=rows, loading=dist, driver=driver.report((1.0, 3.0)),
               joint_names=list(JOINT_NAMES), classes=list(CLASS_NAMES),
               dial=dict(period=p.period, warm_fraction=p.warm_fraction,
                         loss_at_sigma1=p.loss_at_sigma1, rho=p.rho,
                         length_scale=p.length_scale, recv_spread=p.recv_spread,
                         kernel_source=kernel_source()[0], send=driver.send.tolist(),
                         corr_clip=p.corr_clip))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("")
        print("wrote %s -- commit it before any method run (NS-4.1)" % a.out)


if __name__ == "__main__":
    main()
