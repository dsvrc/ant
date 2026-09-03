"""Freeze the task constants, and MEASURE the ones the spec says to measure.

Everything here is run ONCE, BEFORE any method, and its output is committed.
NS_FORM_SPEC E.2 pitfall 10: *"Retuning after seeing a method fail plants the
problem.  Commit the gate output first; the history is the evidence that you did
not."*  Every number this script picks is a property of the TASK and is identical
for every arm.

  --sweep k_scale     the medium's units.  Pick it so the blind team is hurt but
                      still fights (D.2's calibration constraint), then freeze it.
  --sweep severity    what sigma costs a strong scripted controller.
  --sweep mu          PACT_PIPELINE_SPEC 5.1's forgetting table, re-measured here
                      because "the optimum follows the drift rate" and this
                      driver moves in tens of steps, not days.
  --sweep max_trust   8.4's T4 gain cap.  CALIBRATE ON ONE SEED AND VALIDATE ON
                      HELD-OUT SEEDS -- calibrating and reporting on the same seed
                      is fitting to the test set.

``--mock`` runs the whole thing against fc/mock_smac.py, so the pipeline can be
exercised on a machine with no StarCraft II.  Mock numbers calibrate NOTHING;
they only prove the sweep runs.  Every committed number must come from a real
run, and the script stamps which it was.

Usage::

    python -m harl.envs.smac.fc.calibrate --sweep k_scale --episodes 6
    python -m harl.envs.smac.fc.calibrate --sweep mu --steps 40000 --mock
"""

import argparse
import json
import sys

import numpy as np

from . import operator as opmod
from .certificates import pick_controller, reference_action
from .pact_env import PactEnv
from .severity_env import FormationCongestionEnv, N_ACTIONS_NO_ATTACK

#: The reference control is CHOSEN BY MEASUREMENT, not assumed -- see
#: certificates.pick_controller.  Measured on 3s_vs_5z: a focus-fire control
#: returns 3.64 and a kiting one 11.13, so every sweep run against the former was
#: calibrating against a strawman in exactly the dimension the channel acts on.
#: Where a healthy dial should land.  These are the CALIBRATION CONSTRAINTS, and
#: they come from the spec, not from what makes a method look good:
#:   D.2 / G3  -- the blind team must be HURT (mean deficit is not ~0) ...
#:   D.2       -- ... but must still FIGHT: a severity that drives the team into
#:                the farm-damage/timeout basin measures the basin, not the NS.
#:   A.5       -- Phi must keep varying.
TARGET_DELTA = (0.06, 0.25)
TARGET_MOVE_FRAC = 0.15


def _build(map_name, cfg, mock=False, seed=0):
    if mock:
        from .mock_smac import MockSmacEnv
        env = MockSmacEnv(map_name=map_name, seed=seed)
    else:
        from ..StarCraft2_Env import StarCraft2Env
        env = StarCraft2Env({"map_name": map_name, "state_type": "FP"})
        env.seed(seed)
    fc = FormationCongestionEnv(env, {**cfg, "ns_eval": 1, "ns_seed": seed})
    return (PactEnv(fc, cfg) if int(cfg.get("pact", 0)) else fc), fc, env


def _drive(top, fc, env, steps, seed=0, controller=None, privileged=False):
    """Roll a scripted controller and return the measured facts.

    The per-step diagnostics are ACCUMULATED, not read once at the end: the
    wrapper's state is per-step and a reset zeroes it, so a final-state snapshot
    silently reports 0 whenever the run happens to end on an episode boundary.
    NS_FORM_SPEC E.2 pitfall 8's cousin -- read a window, not an endpoint.
    """
    rng = np.random.RandomState(seed)
    top.reset()
    ep_r, eps, lens, t_ep = 0.0, [], [], 0
    u_hat = np.zeros(fc.n_agents)
    acc = {k: [] for k in ("delta", "u", "stride", "peer_share", "dnz", "trust",
                           "fit_gain", "ff", "peer")}
    deliv = []
    for _ in range(int(steps)):
        av = env.get_avail_actions()
        if controller is None:
            acts = []
            for i in range(env.n_agents):
                ok = np.where(np.asarray(av[i]) > 0)[0]
                acts.append(int(rng.choice(ok)) if ok.size else 0)
        else:
            acts = [controller(env, i, av[i]) for i in range(env.n_agents)]
        if privileged:
            # the exact channel inverse on the TRUE deficit, with the driver model
            # known, so it pre-empts the trough (D.1: privileged IN STRATEGY)
            g_next = float(fc.driver.g()[0])
            pre = np.clip(u_hat * (1.0 - g_next), 0.0, fc.harm_max)
            fc.set_command(np.clip(1.0 / np.maximum(0.25, 1.0 - pre), 0.25, 4.0))
        out = top.step(np.array(acts).reshape(-1, 1))
        gnow = float(fc.g)
        if gnow < 1.0:
            u_hat = 0.7 * u_hat + 0.3 * (fc.delta / max(1e-9, 1.0 - gnow))
        info = out[4][0]
        ep_r += float(np.asarray(out[2]).reshape(-1)[0])
        t_ep += 1
        live_m = fc._alive > 0
        if live_m.any():
            deliv.append(float(np.mean((fc.stride
                                        / np.maximum(1e-12, fc.base_frac))[live_m])))
        for k, key in (("delta", "fc_delta_mean"), ("u", "fc_u_mean"),
                       ("stride", "fc_stride_mean"), ("peer_share", "fc_peer_share"),
                       ("dnz", "pact_delta_nonzero_frac"),
                       ("trust", "pact_applied_trust"),
                       ("fit_gain", "pact_fit_gain_now"), ("ff", "pact_ff_abs"),
                       ("peer", "pact_peer_abs")):
            if key in info:
                acc[k].append(info[key])
        if bool(np.all(np.asarray(out[3]))):
            eps.append(ep_r)
            lens.append(t_ep)
            ep_r, t_ep = 0.0, 0
            top.reset()

    def m(v):
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float("nan")

    d = fc.diagnostics()          # the cumulative counters only
    out = {k: m(v) for k, v in acc.items()}
    out.update(move_frac=d["fc_move_frac"], phi_var=d["fc_phi_var"],
               dial_ratio=d["fc_dial_ratio"], odom_err=d["fc_odom_err"],
               ret=m(eps), ep_len=m(lens), n_ep=len(eps),
               ret_se=float(np.std(eps) / max(1.0, np.sqrt(len(eps)))) if eps
               else float("nan"),
               deliv=float(np.mean(deliv)) if deliv else float("nan"))
    return out


def sweep_k_scale(map_name, values, steps, mock, seed, ctl=None):
    """Set the dial's strength, and set it on the number that decides everything.

    The earlier version of this sweep reported ``delta_mean`` and asked whether it
    landed in a plausible band.  That was the wrong target twice over: it was run
    on the wrong map, against a control with no movement skill, and Delta landing
    in band tells you nothing about whether the harm reaches REWARD.  Measured:
    at the same delivery loss (deliv 0.853 vs 0.860) the return cost was 4% on
    1c3s5z and 26% on 3s_vs_5z -- a 6x difference in transmission that a Delta
    band cannot see.

    So each row now runs the privileged oracle too and reports **G4b** -- what a
    perfect compensator on the true deficit buys over blind.  That is the quantity
    the whole campaign rests on, and no method can exceed it.

    Pick the WEAKEST dial that clears the bar while the team still fights (D.2's
    constraint: a severity that drives the team into a collapse basin measures the
    basin, not the NS), then FREEZE it and commit this output before any method
    run.
    """
    ctl = ctl or reference_action
    top, fc, env = _build(map_name, {"ns_severity": 0.0}, mock, seed)
    r0 = _drive(top, fc, env, steps, seed, ctl)
    env.close()
    print("k_scale sweep on %s -- sigma=0 reference: ret %.2f, deliv %.3f"
          % (map_name, r0["ret"], r0["deliv"]))
    print("  %-9s %-8s %-8s %-15s %-8s %-9s %-8s %s"
          % ("k_scale", "delta", "deliv", "ret vs s=0", "G4b", "deliv_rec",
             "move_fr", "verdict"))
    rows = []
    for k in values:
        cfg = {"ns_severity": 1.0, "ns_k_scale": float(k)}
        top, fc, env = _build(map_name, cfg, mock, seed)
        r = _drive(top, fc, env, steps, seed, ctl)
        env.close()
        top, fc, env = _build(map_name, cfg, mock, seed)
        rp = _drive(top, fc, env, steps, seed, ctl, privileged=True)
        env.close()
        hurt = r0["ret"] - r["ret"]
        h2se = 2.0 * float(np.hypot(r0["ret_se"], r["ret_se"]))
        g4b = rp["ret"] / max(1e-9, r["ret"])
        lost = r0["deliv"] - r["deliv"]
        rec = (rp["deliv"] - r["deliv"]) / lost if lost > 1e-6 else float("nan")
        why = []
        if g4b < 1.30:
            why.append("G4b %.2f < 1.30" % g4b)
        if hurt <= h2se:
            why.append("hurt inside noise")
        if r["ret"] < 0.35 * r0["ret"]:
            why.append("team collapsed")        # D.2: this measures the basin
        if not (r["phi_var"] > 0.05):
            why.append("Phi near-constant")
        ok = not why
        rows.append(dict(k_scale=float(k), ok=bool(ok), g4b=float(g4b),
                         deliv_rec=float(rec), hurt=float(hurt), hurt_2se=float(h2se),
                         ret0=r0["ret"], ret_priv=rp["ret"], **r))
        print("  %-9.3f %-8.3f %-8.3f %-15s %-8.3f %-9.2f %-8.2f %s"
              % (k, r["delta"], r["deliv"],
                 "%.1f (-%.1f/%.1f)" % (r["ret"], hurt, h2se), g4b, rec,
                 r["move_frac"], "OK" if ok else ", ".join(why)))
    good = [r for r in rows if r["ok"]]
    if good:
        pick = max(good, key=lambda r: r["k_scale"])     # the WEAKEST dial that works
        print("  -> set ns_k_scale: %.3f (the weakest dial that clears G4b) and "
              "FREEZE it" % pick["k_scale"])
    else:
        best = max(rows, key=lambda r: r["g4b"])
        print("  -> nothing clears G4b >= 1.30; best is k_scale=%.3f at %.3f."
              % (best["k_scale"], best["g4b"]))
        print("     Next lever is severity (--sweep severity), then this map is "
              "not the one.")
    return rows


def sweep_severity(map_name, values, steps, mock, seed, ctl=None):
    print("severity sweep -- gate G3 ('a dial that does not hurt is not a dial')")
    print("  %-8s %-11s %-11s %-10s %-9s %-9s %s"
          % ("sigma", "delta_mean", "dial_ratio", "stride", "ret", "ep_len", "peer_share"))
    rows = []
    for s in values:
        top, fc, env = _build(map_name, {"ns_severity": float(s)}, mock, seed)
        r = _drive(top, fc, env, steps, seed, ctl or reference_action)
        env.close()
        print("  %-8.2f %-11.4f %-11.3f %-10.3f %-9.2f %-9.1f %.3f"
              % (s, r["delta"], r["dial_ratio"], r["stride"], r["ret"],
                 r["ep_len"], r["peer_share"]))
        rows.append(dict(sigma=float(s), **r))
    return rows


def sweep_mu(map_name, values, steps, mock, seed, ctl=None):
    print("mu sweep -- spec 5.1.  RE-MEASURE PER ENVIRONMENT: the optimum follows")
    print("the drift rate, and this driver moves in tens of steps, not days.")
    print("  %-9s %-11s %-11s %-10s %-9s %s"
          % ("mu", "fit_gain", "applied_tr", "delta_nz", "stride", "ff/peer"))
    rows = []
    for mu in values:
        top, fc, env = _build(map_name,
                              {"ns_severity": 1.0, "pact": 1, "pact_mu": float(mu)},
                              mock, seed)
        r = _drive(top, fc, env, steps, seed, ctl or reference_action)
        env.close()
        share = r["ff"] / max(1e-12, r["ff"] + r["peer"])
        print("  %-9.4f %-11.4f %-11.4f %-10.3f %-9.3f %.2f"
              % (mu, r["fit_gain"], r["trust"], r["dnz"], r["stride"], share))
        rows.append(dict(mu=float(mu), ff_share=float(share), **r))
    best = max((r for r in rows if np.isfinite(r["fit_gain"])),
               key=lambda r: r["fit_gain"], default=None)
    if best:
        print("  -> best measured fit_gain at mu = %.4f.  Set pact_mu explicitly if "
              "it differs from the auto-derived value." % best["mu"])
    return rows


def sweep_max_trust(map_name, values, steps, mock, seed, ctl=None):
    print("max_trust sweep -- spec 8.4's T4 gain cap.")
    print("*** CALIBRATE ON ONE SEED AND VALIDATE ON HELD-OUT SEEDS.  Calibrating")
    print("*** and reporting on the same seed is fitting to the test set.")
    print("  %-11s %-11s %-10s %-11s %-9s %s"
          % ("max_trust", "stride_mean", "delta_nz", "delta_clip", "ret", "verdict"))
    rows = []
    base = None
    for g in values:
        top, fc, env = _build(map_name,
                              {"ns_severity": 1.0, "pact": 1,
                               "pact_max_trust": float(g)}, mock, seed)
        r = _drive(top, fc, env, steps, seed, ctl or reference_action)
        env.close()
        if base is None:
            base = r["stride"]
        print("  %-11.3f %-11.4f %-10.3f %-11s %-9.2f %s"
              % (g, r["stride"], r["dnz"], "-", r["ret"],
                 "+" if r["stride"] > base else "-"))
        rows.append(dict(max_trust=float(g), **r))
    print("  An INVERTED U here is T4 appearing in a real environment and is")
    print("  EVIDENCE, not tuning -- report the whole sweep, not the peak.")
    return rows


SWEEPS = {
    # Reaches lower than before: on a map where the team is near its ceiling the
    # dial has to be a good deal stronger before the return moves at all.
    "k_scale": (sweep_k_scale, [0.08, 0.12, 0.18, 0.25, 0.35, 0.50]),
    "severity": (sweep_severity, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]),
    "mu": (sweep_mu, [0.80, 0.88, 0.92, 0.95, 0.99, 0.9995]),
    "max_trust": (sweep_max_trust, [0.0, 0.25, 0.5, 0.75, 1.0]),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=sorted(SWEEPS), required=True)
    ap.add_argument("--map", default="1c3s5z")
    ap.add_argument("--controller", default="best",
                    choices=["best", "focus", "kite"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--values", default="", help="comma-separated override")
    ap.add_argument("--mock", action="store_true",
                    help="run against fc/mock_smac.py -- exercises the sweep, "
                         "calibrates NOTHING")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    fn, default_vals = SWEEPS[args.sweep]
    # D.1, enforced rather than assumed: pick the stronger control on THIS map at
    # sigma=0 before the dial is ever switched on.
    if args.mock:
        ctl = reference_action
    else:
        _, ctl, _ = pick_controller(args.map, args.seed, 8, args.controller)
    vals = ([float(v) for v in args.values.split(",")] if args.values
            else default_vals)
    if args.mock:
        print("*** --mock: these numbers are from a TEST DOUBLE and calibrate "
              "nothing.  Commit only numbers from a real StarCraft II run. ***")
    rows = fn(args.map, vals, args.steps, args.mock, args.seed, ctl)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(dict(sweep=args.sweep, map=args.map, seed=args.seed,
                           mock=bool(args.mock), rows=rows), f, indent=2,
                      default=float)
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
