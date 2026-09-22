"""In-simulator identities.  Needs mujoco; ~2 minutes.

Everything ``selftest.py`` proves about the channel arithmetic is re-checked here
THROUGH MuJoCo, plus the things only the engine can answer: which actuators an
agent's action really reaches (gate 4, physically), whether the declared operator
agrees with the model's own inverse inertia, and -- the one that matters most on
this host -- whether the disturbance stays OUT of the reward.

MuJoCo is deterministic given a seed and an action stream, so unlike ``grf_ns``
the identities here are checked at the TRAJECTORY level, bit for bit.

    python -m harl.envs.mamujoco.ant_ns.smoke
    python -m harl.envs.mamujoco.ant_ns.smoke --steps 600

Ends with ``ALL SMOKE CHECKS PASSED`` or lists the failures and exits 1.
"""

import argparse
import sys

import numpy as np

from harl.utils.configs_tools import get_defaults_yaml_args

from . import make_ant_ns_env
from .coupling import Coupling
from .driver import DialParams, ThermalDriver
from .structure import (JOINT_NAMES, N_JOINTS, kernel_source,
                        model_inverse_inertia)

FAILS = []
P = DialParams()
PEAK = int(P.period * P.warm_fraction / 2)
COLD = int(P.period * P.warm_fraction)


def check(name, cond, detail=""):
    ok = bool(cond)
    print("  [%s] %-56s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def make(agent_conf="4x2", **kw):
    _, ea = get_defaults_yaml_args("happo", "mamujoco_ns")
    ea = dict(ea)
    ea["agent_conf"] = agent_conf
    if agent_conf != "4x2":
        ea["ns_load_norm"] = None          # the committed value is 4x2's
    ea.update(kw)
    return make_ant_ns_env(ea, 0, 1)


def rollout(env, seed, steps, policy_seed=0):
    """A fixed open-loop action stream.  Returns the trajectory and, per step,
    the commanded actions and the layer's own panel."""
    env.seed(seed)
    obs, _, _ = env.reset()
    n = len(obs)
    dims = [int(sp.shape[0]) for sp in env.action_space]
    rng = np.random.RandomState(policy_seed)
    traj, cmds, infos = [], [], []
    for t in range(steps):
        a = [np.clip(0.7 * np.sin(0.25 * t + 0.8 * i) + 0.25 * rng.randn(dims[i]),
                     -1.0, 1.0) for i in range(n)]
        obs, state, rew, done, info, _ = env.step(a)
        traj.append((np.concatenate([np.asarray(o, dtype=np.float64).ravel() for o in obs]),
                     float(np.asarray(rew).ravel()[0]), bool(np.all(done))))
        cmds.append(np.concatenate(a))
        infos.append(info[0] if isinstance(info, (list, tuple)) else info)
        if np.all(done):
            obs, _, _ = env.reset()
    return traj, np.array(cmds), infos


def identical(t1, t2):
    if len(t1) != len(t2):
        return False, "lengths %d vs %d" % (len(t1), len(t2))
    for k, ((o1, r1, d1), (o2, r2, d2)) in enumerate(zip(t1, t2)):
        if o1.shape != o2.shape or not np.array_equal(o1, o2) or r1 != r2 or d1 != d2:
            return False, "first divergence at step %d" % k
    return True, "%d steps identical, observation and reward bit for bit" % len(t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    S = args.steps

    # ------------------------------------------------------------------ gate 4
    print("gate 4 -- which actuators an agent's action REALLY reaches", flush=True)
    e = make(ns_severity=0.0)
    raw = e.env
    names = []
    for k in range(int(raw.model.nu)):
        jid = int(np.asarray(raw.model.actuator_trnid)[k, 0])
        names.append(str(raw.model.joint_names[jid] if hasattr(raw.model, "joint_names")
                         else raw.model.joint_id2name(jid)))
    check("actuator_order_matches_the_declared_table", tuple(names) == JOINT_NAMES,
          "%s" % names)
    #  the definitive measurement: drive ONE agent and read the engine's ctrl.
    #  MAMuJoCo's obsk table and its own step() disagree about which joints an
    #  agent owns; this is what settles it, and structure.PARTITIONS follows it.
    e.seed(1)
    e.reset()
    probe = [np.zeros(int(sp.shape[0])) for sp in e.action_space]
    probe[0][:] = 1.0
    e.step(probe)
    reached = np.where(np.abs(np.asarray(raw.sim.data.ctrl)) > 1e-9)[0].tolist()
    declared = list(e.coupling.parts[0])
    check("agent_0_drives_the_joints_structure.py_says_it_does",
          reached == declared,
          "engine ctrl touched %s (%s); structure.PARTITIONS says %s"
          % (reached, [names[i] for i in reached], declared))
    Minv = model_inverse_inertia(raw)
    src, note = kernel_source()
    print("  [ -- ] transmission structure in force: %s -- %s" % (src.upper(), note))
    if Minv is None:
        print("  [ -- ] the installed binding exposes no dense mass matrix; the "
              "declared operator could not be compared with the model")
    else:
        off = ~np.eye(N_JOINTS, dtype=bool)
        corr = float(np.corrcoef(e.coupling.kappa[off], np.abs(Minv)[off])[0, 1])
        if src == "committed":
            check("committed_operator_is_the_model's_own_inverse_inertia", corr > 0.65,
                  "corr(kappa, |M^-1|) = %+.3f at the running pose (the committed "
                  "matrix was dumped at the NOMINAL pose, so < 1 is expected)" % corr)
        else:
            #  NOT a failure: the surrogate is a declared transmission model, and
            #  this is the measurement that says so.  Run dump_operator.py to
            #  replace it with the machine's own |M^-1|.
            print("  [ -- ] surrogate kappa vs the model's own |M^-1|: corr = %+.3f "
                  "-- same support and ordering, different shape.  Run "
                  "dump_operator.py to use the machine's own operator." % corr)
    e.close()

    # ------------------------------------------------ NS-1.4: the reward is untouched
    print("NS-1.4 -- the disturbance never reaches the reward function", flush=True)
    e = make(ns_severity=2.0, ns_phase0=PEAK)
    e.seed(3)
    e.reset()
    worst, n = 0.0, 0
    for t in range(120):
        a = [np.clip(0.6 * np.sin(0.3 * t + i) * np.ones(int(sp.shape[0])), -1, 1)
             for i, sp in enumerate(e.action_space)]
        _, _, _, _, info, _ = e.step(a)
        row = info[0]
        if "reward_ctrl" in row:
            expect = -0.5 * float(np.square(np.concatenate(a)).sum())
            worst = max(worst, abs(float(row["reward_ctrl"]) - expect))
            n += 1
    check("control_cost_is_charged_on_the_POLICY's_torque_not_the_disturbed_one",
          n > 0 and worst < 1e-9,
          "max |reward_ctrl - (-0.5|a_commanded|^2)| = %.2e over %d steps with the "
          "dial live -- so the disturbance removes capability and is NOT a penalty "
          "term (NS-1.4)" % (worst, n))
    #  and the executed torque really did differ from the commanded one
    check("the_executed_torque_differed_from_the_commanded_one",
          e.chan.n_harmed > 0 and e.chan.n_dial_live > 0,
          "dial_live=%d harmed=%d" % (e.chan.n_dial_live, e.chan.n_harmed))
    check("the_engine's_ctrl_is_the_layer's_executed_torque",
          float(np.abs(np.asarray(e.env.sim.data.ctrl) - e.chan.tau_prev).max()) < 1e-12,
          "max |sim.data.ctrl - u_exec| = %.2e"
          % float(np.abs(np.asarray(e.env.sim.data.ctrl) - e.chan.tau_prev).max()))
    e.close()

    # ------------------------------------------------------------------ NS-2.1
    print("NS-2.1 -- sigma = 0 is stock Ant, bit for bit", flush=True)
    stock = make(ns_on=0)
    layer0 = make(ns_severity=0.0)
    t_stock, _, _ = rollout(stock, 7, S)
    t_zero, _, _ = rollout(layer0, 7, S)
    ok, msg = identical(t_stock, t_zero)
    check("sigma_0_is_bit_identical_to_the_stock_host", ok, msg)
    stock.close()

    # ------------------------------------------------------------------ NS-2.5
    print("NS-2.5 -- the cold half is inert at sigma = 3", flush=True)
    plc = make(ns_severity=3.0, ns_phase0=COLD)
    t_plc, _, _ = rollout(plc, 7, S)
    ok, msg = identical(t_zero, t_plc)
    check("placebo_regime_is_bit_identical_to_sigma_0", ok, msg)
    check("placebo_counters_say_so",
          plc.chan.n_dial_live == 0 and plc.chan.n_harmed == 0,
          "dial_live=%d harmed=%d" % (plc.chan.n_dial_live, plc.chan.n_harmed))
    plc.close()

    # ------------------------------------------------------------------ NS-3.3
    print("NS-3.3 -- the layer fires at the driver peak", flush=True)
    blind = make(ns_severity=2.0, ns_phase0=PEAK)
    t_blind, cmds, infos = rollout(blind, 7, S)
    c = blind.chan
    check("dial_live_and_harmed_are_non_zero",
          c.n_dial_live > 0 and c.n_harmed > 0,
          "dial_live=%d harmed=%d actuator_clipped=%d of %d steps"
          % (c.n_dial_live, c.n_harmed, c.n_clipped, c.n_steps))
    ok, _ = identical(t_zero, t_blind)
    check("sigma_2_actually_changes_the_trajectory", not ok,
          "mean |d| = %.4f of the torque range"
          % float(np.mean([r["ns_dmax"] for r in infos if "ns_dmax" in r])))
    check("info_carries_the_panel",
          all(k in infos[0] for k in ("ns_d", "ns_A", "ns_fit_gain", "ns_dial_live",
                                      "reward_forward", "reward_ctrl")),
          "%d ns_* keys + the host's own reward decomposition"
          % len([k for k in infos[0] if str(k).startswith("ns_")]))
    check("layer_close_report_accepts_a_live_dial", c.close_report(tag="  [smoke]"))
    blind.close()

    # ------------------------------------------------------------------ P-7.1
    print("P-7.1 -- pactoff is the blind arm, bit for bit", flush=True)
    off = make(ns_severity=2.0, ns_phase0=PEAK, ns_pact=1, ns_trust="off")
    t_off, _, _ = rollout(off, 7, S)
    ok, msg = identical(t_blind, t_off)
    check("pactoff_is_bit_identical_to_blind", ok, msg)
    check("pactoff_estimator_ran_but_corrected_nothing",
          off.chan.rows.sum() > 0 and float(np.abs(off.chan.c).max()) == 0.0,
          "rows=%d max|c|=%g" % (int(off.chan.rows.sum()), float(np.abs(off.chan.c).max())))
    off.close()

    # ------------------------------------------------------------------ II.6
    print("II.6 -- the oracle cancels the disturbance exactly", flush=True)
    orc = make(ns_severity=2.0, ns_phase0=PEAK, ns_pact=1, ns_trust="fixed",
               ns_g_fixed=1.0, ns_oracle=1, ns_corr_clip=0.0)
    t_orc, _, _ = rollout(orc, 7, S)
    ok, msg = identical(t_zero, t_orc)
    check("oracle_at_sigma_2_is_bit_identical_to_sigma_0", ok, msg)
    check("oracle_saw_a_live_dial_and_enacted_nothing",
          orc.chan.n_dial_live > 0 and orc.chan.n_harmed == 0,
          "dial_live=%d harmed=%d -- the ceiling IS B0, and it is an identity "
          "rather than an argument" % (orc.chan.n_dial_live, orc.chan.n_harmed))
    orc.close()
    layer0.close()

    # ------------------------------------------------------- the layer's own wiring
    print("the layer computes what coupling.py says it computes", flush=True)
    e = make(ns_severity=2.0, ns_phase0=PEAK)
    e.seed(11)
    e.reset()
    drv = ThermalDriver(e.ns)
    ref_c = Coupling("4x2", e.ns, drv.send)
    Q = np.zeros((ref_c.r, N_JOINTS))
    tau_prev = np.zeros(N_JOINTS)
    worst = 0.0
    for t in range(80):
        a = [np.clip(0.5 * np.sin(0.2 * t + i) * np.ones(int(sp.shape[0])), -1, 1)
             for i, sp in enumerate(e.action_space)]
        Q = ref_c.step_channels(Q, tau_prev)
        ee, xx = ref_c.project(Q)
        amp = float(drv.amplitude(e.chan.clock))
        expect = np.zeros(N_JOINTS)
        for i, grp in enumerate(ref_c.parts):
            idx = np.asarray(grp)
            expect[idx] = ee[idx] * (float(drv.send @ xx[i]) * amp / e.chan.load_norm)
        e.step(a)
        worst = max(worst, float(np.abs(e.chan.d - expect).max()))
        tau_prev = e.chan.tau_prev.copy()
    check("the_disturbance_in_the_engine_matches_an_offline_recomputation",
          worst < 1e-12,
          "max |d_engine - d_offline| = %.2e over 80 steps" % worst)
    e.close()

    # ------------------------------------------------------------------ N-scaling
    print("NS-4.2 -- the per-actuator disturbance rises with N, in the engine",
          flush=True)
    per = {}
    #  the committed reference load_norm for EVERY partition: that is the whole
    #  point of holding it fixed, and letting each partition normalise by its own
    #  divides the effect out (measured: it inverted the table).
    _, ea0 = get_defaults_yaml_args("happo", "mamujoco_ns")
    ln_ref = ea0["ns_load_norm"]
    for conf in ("2x4", "4x2", "8x1"):
        env = make(conf, ns_severity=2.0, ns_phase0=PEAK, ns_load_norm=ln_ref)
        _, _, infos = rollout(env, 7, 200)
        per[conf] = float(np.mean([r["ns_dmax"] for r in infos if "ns_dmax" in r]))
        env.close()
    check("per_actuator_disturbance_rises_with_N_until_saturation",
          per["2x4"] < per["4x2"],
          "mean max|d|: 2x4=%.4f -> 4x2=%.4f; 8x1=%.4f (saturated -- the only pair "
          "inside a 4x2 agent is its own hip-ankle, and with the machine's own "
          "operator that path is exactly zero).  All three are runnable configs "
          "with the identical dial and basis."
          % (per["2x4"], per["4x2"], per["8x1"]))

    # ------------------------------------------------------------------ obs tail
    aug = make(ns_severity=1.0, ns_observe_residual=1)
    aug.seed(1)
    ob, _, _ = aug.reset()
    base = make(ns_severity=1.0)
    ob0, _, _ = base.reset()
    check("observe_residual_adds_exactly_one_obs_dim",
          np.asarray(ob[0]).shape[0] == np.asarray(ob0[0]).shape[0] + 1,
          "%d -> %d" % (np.asarray(ob0[0]).shape[0], np.asarray(ob[0]).shape[0]))
    aug.close()
    base.close()

    if FAILS:
        print("")
        print("FAILED %d smoke check(s): %s" % (len(FAILS), FAILS))
        sys.exit(1)
    print("")
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
