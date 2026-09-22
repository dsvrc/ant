"""The severity layer -- a MIXIN, below the method and above the host.

`PACT_NS_SPEC` NS-3.1: the dial sits below the method in the class hierarchy and
is read from the TASK configuration, never from a method's own block.

    MujocoMulti                        stock HARL host, ZERO edits
      +-- AntNsEnv(SeverityMixin, ..)  THIS FILE.  Every arm gets it.
            +-- (the compensator lives INSIDE the layer: II.6's invertible row)

THE HOOK, AND WHY IT IS *BELOW* THE REWARD
--------------------------------------------------------------------------------
``do_simulation`` is the single place every MuJoCo host reads an action.  gym's
``AntEnv.step(a)`` does, in this order:

    xposbefore ; do_simulation(a, frame_skip) ; xposafter
    forward_reward = (xposafter - xposbefore) / dt
    ctrl_cost      = 0.5 * |a|^2                 <-- the POLICY's own command
    contact_cost, survive_reward, done

so wrapping ``do_simulation`` puts the disturbance on the ACTUATORS while the
reward function keeps being computed from the torque the POLICY asked for.  That
is NS-1.4 exactly: the reward function is byte-for-byte the host's own, nothing
is subtracted anywhere, and the ant earns less only because it physically walked
worse.  Handing the disturbed torque to ``step`` instead would feed the
disturbance straight into ``ctrl_cost`` -- a penalty term in disguise, and the
one thing NS-1.4 forbids.

The same gap is what makes the compensation free of any objective change: the
feed-forward is applied below ``ctrl_cost`` too, so with a correct estimate the
executed torque, the reward and the whole trajectory are the sigma = 0 ones,
exactly.  That is the conjugacy claim, and ``smoke.py`` asserts it bit for bit
rather than arguing it.  What it does NOT model is the extra current a real
feed-forward draws; the relief valve ``ns_corr_clip`` and the actuator's own
range are what bound it, and the README states the limitation.

WHAT THE HOST GAINS: NOTHING
--------------------------------------------------------------------------------
No file of the host is edited -- not ``mujoco_multi.py``, not gym's ``ant.py``.
``do_simulation`` is re-bound on the instance the layer owns, so a stock
checkout runs the stock task and ``ns_on: 0`` is the stock host byte for byte.

NS-3.2 -- the records are harmed because the TRAJECTORY is: the policy trains on
the actions it commanded and the states that actually followed.
NS-3.3 -- FAIL LOUDLY: ``close()`` prints the counters and refuses to describe a
sigma > 0 run as a severity arm if the dial never produced a disturbance.
NS-3.4 -- the CLOCK persists across episodes and is de-phased across rollout
threads, so a batch is a cycle average rather than one phase of it.
"""

import numpy as np
from gym.spaces import Box

from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

from .channel import PactConfig, TrunkChannel
from .coupling import Coupling
from .driver import DialParams, ThermalDriver
#: kept mujoco-free so ``check_plumbing.py`` can read it without a simulator
from .keys import NS_KWARGS
from .structure import (N_JOINTS, REFERENCE_PARTITION, describe, kernel_source,
                        model_inverse_inertia,
                        verify_against_model, verify_host_is_stock,
                        verify_operator_against_model)

__all__ = ["SeverityMixin", "NS_KWARGS", "make_ant_ns_env"]

#: Scenarios ant_ns declares a structure for.  Anything else must run with
#: ``ns_on: 0`` -- a layer that silently did nothing would be far worse.
SUPPORTED = ("Ant-v2", "Ant-v4")


def _b(v, default=False):
    if v is None:
        return bool(default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(int(v))


class SeverityMixin(object):
    """The dial and the channel.  Mix in ABOVE ``MujocoMulti``."""

    def __init__(self, batch_size=None, **kwargs):
        ea = dict(kwargs.get("env_args") or {})
        raw = {k: ea.pop(k) for k in list(ea.keys()) if str(k).startswith("ns_")}
        unknown = sorted(k for k in raw if k not in NS_KWARGS)
        if unknown:
            raise KeyError("unknown ns_* keys in the task config: %s -- a silently "
                           "ignored knob is a rigged knob; add it to keys.NS_KWARGS or "
                           "remove it" % unknown)
        kwargs = dict(kwargs)
        kwargs["env_args"] = ea
        super(SeverityMixin, self).__init__(batch_size=batch_size, **kwargs)

        self.ns_on = _b(raw.get("ns_on", 1), True)
        if not self.ns_on:
            self.chan = None
            self._observe_residual = False
            print("[ANT-NS] ns_on=0 -- the stock host, byte for byte (B0 reference run)")
            return
        if str(self.scenario) not in SUPPORTED:
            raise ValueError(
                "ant_ns declares a body structure for %s; got scenario=%r.  Run it with "
                "ns_on: 0, or add the structure to structure.py." % (SUPPORTED, self.scenario))

        # ---- the dial, from the TASK config (NS-3.1) -----------------------------
        self.ns = DialParams(
            severity=float(raw.get("ns_severity", 1.0)),
            period=int(raw.get("ns_period", 20000)),
            warm_fraction=float(raw.get("ns_warm_fraction", 0.5)),
            loss_at_sigma1=float(raw.get("ns_loss_at_sigma1", 0.14)),
            mean_preserving=_b(raw.get("ns_mean_preserving", 0)),
            rho=float(raw.get("ns_rho", 0.8)),
            length_scale=float(raw.get("ns_length_scale", 0.25)),
            recv_spread=float(raw.get("ns_recv_spread", 0.35)),
            send_hh=float(raw.get("ns_send_hh", 1.5)),
            send_aa=float(raw.get("ns_send_aa", 0.9)),
            send_cross=float(raw.get("ns_send_cross", 0.6)),
            y_clip=float(raw.get("ns_y_clip", 10.0)),
            corr_clip=float(raw.get("ns_corr_clip", 0.5)),
            direct=_b(raw.get("ns_direct", 0)),
        )
        self.driver = ThermalDriver(self.ns)
        self.driver.certify()                               # NS-2.x, in every process
        self.coupling = Coupling(self.agent_conf, self.ns, self.driver.send)
        assert self.coupling.n == self.n_agents, (
            "structure.PARTITIONS says %s has %d agents, the host built %d"
            % (self.agent_conf, self.coupling.n, self.n_agents))

        # ---- the compensator (PACT family only; a baseline never sets ns_pact) ---
        trust = raw.get("ns_trust", "off")
        # YAML 1.1 reads a bare `off` as False; a CLI `--ns_trust off` is eval'd to
        # the string.  Both mean the same arm.
        trust = "off" if trust in (False, 0, None) else str(trust).lower()
        assert trust in ("off", "fixed"), (
            "ns_trust must be 'off' or 'fixed' on this channel; learned trust (P-6.2) "
            "is not implementable when the correction is applied below the policy -- "
            "see channel.py")
        self.pact = PactConfig(
            enabled=_b(raw.get("ns_pact", 0)),
            trust=trust,
            g_fixed=float(raw.get("ns_g_fixed", 0.9)),
            oracle=_b(raw.get("ns_oracle", 0)),
            intercept_only=_b(raw.get("ns_intercept_only", 0)),
            mu=float(raw.get("ns_mu", 0.99)),
            p0=float(raw.get("ns_p0", 10.0)),
            warmup=int(raw.get("ns_warmup", 50)),
            p_trace_max=float(raw.get("ns_p_trace_max", 100.0)),
        )

        # ---- II.9 gates 1, 3 and 4 -- all abort, none warn ----------------------
        #  Gate 0 first, because it invalidates everything downstream: is the
        #  HOST itself already disturbed?  A patched gym `ant.py` (this repo has
        #  shipped one) would make `ns_severity: 0` something other than the
        #  stock task, and every arm would be measured against a disturbed
        #  baseline -- plausible numbers, wrong experiment.
        allow_patched = _b(raw.get("ns_allow_patched_host", 0))
        try:
            host_note = verify_host_is_stock(self.env)
        except AssertionError:
            if not allow_patched:
                raise
            host_note = ("host: PATCHED, and ns_allow_patched_host=1 -- THIS RUN IS "
                         "NOT REPORTABLE AS A SEVERITY ARM")
            print("[ANT-NS][REFUSE] " + host_note)
        gates = [host_note, verify_against_model(self.env), self.coupling.verify()]
        op_note = verify_operator_against_model(self.env)
        if op_note is not None:
            gates.append(op_note)

        ctrl_range = np.abs(np.asarray(self.env.model.actuator_ctrlrange,
                                       dtype=np.float64)).max(axis=1)
        assert ctrl_range.shape[0] == N_JOINTS, (
            "the model exposes %d actuators, ant_ns declares %d"
            % (ctrl_range.shape[0], N_JOINTS))

        ref, scale = self.coupling.geometric_reference()
        computed = self.coupling.load_norm()
        committed = raw.get("ns_load_norm", None)
        if committed in ("", "~", "None", "null"):
            committed = None
        load_norm = float(committed) if committed is not None else computed
        self.chan = TrunkChannel(self.ns, self.driver, self.coupling, self.pact,
                                 load_norm=load_norm, ref=ref, scale=scale,
                                 clock0=int(raw.get("ns_phase0", 0)),
                                 ctrl_range=ctrl_range)

        # ---- optional ablation: the policy is handed its own residual -----------
        self._observe_residual = _b(raw.get("ns_observe_residual", 0))
        if self._observe_residual:
            self.observation_space = [
                Box(low=-np.inf, high=np.inf, shape=(int(sp.shape[0]) + 1,),
                    dtype=np.float32)
                for sp in self.observation_space
            ]

        self._wrap_do_simulation()
        self._banner(gates, computed, committed, ref, scale)

    # ------------------------------------------------------------------ the hook
    def _wrap_do_simulation(self):
        """Re-bind ``do_simulation`` on the raw gym env this layer owns.

        The host's own file is untouched; the substitution lives on the instance,
        which is why a stock checkout still runs the stock task.
        """
        env = self.env
        original = env.do_simulation
        chan = self.chan

        def _patched(ctrl, n_frames):
            return original(chan.step(np.asarray(ctrl, dtype=np.float64).reshape(-1)),
                            n_frames)

        env.do_simulation = _patched
        self._do_simulation_original = original

    def _banner(self, gates, computed, committed, ref, scale):
        for g in gates:
            print("[ANT-NS] gate  " + g)
        print("[ANT-NS] " + describe(self.ns.length_scale))
        print(self.driver.banner())
        print(self.coupling.banner(self.chan.load_norm, ref, scale))
        if committed is not None and abs(float(committed) - computed) > 0.05 * max(computed, 1e-9):
            if str(self.agent_conf) == REFERENCE_PARTITION:
                #  On the partition the committed value BELONGS to, a mismatch can
                #  only mean the operator changed underneath it -- and a stale
                #  load_norm silently rescales the whole dial (measured: 16x, which
                #  put |d| at 13.3 of a torque range of 1).  Abort.
                raise AssertionError(
                    "STALE ns_load_norm: the committed value is %.6f but %s -- the "
                    "partition it belongs to -- now computes %.6f from the structure "
                    "in force (%s).  The operator changed; sigma no longer means what "
                    "the config says.  FIX: re-run\n"
                    "    python -m harl.envs.mamujoco.ant_ns.ceiling --out "
                    "harl/envs/mamujoco/ant_ns/ceiling.json\n"
                    "and copy the load_norm it prints into mamujoco_ns.yaml."
                    % (float(committed), REFERENCE_PARTITION, computed,
                       kernel_source()[0]))
            print("[ANT-NS] committed ns_load_norm=%.6f (the %s reference); this "
                  "partition's own reference is %.6f.  EXPECTED -- holding the "
                  "reference fixed is what makes the N-scaling visible rather than "
                  "normalising it away."
                  % (float(committed), REFERENCE_PARTITION, computed))
        src, note = kernel_source()
        print("[ANT-NS] transmission structure: %s -- %s" % (src.upper(), note))
        Minv = model_inverse_inertia(self.env)
        if Minv is None:
            print("[ANT-NS] the installed mujoco binding exposes no dense mass matrix; "
                  "the declared operator could not be compared with the model's own "
                  "inverse inertia")
        else:
            off = ~np.eye(N_JOINTS, dtype=bool)
            a_, b_ = self.coupling.kappa[off], np.abs(Minv)[off]
            corr = (float(np.corrcoef(a_, b_)[0, 1])
                    if a_.std() > 0 and b_.std() > 0 else float("nan"))
            msg = ("[ANT-NS] kappa vs the model's OWN inverse inertia at the running "
                   "pose: corr = %+.3f" % corr)
            if src == "surrogate" and corr < 0.5:
                msg += ("   <-- the geometric surrogate has the right support and "
                        "ordering but NOT the machine's shape.  Run dump_operator.py "
                        "to use the model's own |M^-1| instead, or say in the paper "
                        "that W is a declared transmission model.")
            print(msg)
        print("[ANT-NS] sigma=%.3f on=%d direct(B)=%d | pact=%d trust=%s g=%.2f oracle=%d "
              "intercept_only=%d mu=%.4f p0=%.1f warmup=%d | load_norm %s | clock0=%d | "
              "observe_residual=%d"
              % (self.ns.severity, int(self.ns_on), int(self.ns.direct),
                 int(self.pact.enabled), self.pact.trust, self.pact.g_fixed,
                 int(self.pact.oracle), int(self.pact.intercept_only), self.pact.mu,
                 self.pact.p0, self.pact.warmup,
                 "COMMITTED" if committed is not None else "computed",
                 self.chan.clock, int(self._observe_residual)))

    # ------------------------------------------------------------------ lifecycle
    def _aug(self, obs):
        if not self._observe_residual:
            return obs
        out = []
        for i in range(self.n_agents):
            y = self.chan.y[i]
            tail = np.array([0.0 if not np.isfinite(y) else float(y)], dtype=np.float32)
            out.append(np.concatenate([np.asarray(obs[i], dtype=np.float32), tail]))
        return out

    def reset(self, **kwargs):
        out = super(SeverityMixin, self).reset(**kwargs)
        if self.chan is not None:
            # NS-3.4: the clock and the estimators persist.  The drivetrain does
            # not cool because a training episode ended.
            self.chan.reset_episode()
            obs, state, avail = out
            return self._aug(obs), state, avail
        return out

    def step(self, actions):
        obs, state, rew, done, infos, avail = super(SeverityMixin, self).step(actions)
        if self.chan is None:
            return obs, state, rew, done, infos, avail
        out = []
        for i in range(self.n_agents):
            d = dict(infos[i]) if isinstance(infos[i], dict) else {}
            d.update(self.chan.info(i))
            out.append(d)
        return self._aug(obs), state, rew, done, out, avail

    def close(self):
        try:
            if self.chan is not None:
                self.chan.close_report()
        finally:
            super(SeverityMixin, self).close()


def make_ant_ns_env(env_args):
    """Build ``AntNsEnv`` -- the host with the severity layer mixed in."""

    class AntNsEnv(SeverityMixin, MujocoMulti):
        pass

    return AntNsEnv(env_args=env_args)
