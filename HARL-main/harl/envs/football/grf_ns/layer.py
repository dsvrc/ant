"""The severity layer -- a MIXIN, below the method and above the host.

`PACT_NS_SPEC` NS-3.1: the dial sits below the method in the class hierarchy and
is read from the TASK configuration, never from a method's own block.

    FootballEnv                          stock HARL host, untouched
      +-- FootballNsEnv(SeverityMixin, ..)  THIS FILE.  Every arm gets it.
            +-- (the compensator lives INSIDE the layer: II.6's invertible row)

The hook is ``step``: the single place every host reads an action.  The layer
substitutes the agents' commanded headings with the headings the surface lets
them run, calls the host's own ``step`` with those, and hands back the host's own
``obs``, ``reward``, ``done`` and ``available_actions`` untouched.  Only ``info``
gains keys (the II.10 panel).

NS-1.4 -- THE REWARD FUNCTION IS UNTOUCHED, BYTE FOR BYTE.  GRF pays
``scoring,checkpoints`` exactly as before; the team simply scores less because
its runs went a few degrees off line.  No penalty is subtracted anywhere.

NS-3.2 -- the records are harmed because the TRAJECTORY is: the policy is
trained on the actions it commanded and the states that actually followed.  The
eval win rate is read from the same env.

NS-3.3 -- FAIL LOUDLY.  ``close()`` prints the dial-live / harmed / rotated
counters and refuses to describe a sigma > 0 run as a severity arm if the dial
never produced a non-zero disturbance.

NS-3.4 -- the CLOCK persists across episodes.  The pitch does not dry because a
training episode ended.  Across rollout threads the clock is de-phased so a
batch is a cycle average rather than one phase of it (see ``__init__``).

WHO THE AGENTS ARE -- controller slots, not players
--------------------------------------------------------------------------------
GRF exposes ``n`` controller SLOTS; which player a slot drives is the engine's
decision (``active`` in the raw observation) and it can change: at kick-off the
engine may hand a slot the goalkeeper and, on the first tick, auto-switch it
onto the designated (ball) player, and it switches again whenever the ball
reaches an uncontrolled teammate.  The layer therefore keys every per-agent
quantity on the SLOT, re-reads each slot's position every step, and when a slot
changes player it drops that slot's lane state (``SwerveChannel.reset_slot``);
switches are counted in the panel.  The references (``load_norm``, centring)
come from the scenario's DECLARED attacker geometry (``ceiling.SPAWN``), not
from whatever the kick-off frame happens to show, and the layer checks that
geometry against the engine once, after the first tick, and warns on drift.

WHY THE HEADING, AND NOT THE PACE
--------------------------------------------------------------------------------
Sprint is a two-level effort and a well-trained attacker sprints nearly always,
so a pace derate has no headroom for an inverse: it would put this instance in
the bounded cell.  The eight headings are a group -- every rotation has an
inverse and nothing saturates -- and a swerve round a teammate IS a heading
error, which is what makes the compensation claim available and honest here.
"""

import numpy as np
from gym.spaces import Box

from harl.envs.football.football_env import FootballEnv

from .actions import N_ACTIONS, N_DIRS, verify_action_table
from .ceiling import SPAWN
from .channel import PactConfig, SwerveChannel
from .coupling import Coupling
from .driver import DialParams, PitchDriver
#: Every key the layer reads from the task config, kept gfootball-free in
#: ``keys.py`` so ``check_plumbing.py`` can check it against ``football.yaml``.
from .keys import NS_KWARGS

__all__ = ["SeverityMixin", "NS_KWARGS", "make_football_ns_env"]

#: Extra single-agent academy scenarios, so the N = 1 identity can be checked in
#: the real engine (``smoke.py``).  FootballEnv's own table is left untouched.
EXTRA_NUM_AGENTS = {
    "academy_empty_goal_close": 1,
    "academy_empty_goal": 1,
    "academy_run_to_score": 1,
    "academy_run_to_score_with_keeper": 1,
}

#: The engine places players a little off the scenario file's coordinates
#: (~1 %); a larger disagreement between the declared attacker geometry and what
#: the engine shows after the first tick means the yaml belongs to another
#: scenario, and sigma no longer means what the paper says.
GEOMETRY_TOL = 0.10


def _b(v, default=False):
    if v is None:
        return bool(default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(int(v))


class SeverityMixin(object):
    """The dial and the channel.  Mix in ABOVE ``FootballEnv``."""

    def __init__(self, args):
        a = dict(args or {})
        raw = {k: a.pop(k) for k in list(a.keys()) if str(k).startswith("ns_")}
        unknown = sorted(k for k in raw if k not in NS_KWARGS)
        if unknown:
            raise KeyError("unknown ns_* keys in the task config: %s -- a silently ignored "
                           "knob is a rigged knob; add it to keys.NS_KWARGS or remove it"
                           % unknown)
        # gfootball's create_environment rejects unknown kwargs, so the host must
        # never see an ns_* key.
        from harl.envs.football import football_env as _fe
        for k, v in EXTRA_NUM_AGENTS.items():
            _fe.env_num_agents.setdefault(k, v)
        super(SeverityMixin, self).__init__(a)
        self._env_name = str(a.get("env_name", ""))

        # ---- the dial, from the TASK config (NS-3.1) -----------------------------
        self.ns_on = _b(raw.get("ns_on", 1), True)
        self.ns = DialParams(
            severity=float(raw.get("ns_severity", 1.0)),
            period=int(raw.get("ns_period", 12000)),
            wet_fraction=float(raw.get("ns_wet_fraction", 0.5)),
            loss_at_sigma1=float(raw.get("ns_loss_at_sigma1", 0.14)),
            mean_preserving=_b(raw.get("ns_mean_preserving", 0)),
            rho=float(raw.get("ns_rho", 0.6)),
            kernel_lambda=float(raw.get("ns_kernel_lambda", 0.12)),
            recv_ball=float(raw.get("ns_recv_ball", 1.5)),
            send_front=float(raw.get("ns_send_front", 1.4)),
            send_flank=float(raw.get("ns_send_flank", 0.6)),
            y_clip=float(raw.get("ns_y_clip", 10.0)),
            corr_clip=float(raw.get("ns_corr_clip", 1.0)),
            direct=_b(raw.get("ns_direct", 0)),
        )
        self.driver = PitchDriver(self.ns)
        self.driver.certify()                                   # NS-2.x, every process
        self.coupling = Coupling(self.n_agents, self.ns, self.driver.send)

        # ---- II.9 gates 1, 3 and 4 -- all abort, none warn ------------------------
        self._gate_msgs = [verify_action_table()]
        assert int(self.action_space[0].n) == N_ACTIONS, (
            "GATE 4 FAILED: the host exposes %d actions, the layer assumes GRF's "
            "default set of %d" % (int(self.action_space[0].n), N_ACTIONS))
        rng = np.random.RandomState(0)
        synth = np.stack([np.column_stack([rng.uniform(0.3, 1.0, self.n_agents),
                                           rng.uniform(-0.35, 0.35, self.n_agents)])
                          for _ in range(4)])
        self._gate_msgs.append(self.coupling.verify(synth))

        # ---- the compensator (PACT family only; a baseline never sets ns_pact) ----
        trust = raw.get("ns_trust", "off")
        # YAML 1.1 reads a bare `off` as False; a CLI `--ns_trust off` is eval'd to
        # the string.  Both mean the same arm.
        trust = "off" if trust in (False, 0, None) else str(trust).lower()
        assert trust in ("off", "fixed"), (
            "ns_trust must be 'off' or 'fixed' on this channel; learned trust "
            "(P-6.2) is not implementable when the correction is a transform of "
            "the sampled action -- see channel.py")
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
        self.chan = SwerveChannel(self.n_agents, self.ns, self.driver, self.coupling,
                                  self.pact, clock0=int(raw.get("ns_phase0", 0)))
        self._committed_load_norm = raw.get("ns_load_norm", None)
        if self._committed_load_norm in ("", "~", "None", "null"):
            self._committed_load_norm = None

        # ---- optional ablation: the policy is handed its own residual ------------
        self._observe_residual = _b(raw.get("ns_observe_residual", 0))
        if self._observe_residual:
            self.observation_space = [
                Box(low=-np.inf, high=np.inf, shape=(int(sp.shape[0]) + 1,), dtype=np.float32)
                for sp in self.observation_space
            ]

        self._raw = None
        self._last_pos = np.zeros((self.n_agents, 2))
        self._active_prev = np.full(self.n_agents, -1, dtype=np.int64)
        self._steps_total = 0
        self._geometry_checked = False
        self._banner_done = False

    # ------------------------------------------------------------------ raw obs
    def _raw_obs(self):
        return self.env.unwrapped.observation()

    def _squad(self, raw):
        """Per SLOT: the position of the player it drives, who has the ball, the
        engine's sticky heading, and the player index -- all public, all from
        the last observation."""
        n = self.n_agents
        pos = self._last_pos.copy()
        owner = -1
        engine_head = np.full(n, -1, dtype=np.int64)
        active = np.full(n, -1, dtype=np.int64)
        if raw is None:
            return pos, owner, engine_head, active
        team = raw[0].get("ball_owned_team", -1)
        bp = raw[0].get("ball_owned_player", -1)
        for i in range(n):
            o = raw[i]
            act = int(o.get("active", -1))
            active[i] = act
            if act >= 0:
                pos[i] = np.asarray(o["left_team"][act], dtype=np.float64)
                if team == 0 and bp == act:
                    owner = i
            st = np.asarray(o.get("sticky_actions", np.zeros(10)))[:N_DIRS]
            if st.size == N_DIRS and st.any():
                engine_head[i] = int(np.argmax(st))
        self._last_pos = pos
        return pos, owner, engine_head, active

    # ------------------------------------------------------------------ references
    def _declared_geometry(self):
        """The scenario's declared attacker geometry, if this instance has one."""
        tbl = SPAWN.get(self._env_name)
        if tbl is not None and tbl.shape[0] == self.n_agents:
            return np.asarray(tbl, dtype=np.float64)
        return None

    def _ensure_references(self, pos_observed, owner=-1):
        if self.chan.references_ready():
            return
        decl = self._declared_geometry()
        src = "declared table (ceiling.SPAWN)" if decl is not None else "kick-off frame"
        pos_ref = (decl if decl is not None else pos_observed)[None, :, :]
        ref, scale = self.coupling.geometric_reference(pos_ref)
        if self.n_agents == 1:
            # a lone player has NO lane load (gate 3), so there is nothing to
            # normalise by; the (C) disturbance is identically zero and the (B)
            # control never divides by this.  Any positive constant will do.
            computed = 1.0
        else:
            computed = self.coupling.load_norm(pos_ref)
        if self._committed_load_norm is not None:
            load_norm = float(self._committed_load_norm)
            if abs(load_norm - computed) > GEOMETRY_TOL * max(computed, 1e-9):
                print("[GRF-NS][WARN] committed ns_load_norm=%.6f but the %s gives %.6f "
                      "-- the yaml value belongs to a different scenario; sigma does not "
                      "mean what the paper says it means until this is fixed."
                      % (load_norm, src, computed))
        else:
            load_norm = computed
        self.chan.set_references(load_norm, ref, scale)
        if not self._banner_done:
            self._banner_done = True
            for m in self._gate_msgs:
                print("[GRF-NS] gate  " + m)
            print(self.driver.banner())
            print(self.coupling.banner(load_norm, ref, scale))
            print("[GRF-NS] references from the %s: %s" % (src, np.round(pos_ref[0], 3).tolist()))
            recv0 = np.ones(self.n_agents)
            if 0 <= int(owner) < self.n_agents:
                recv0[int(owner)] = self.ns.recv_ball
            st = self.coupling.operator_stats(pos_ref[0], np.full(self.n_agents, 4), recv0)
            print("[GRF-NS] operator on the reference geometry (all heading right, ball owner "
                  "recv): zero_diag=%s spread=%.3f ratio=%.1fx asym=%.3f links=%d"
                  % (st["diag_max"] == 0.0, st["spread"], st["ratio"], st["asymmetry"],
                     st["n_links"]))
            print("[GRF-NS] sigma=%.3f on=%d direct(B)=%d | pact=%d trust=%s g=%.2f "
                  "oracle=%d intercept_only=%d mu=%.4f p0=%.1f warmup=%d | "
                  "load_norm %s | clock0=%d | observe_residual=%d"
                  % (self.ns.severity, int(self.ns_on), int(self.ns.direct),
                     int(self.pact.enabled), self.pact.trust, self.pact.g_fixed,
                     int(self.pact.oracle), int(self.pact.intercept_only), self.pact.mu,
                     self.pact.p0, self.pact.warmup,
                     "COMMITTED" if self._committed_load_norm is not None else "computed",
                     self.chan.clock, int(self._observe_residual)))

    def _check_geometry(self, pos, active):
        """Once per process, after the engine's first tick (when controller slots
        have settled onto their players): does the engine's controlled geometry
        agree with the reference in force?"""
        self._geometry_checked = True
        if self.n_agents == 1:
            return
        observed = self.coupling.load_norm(pos[None, :, :])
        in_force = self.chan.load_norm
        rel = abs(observed - in_force) / max(in_force, 1e-9)
        print("[GRF-NS] engine geometry after the first tick: slots drive players %s at %s; "
              "load_norm there %.6f vs %.6f in force (%.0f%% off)"
              % (active.tolist(), np.round(pos, 3).tolist(), observed, in_force, 100 * rel))
        if rel > GEOMETRY_TOL:
            print("[GRF-NS][WARN] the controlled players' geometry after the first tick does "
                  "not match the reference the dial is normalised to.  Either the engine "
                  "assigns slots differently from ceiling.SPAWN's assumption or the yaml "
                  "belongs to another scenario -- sigma does not mean what the paper says.")

    # ------------------------------------------------------------------ obs tail
    def _aug(self, obs):
        if not self._observe_residual:
            return obs
        out = []
        for i in range(self.n_agents):
            y = self.chan.y[i]
            tail = np.array([0.0 if not np.isfinite(y) else float(y)], dtype=np.float32)
            out.append(np.concatenate([np.asarray(obs[i], dtype=np.float32), tail]))
        return out

    # ------------------------------------------------------------------ lifecycle
    def reset(self):
        obs, state, avail = super(SeverityMixin, self).reset()
        self._raw = self._raw_obs()
        pos, owner, _, active = self._squad(self._raw)
        self._ensure_references(pos, owner)
        self.chan.reset_episode()                 # clock and estimators persist
        self._active_prev = active
        return self._aug(obs), state, avail

    def step(self, actions):
        a = np.asarray(actions).reshape(-1).astype(np.int64)
        assert a.size == self.n_agents, "expected one action per controlled player"
        pos, owner, engine_head, active = self._squad(self._raw)
        # a slot that now drives a different player drops the old player's lane
        # state; its heading belief is re-seeded from the engine's sticky state
        for i in range(self.n_agents):
            if active[i] >= 0 and self._active_prev[i] >= 0 and active[i] != self._active_prev[i]:
                self.chan.reset_slot(i, engine_head[i])
        self._active_prev = active
        if self._steps_total == 1 and not self._geometry_checked:
            self._check_geometry(pos, active)
        if not self.ns_on:
            exec_a = a
        else:
            exec_a = self.chan.step(a, pos, owner)
        obs, state, rew, done, info, avail = super(SeverityMixin, self).step(
            np.asarray(exec_a, dtype=np.int64).reshape(self.n_agents, 1))
        self._raw = self._raw_obs()
        self._steps_total += 1
        infos = []
        for i in range(self.n_agents):
            d = dict(info[i]) if isinstance(info[i], dict) else {}
            d["ns_commanded"] = int(a[i])
            d["ns_executed"] = int(exec_a[i])
            d["ns_active"] = int(active[i])
            d.update(self.chan.info(i))
            infos.append(d)
        return self._aug(obs), state, rew, done, infos, avail

    def close(self):
        try:
            if self.ns_on:
                self.chan.close_report()
        finally:
            super(SeverityMixin, self).close()


def make_football_ns_env(args):
    """Build ``FootballNsEnv`` -- the host with the severity layer mixed in."""

    class FootballNsEnv(SeverityMixin, FootballEnv):
        pass

    return FootballNsEnv(args)
