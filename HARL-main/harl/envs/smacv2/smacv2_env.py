from __future__ import absolute_import, division, print_function

import time
from os import replace

import numpy as np
from absl import logging
from smacv2.env import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

logging.set_verbosity(logging.DEBUG)
import math
import os
import os.path as osp
from pathlib import Path
import yaml

from gym.spaces import Box, Discrete


# ===========================================================================
# Non-Stationarity  ::  Concussion-Coupled Wake Displacement (CWD)
# ---------------------------------------------------------------------------
# A category-C, dynamics-only, oracle-separable non-stationarity for SMACv2.
# Same family template as PCR (Ant) and SND (SMAC), a *fresh* instance of it.
#
# Story: in a heavy firefight, sustained weapons discharge and impacts throw off
# concussive overpressure and kick up debris.  A unit maneuvering near where its
# squadmates are trading fire gets physically buffeted -- shoved AWAY from the
# locus of the firefight (overpressure radiates outward, closer blasts hit
# harder).  An exogenous BOMBARDMENT-TEMPO driver A2(t) (munitions expenditure
# ramping over the engagement, dropping abruptly at each resupply / lull) gates
# how hard the buffeting is.  The shove d_i is ADDED to the unit's move target
# inside the env (delivered = commanded + d_i); the reward is left exactly as in
# stock SMACv2.  A controller that knows d_i can brace and counter-step; a blind
# one is knocked off its micro, scattered out of position, and cut down.
#
#   d_i  <-  RHO * d_i  +  (1 - RHO) * A2(t) * SEVERITY * sum_{j != i, firing} w_ij * u_ij
#            \_ impulse _/          \_ exog _/                     \__ the OTHER units __/
#
#   u_ij = unit vector (p_i - p_j) pointing away from firing peer j
#   w_ij = 1 / (1 + ||p_i - p_j|| / R)   (closer firefights buffet harder)
#
#   * the driver only MULTIPLIES the cross-agent sum -> empty sum at N=1 => stock SMACv2
#   * the driver is a function of a global clock      -> survives frozen partners
#   * the sum excludes i and needs FIRING peers        -> individually exogenous,
#                                                          collectively endogenous
#
# This is a genuinely different instance of the family idea than SND: the fed
# quantity is peers' COMBAT (firing), not their movement; the geometry is a
# range-weighted radial push from relative positions, not a common move-flow;
# the accumulator is IMPULSIVE (short memory), not slowly-building; and the
# driver is an asymmetric slow-build / fast-collapse ramp, not a raised cosine.
#
# One dial:  SMACV2_CWD_SEVERITY.  Everything else is a fixed internal constant.
# Oracle proof:  SMACV2_CWD_ORACLE=1 appends d_i to the observation (and the full
# drift to the centralized state) -- the runnable recoverability existence proof.
#
# PHASE-1 (PACT pipeline) HOOKS -- additive, all OFF by default, never change the
# training semantics of a blind/oracle run:
#   * knobs are now resolved PER INSTANCE (env_args -> env-var -> module default)
#     so a severity sweep can hold ONE env open and change c between cells, and so
#     parallel envs can run different severities.  (Previously the module-level
#     value was read directly and the documented env-var was silently ignored.)
#   * SMACV2_CWD_FREEZE / env_args["cwd_freeze"] holds the exogenous driver A(t)
#     at a constant value (the PEAK is 1.0) -> a stationary game at effective
#     coupling c = A*severity.  This is the Phase-1 "freeze knob".
#   * info now also carries the full per-unit shove that was applied this step
#     ("cwd_d_applied") and the one the next step will face ("cwd_d_next"), for
#     the scripted controller and its residual telemetry.  Diagnostics only.
# The scripted compensation itself lives in the SMACv2ProbeEnv subclass
# (harl/envs/smacv2/phase1/probe_env.py), which overrides step()/_install_cwd().
# ===========================================================================

# --- the ONE severity dial.  Default 0.0 == STOCK SMACv2, so a naked run (or the
#     shipped happo config) is the clean stationary baseline and never silently
#     carries the NS.  Every non-stationary experiment sets the severity EXPLICITLY
#     (in the config's env_args, or via $SMACV2_CWD_SEVERITY).  Resolved PER INSTANCE
#     in seed(), in priority order:
#         env_args["cwd_severity"]  ->  $SMACV2_CWD_SEVERITY  ->  this default.
#     (Historically a hard-coded module value of 5 was read directly and the
#      documented env-var was silently ignored; both are fixed here.) ---
_CWD_SEVERITY_DEFAULT = 0.0
# --- oracle ablation toggle: expose the hidden drift in obs/state (0/1) ---
_CWD_ORACLE_DEFAULT = 0
# --- Phase-1 freeze: hold the exogenous driver A(t) constant (PEAK = 1.0), so the
#     game is stationary at effective coupling c = A*severity.  None = live ramp. ---
_CWD_FREEZE_DEFAULT = None
# --- fixed internal constants (NOT knobs) ---
_CWD_P = 3000      # bombardment-tempo period in env-steps; persists across episodes
_CWD_RHO = 0.5     # concussive memory (impulsive leaky integrator); time const ~ 2 steps
_CWD_R = 5.0       # blast falloff radius (world units)
_CWD_DCAP = 2.0    # per-axis drift cap (= one move step): no teleport, keeps the drift
#                    inside the discrete action's coarse-cancellation budget (recoverable)
_CWD_EPS = 1e-6

_CWD_BANNER_SHOWN = False  # print the resolved CWD config once per process


def _cwd_driver(clock):
    """Exogenous bombardment tempo A2(t) in [0, 1]: slow convex escalation, abrupt drop.

    A munitions-expenditure story: intensity accelerates over most of the cycle,
    then collapses abruptly at a resupply / ceasefire.  Continuous in value (no
    value shocks), asymmetric, and deliberately a different functional form from
    both PCR's smoothstep and SND's raised cosine.  Hidden from the agents; only
    the (persisting) global clock drives it.
    """
    phase = (clock % _CWD_P) / _CWD_P
    c = 0.85  # fraction of the cycle spent escalating before the abrupt drop
    if phase < c:
        x = phase / c
        return x * x  # accelerating (convex) escalation
    return 1.0 - (phase - c) / (1.0 - c)  # abrupt linear collapse back to 0


def _cwd_cos_rows(a, b):
    """Mean per-row cosine between two (n,2) arrays over rows where both are non-zero.

    The Phase-2 gate (pipeline §VI): the computed waveform ``x2_i`` and the true
    shove ``d_i`` must be co-directional every step (>0.999).  d = c*x2 with c>=0
    by construction, so this is ~1 except on the DCAP-saturated rows (the small
    leak).  Rows where either is ~0 carry no direction and are skipped; if none
    qualify the step is trivially aligned (return 1.0).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    m = (na > 1e-8) & (nb > 1e-8)
    if not np.any(m):
        return 1.0
    cos = (a[m] * b[m]).sum(axis=1) / (na[m] * nb[m])
    return float(np.clip(cos, -1.0, 1.0).mean())


class SMACv2Env:
    def __init__(self, args):
        self.args = dict(args)
        self.map_config = self.load_map_config(args["map_name"])

    def step(self, actions):
        # The hidden drift for THIS step (self._cwd_d, set at the end of the previous
        # step / at reset) is applied inside self.env.step via the patched
        # get_agent_action.  The reward is the stock SMACv2 reward, untouched.
        d_applied = self._cwd_d.copy()  # the shove THIS step's move commands receive
        x2_applied = self._cwd_x2.copy()  # the waveform the policy SAW when it acted
        reward, terminated, info = self.env.step(actions)

        # Advance the campaign + the hidden drift for the NEXT step, using this
        # step's commanded actions and the post-step unit positions.
        self._cwd_advance(actions)

        obs, state = self._cwd_obs_state()
        rewards = [[reward]] * self.n_agents
        dones = [terminated] * self.n_agents
        if terminated:
            if self.env.env.timeouts > self.timeouts:
                assert (
                    self.env.env.timeouts - self.timeouts == 1
                ), "Change of timeouts unexpected."
                info["bad_transition"] = True
                self.timeouts = self.env.env.timeouts
        # CWD diagnostics (never used by the reward; for TensorBoard / calibration)
        info = dict(info)
        info["cwd_payload"] = self._cwd_payload  # A2(t), the exogenous bombardment driver
        info["cwd_load"] = self._cwd_load_mean  # mean |d| over units (calibration target)
        info["cwd_loadmax"] = self._cwd_load_max  # max |d| over units (saturation watch)
        info["cwd_d_applied"] = d_applied  # the (2N) shove applied to THIS step's moves
        info["cwd_d_next"] = self._cwd_d.copy()  # the shove the NEXT step will face
        if self.cwd_pact:
            # Phase-2 gate: the waveform the policy saw vs the shove it got. ~1 by
            # construction (d = c*x2); a drop flags an index/reset/timing bug.
            # Neutral pact_* keys: OnPolicyPactSmacRunner reads these for BOTH the
            # smacv2 CWD env and the smac SND env (one runner serves both).
            _cos = _cwd_cos_rows(x2_applied, d_applied)
            _x2l = float(np.abs(x2_applied).mean()) if x2_applied.size else 0.0
            info["cwd_pact_cos"] = _cos
            info["cwd_pact_x2load"] = _x2l
            info["pact_cos"] = _cos
            info["pact_x2load"] = _x2l
            info["pact_payload"] = self._cwd_payload
            info["pact_dload"] = self._cwd_load_mean
        infos = [info] * self.n_agents
        avail_actions = self.env.get_avail_actions()
        return obs, state, rewards, dones, infos, avail_actions

    def reset(self):
        self.env.reset()
        # CWD: units start each episode un-buffeted; the drift re-accumulates as the
        # team begins to engage.  The clock (campaign) is NOT reset.
        self._cwd_d = np.zeros((self.n_agents, 2), dtype=np.float32)
        self._cwd_x2 = np.zeros((self.n_agents, 2), dtype=np.float32)  # PACT waveform
        self._cwd_payload = 0.0
        self._cwd_load_mean = 0.0
        self._cwd_load_max = 0.0
        self._install_cwd(self.env.env)  # ensure the patch survives any env re-creation
        obs, state = self._cwd_obs_state()
        avail_actions = self.env.get_avail_actions()
        return obs, state, avail_actions

    def seed(self, seed):
        self.env = StarCraftCapabilityEnvWrapper(seed=seed, **self.map_config)
        env_info = self.env.get_env_info()
        n_actions = env_info["n_actions"]
        state_shape = env_info["state_shape"]
        obs_shape = env_info["obs_shape"]
        self.n_agents = env_info["n_agents"]
        self.timeouts = self.env.env.timeouts

        # --- resolve the CWD knobs for THIS env instance (env_args -> env-var ->
        #     module default).  Doing this per instance (not via module globals) is
        #     what makes the Phase-1 sweep and parallel-severity envs possible. ---
        self._cwd_resolve_knobs()
        self._cwd_banner()

        # --- Concussion-Coupled Wake Displacement (CWD) state ---
        self._cwd_clock = 0
        self._cwd_d = np.zeros((self.n_agents, 2), dtype=np.float32)
        self._cwd_payload = 0.0
        self._cwd_load_mean = 0.0
        self._cwd_load_max = 0.0
        # world units of one discrete move step (used by the Phase-1 scripted
        # controller); read off the underlying env, default 2 (standard SMAC).
        self._cwd_move_amount = float(getattr(self.env.env, "_move_amount", 2.0))
        self._cwd_x2 = np.zeros((self.n_agents, 2), dtype=np.float32)  # PACT waveform
        self._install_cwd(self.env.env)  # patch the delivered move target

        if self.cwd_oracle:
            obs_shape = obs_shape + 2  # d_i (2-vector) exposed to the actor
            state_shape = state_shape + 2 * self.n_agents  # full drift to the critic
        elif self.cwd_pact:
            # PACT soft variant (Phase 2): append the COMPUTED waveform x2_i -- the
            # env's leaky accumulator MINUS the hidden scalar c = A*severity
            # (x2 = d/c) -- so a recurrent policy can pre-empt the shove and learn c
            # (the phase) itself.  x2_i is decentralizable arithmetic (peers' firing
            # bits, one-step delayed, + their relative positions from own obs); the
            # TRUE d is never exposed here (that is the oracle).  Discrete actions
            # can't express a continuous re-aim, so we use the obs-feature soft
            # variant (pipeline Part V).  Floor property: if the policy ignores these
            # dims it is exactly blind.
            obs_shape = obs_shape + 3  # x2_i (2) + |x2_i| (1)
            state_shape = state_shape + 2 * self.n_agents  # stacked x2 to the critic
            if self.cwd_pact_ctde:
                state_shape = state_shape + 1  # + true driver A(t): CTDE, critic-only

        self.share_observation_space = self.repeat(
            Box(low=-np.inf, high=np.inf, shape=(state_shape,))
        )
        self.observation_space = self.repeat(
            Box(low=-np.inf, high=np.inf, shape=(obs_shape,))
        )
        self.action_space = self.repeat(Discrete(n_actions))

    # ------------------------------------------------------------------ CWD --
    @staticmethod
    def _cwd_env_num(name, default):
        """Read a numeric env-var; blank/missing/garbage -> default."""
        v = os.environ.get(name, None)
        if v is None or v == "":
            return default
        try:
            return float(v)
        except ValueError:
            return default

    def _cwd_resolve_knobs(self):
        """Resolve severity / oracle / freeze: env_args -> env-var -> module default.

        Priority is env_args first so a driver (the Phase-1 sweep, or a config)
        can pin a value regardless of the shell environment; the documented
        env-vars remain honoured for the plain train.py workflow.
        """
        a = getattr(self, "args", {}) or {}
        if a.get("cwd_severity", None) is not None:
            self.cwd_severity = float(a["cwd_severity"])
        else:
            self.cwd_severity = float(
                self._cwd_env_num("SMACV2_CWD_SEVERITY", _CWD_SEVERITY_DEFAULT)
            )
        if a.get("cwd_oracle", None) is not None:
            self.cwd_oracle = int(a["cwd_oracle"])
        else:
            self.cwd_oracle = int(
                self._cwd_env_num("SMACV2_CWD_ORACLE", _CWD_ORACLE_DEFAULT)
            )
        if a.get("cwd_freeze", None) is not None:
            self.cwd_freeze = float(a["cwd_freeze"])
        else:
            fv = os.environ.get("SMACV2_CWD_FREEZE", None)
            self.cwd_freeze = float(fv) if fv not in (None, "") else _CWD_FREEZE_DEFAULT
        # --- Phase-2 PACT: expose the COMPUTED waveform x2 (not the true d) ---
        if a.get("cwd_pact", None) is not None:
            self.cwd_pact = int(a["cwd_pact"])
        else:
            self.cwd_pact = int(self._cwd_env_num("SMACV2_CWD_PACT", 0))
        if a.get("cwd_pact_ctde", None) is not None:
            self.cwd_pact_ctde = int(a["cwd_pact_ctde"])
        else:
            self.cwd_pact_ctde = int(self._cwd_env_num("SMACV2_CWD_PACT_CTDE", 0))
        assert not (self.cwd_oracle and self.cwd_pact), (
            "CWD: oracle (true d) and pact (computed x2) fill the same obs slot; "
            "enable at most one."
        )

    def _cwd_banner(self):
        """Print the resolved CWD config once per process, so a training log makes
        the active severity / mode unmistakable (e.g. confirm an arm is at 1.5)."""
        global _CWD_BANNER_SHOWN
        if _CWD_BANNER_SHOWN:
            return
        _CWD_BANNER_SHOWN = True
        mode = "blind"
        if self.cwd_oracle:
            mode = "ORACLE (true d in obs)"
        elif self.cwd_pact:
            mode = "PACT (computed x2 in obs)" + (" +CTDE" if self.cwd_pact_ctde else "")
        print(
            f"[CWD] SMACv2 severity={self.cwd_severity} mode={mode} "
            f"freeze(A)={self.cwd_freeze}  (severity 0 == stock SMACv2)",
            flush=True,
        )

    def _cwd_driver_value(self):
        """A(t): the frozen constant if a freeze is set, else the live ramp.

        The peak of the live ramp is 1.0, so freeze=1.0 gives a stationary game at
        effective coupling c = severity (the Phase-1 peak slice).
        """
        if self.cwd_freeze is not None:
            return float(self.cwd_freeze)
        return _cwd_driver(self._cwd_clock)

    def _install_cwd(self, env):
        """Monkeypatch the underlying StarCraft2Env's action builder so that move
        commands are delivered to a biased world position.  We wrap the original
        method and only mutate the target of move commands, so we stay robust to
        the exact smacv2 version (no reimplementation of the action pipeline)."""
        if env is None or getattr(env, "_cwd_installed", False):
            return
        orig_get_agent_action = env.get_agent_action

        def patched(a_id, action):
            sc_action = orig_get_agent_action(a_id, action)
            if sc_action is None or self.cwd_severity == 0.0:
                return sc_action
            try:
                cmd = sc_action.action_raw.unit_command
                if cmd.HasField("target_world_space_pos"):
                    unit = env.get_unit_by_id(a_id)
                    if unit is not None and unit.health > 0:
                        shift = self._cwd_delivered_shift(a_id)
                        cmd.target_world_space_pos.x += float(shift[0])
                        cmd.target_world_space_pos.y += float(shift[1])
            except Exception:
                pass
            return sc_action

        env.get_agent_action = patched
        env._cwd_installed = True

    def _cwd_delivered_shift(self, a_id):
        """The offset actually added to unit ``a_id``'s move target this step.

        The base env adds the full hidden shove ``d_i``.  A subclass (the Phase-1
        probe running the *continuous* re-aim controller) overrides this to add
        only the uncompensated remainder ``(1 - beta) * d_i``.
        """
        return self._cwd_d[a_id]

    def _cwd_advance(self, actions):
        """Advance the bombardment campaign and the hidden per-unit shove d_i."""
        self._cwd_clock += 1
        A = self._cwd_driver_value()
        self._cwd_payload = A

        env = self.env.env
        acts = np.asarray(actions).reshape(-1).astype(int)
        n = self.n_agents
        n_no_attack = int(getattr(env, "n_actions_no_attack", 6))

        pos = np.zeros((n, 2), dtype=np.float32)
        alive = np.zeros(n, dtype=bool)
        firing = np.zeros(n, dtype=bool)
        for j in range(n):
            try:
                uj = env.get_unit_by_id(j)
            except Exception:
                uj = None
            if uj is not None and uj.health > 0:
                alive[j] = True
                pos[j, 0] = uj.pos.x
                pos[j, 1] = uj.pos.y
                # an attack / heal target action (>= n_no_attack) == engaged / firing
                if j < acts.shape[0] and acts[j] >= n_no_attack:
                    firing[j] = True

        # category-C channel: range-weighted radial push away from FIRING peers
        S = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            if not alive[i]:
                continue
            acc = np.zeros(2, dtype=np.float32)
            for j in range(n):
                if j == i or not firing[j] or not alive[j]:
                    continue
                diff = pos[i] - pos[j]
                dist = math.hypot(float(diff[0]), float(diff[1]))
                u = diff / (dist + _CWD_EPS)  # unit vector away from peer j
                w = 1.0 / (1.0 + dist / _CWD_R)  # closer firefights buffet harder
                acc += w * u
            S[i] = acc

        self._cwd_d = _CWD_RHO * self._cwd_d + (1.0 - _CWD_RHO) * (
            A * self.cwd_severity * S
        )
        np.clip(self._cwd_d, -_CWD_DCAP, _CWD_DCAP, out=self._cwd_d)
        self._cwd_d[~alive] = 0.0  # a dead unit carries no shove

        # PACT waveform: the SAME leaky accumulator without the hidden scalar
        # c = A*severity, so x2 = d/c on the unsaturated set.  UNCLIPPED (the pure
        # waveform); the DCAP clip on d is the small saturation leak (watch dsat),
        # not part of x2.  This is exactly the recursion a decentralized agent can
        # run from shared peer firing bits + relative positions.
        self._cwd_x2 = _CWD_RHO * self._cwd_x2 + (1.0 - _CWD_RHO) * S
        self._cwd_x2[~alive] = 0.0

        absd = np.abs(self._cwd_d)
        self._cwd_load_mean = float(absd.mean()) if absd.size else 0.0
        self._cwd_load_max = float(absd.max()) if absd.size else 0.0

    def _cwd_obs_state(self):
        """Build (per-agent obs list, repeated state), appending the oracle
        channels when the oracle ablation is set."""
        obs = self.env.get_obs()
        state = self.env.get_state()
        if self.cwd_oracle:
            obs = [
                np.concatenate(
                    [np.asarray(o, dtype=np.float32), self._cwd_d[i]]
                ).astype(np.float32)
                for i, o in enumerate(obs)
            ]
            state = np.concatenate(
                [np.asarray(state, dtype=np.float32), self._cwd_d.flatten()]
            ).astype(np.float32)
        elif self.cwd_pact:
            # decentralized: agent i sees only its OWN waveform x2_i (+ its norm)
            obs = [
                np.concatenate(
                    [
                        np.asarray(o, dtype=np.float32),
                        self._cwd_x2[i],
                        np.array(
                            [float(np.linalg.norm(self._cwd_x2[i]))], dtype=np.float32
                        ),
                    ]
                ).astype(np.float32)
                for i, o in enumerate(obs)
            ]
            # centralized critic: all agents' x2, plus (CTDE) the true driver A(t)
            extra = [np.asarray(state, dtype=np.float32), self._cwd_x2.flatten()]
            if self.cwd_pact_ctde:
                extra.append(np.array([self._cwd_payload], dtype=np.float32))
            state = np.concatenate(extra).astype(np.float32)
        return obs, self.repeat(state)

    # -------------------------------------------------------------------------
    def close(self):
        self.env.close()

    def load_map_config(self, map_name):
        base_path = osp.split(osp.split(osp.dirname(osp.abspath(__file__)))[0])[0]
        map_config_path = (
            Path(base_path)
            / "configs"
            / "envs_cfgs"
            / "smacv2_map_config"
            / f"{map_name}.yaml"
        )
        with open(str(map_config_path), "r", encoding="utf-8") as file:
            map_config = yaml.load(file, Loader=yaml.FullLoader)
        return map_config

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
