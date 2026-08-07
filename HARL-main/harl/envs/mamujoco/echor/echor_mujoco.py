"""ECHO-R env-wrapper shim for HARL / MAMuJoCo (spec Part 4 + Part 6.1).

This is the *preferred* HARL integration slot ("wrap ``MujocoMulti`` ... prefer
the env-wrapper slot; fewer host-code touches").  It subclasses ``MujocoMulti``
and, transparently to the host algorithm:

  1. injects each agent's microscopic orthogonal probe into the effort
     coordinate the PCR coupling reads (the leg's **hip** action) *before* the
     action reaches the simulator (6.1 wiring note);
  2. reads out each agent's own delivered effectiveness (its hip joint-velocity
     delta) and demodulates it into an estimate ``c_hat`` of the hidden severity;
  3. appends ``c_hat`` to the observation (and the shared/critic state).

Because the probe is added *inside* ``step`` and the host policy's raw action is
what the runner passes in and stores, the replay/rollout buffer automatically
keeps the **pre-probe** action ``u`` (spec 5.2) with zero host-code changes, and
the host actor/critic are sized from this wrapper's *augmented* observation and
share-observation spaces (declared in ``__init__``, before HARL probes them).

No environment / NS file is touched (Part 10.1); ``ant.py`` stays exactly as
designed.  Nothing hidden (``A(t)``, the env clock, the liability, ``info``
diagnostics) is ever read by the estimator (Part 10.2).  The probe runs at BOTH
training and evaluation (the eval envs are wrapped too), satisfying the
train/exec-symmetry requirement (Part 10.7) automatically.
"""

import numpy as np
from gym.spaces import Box

from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
from harl.envs.mamujoco.echor.echor_adapter import EchoRAdapter


class EchoRMujocoMulti(MujocoMulti):
    """``MujocoMulti`` with the ECHO-R probe/demod/conditioning layer wrapped in."""

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)

        env_args = kwargs["env_args"]
        cfg = dict(env_args.get("echor_cfg", {}))

        # ---- ECHO-R adapter (one per env instance; the campaign clock lives
        # here and persists across episodes because HARL reuses the env) ----
        self.adapter = EchoRAdapter(self.n_agents, cfg)

        # ---- injection geometry: which action coordinate(s) carry the probe.
        # Ant partitions order each agent's actuators as [hip, ankle, ...]; the
        # coupling reads hip<->hips, so we probe the *even* (hip) coordinates
        # (6.1: hip only for 4x2).  Overridable via ``echor_cfg.inject_mask``.
        act_dim = int(self.action_space[0].shape[0])
        default_mask = (np.arange(act_dim) % 2 == 0).astype(np.float64)
        self.inject_mask = np.asarray(
            cfg.get("inject_mask", default_mask), dtype=np.float64
        )

        # ---- readout geometry: the obs index of each agent's own hip joint
        # velocity in the *raw* Ant ``_get_obs`` = [qpos[2:] (13), qvel (14), ...].
        # Torso qvel occupies qvel[0:6]; leg-i hip velocity is qvel[6 + 2*i] =
        # obs[13 + 6 + 2*i] = obs[19 + 2*i].  Correct for the reference
        # ``agent_conf 4x2`` (agent == leg); override for other partitions.
        default_ro = [[19 + 2 * i] for i in range(self.n_agents)]
        ro = cfg.get("readout_qvel_idx", default_ro)
        self.readout_idx = [np.atleast_1d(np.asarray(r, dtype=np.int64)) for r in ro]
        assert len(self.readout_idx) == self.n_agents, (
            "ECHO-R: readout_qvel_idx must have one entry per agent."
        )
        self._readout_warned = False

        # ---- augment the declared observation / share-observation spaces ----
        # per-agent obs gets its own scalar c_hat_i; the shared (critic) state
        # gets the full vector of all agents' estimates (5.2).
        self.echor_obs_extra = 1
        self.echor_share_extra = self.n_agents
        self.raw_obs_size = self.get_obs_size()
        self.raw_share_size = self.get_state_size()
        aug_obs = self.raw_obs_size + self.echor_obs_extra
        aug_share = self.raw_share_size + self.echor_share_extra
        self.observation_space = [
            Box(low=-10, high=10, shape=(aug_obs,)) for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            Box(low=-10, high=10, shape=(aug_share,)) for _ in range(self.n_agents)
        ]

        # ---- c-oracle diagnostic (spec Part 8, D0 gate) ----
        # When true, the policy is conditioned on the TRUE driver A(t) read from
        # the env `info` (the one sanctioned hygiene exception, for D0 only)
        # instead of the demodulated estimate.  This isolates "does conditioning
        # on the perfect driver beat blind?" from estimator quality: if this
        # arm does not beat blind, no estimator can help and the bottleneck is
        # the NS calibration / conditioning, not ECHO-R.
        self._c_oracle = bool(cfg.get("c_oracle", False))
        self._last_drv = 0.0

        # readout one-step memory (raw preferred; normalised obs as fallback)
        self._prev_raw = self._raw_obs()
        self._prev_norm = [np.asarray(o, dtype=np.float64) for o in self.get_obs()]

        # ---- optional detailed debug trace (opt-in via echor_cfg.debug) ----
        self._dbg = None
        if bool(cfg.get("debug", False)):
            from harl.envs.mamujoco.echor.echor_debug import EchoRDebugLogger

            self._dbg = EchoRDebugLogger(
                self.n_agents,
                debug_dir=cfg.get("debug_dir", "./echor_debug"),
                interval=int(cfg.get("debug_interval", 1)),
            )

    # ------------------------------------------------------------------ utils
    def _raw_obs(self):
        """Raw (unnormalised) Ant proprioception, for a clean readout.

        Falls back to the normalised per-agent obs slice if ``_get_obs`` is
        unavailable; the ratio (P4) cancels the per-vector normalisation gain
        either way, so both are valid.
        """
        try:
            return np.asarray(self.env._get_obs(), dtype=np.float64)
        except Exception:
            return None

    def _readout(self, prev_raw, next_raw, obs_n):
        """Per-agent one-step hip-velocity delta ``y_i`` (3.3 / 6.1)."""
        if prev_raw is not None and next_raw is not None:
            src_prev, src_next, ncol = prev_raw, next_raw, next_raw.shape[0]
        else:  # fallback: use the normalised per-agent obs (state block prefix)
            src_prev, src_next, ncol = None, None, len(obs_n[0])
        y = np.zeros(self.n_agents, dtype=np.float64)
        for i in range(self.n_agents):
            idx = self.readout_idx[i]
            if idx.max() >= ncol:
                if not self._readout_warned:
                    print(
                        f"[ECHO-R] WARNING: readout index {idx.max()} out of range "
                        f"(obs dim {ncol}); check echor_cfg.readout_qvel_idx against "
                        f"your Ant obs layout. Clamping."
                    )
                    self._readout_warned = True
                idx = np.clip(idx, 0, ncol - 1)
            if src_next is not None:
                y[i] = float(np.mean(src_next[idx] - src_prev[idx]))
            else:  # normalised-obs fallback needs the previous normalised obs
                y[i] = float(
                    np.mean(np.asarray(obs_n[i])[idx] - self._prev_norm[i][idx])
                )
        return y

    def _augment_obs(self, obs_n, c_hat):
        return [
            np.concatenate([np.asarray(obs_n[i], dtype=np.float64), [c_hat[i]]])
            for i in range(self.n_agents)
        ]

    def _augment_state(self, state_n, c_hat):
        return [
            np.concatenate([np.asarray(state_n[i], dtype=np.float64), c_hat])
            for i in range(self.n_agents)
        ]

    # ------------------------------------------------------------------ step
    def step(self, actions):
        # (3.7 step 3) inject the probe using the code at the current clock,
        # landing it in the commanded torque both the sim and the coupling read.
        # The env, learner and reward are all untouched (spec P6 / Prohibition 9:
        # ECHO-R conditions on the estimate, it never cancels).
        a_exec = self.adapter.modulate_cont(actions, self.inject_mask)

        obs_n, state_n, rewards, dones, infos, avail = super().step(a_exec)

        # (3.3) readout, then (3.4-3.7) demodulate + advance the clock.
        next_raw = self._raw_obs()
        y = self._readout(self._prev_raw, next_raw, obs_n)
        self._prev_raw = next_raw
        self._prev_norm = obs_n  # kept for the normalised-obs fallback path
        done_env = bool(np.all(dones))
        c_hat = self.adapter.observe(y, done=done_env)

        info0 = infos[0]
        pcr_payload = float(info0.get("pcr_payload", float("nan")))
        if not np.isnan(pcr_payload):
            self._last_drv = pcr_payload

        # what actually conditions the policy: the estimate, or -- in the D0
        # diagnostic arm -- the true driver read from info.
        c_used = (
            np.full(self.n_agents, self._last_drv, dtype=np.float64)
            if self._c_oracle
            else c_hat
        )
        obs_aug = self._augment_obs(obs_n, c_used)
        state_aug = self._augment_state(state_n, c_used)

        # diagnostics for logging / the Part-8 D2 tracking test (never fed back).
        info0["echor_chat_mean"] = float(np.mean(c_hat))
        info0["echor_chat"] = c_hat.tolist()

        # focused per-step debug trace (opt-in): the quantities that actually
        # explain a non-working run -- true driver vs estimate, and the channel
        # SNRs that reveal whether the probe is even detectable.
        if self._dbg is not None:
            self._dbg.log(
                self.adapter.debug_snapshot(),
                {
                    "pcr_payload": pcr_payload,
                    "pcr_load": float(info0.get("pcr_load", float("nan"))),
                    "reward": float(np.mean(np.asarray(rewards))),
                    "done": float(done_env),
                },
            )

        return obs_aug, state_aug, rewards, dones, infos, avail

    # ----------------------------------------------------------------- close
    def close(self):
        if getattr(self, "_dbg", None) is not None:
            self._dbg.close()
        super().close()

    # ----------------------------------------------------------------- reset
    def reset(self, **kwargs):
        obs_n, state_n, avail = super().reset(**kwargs)
        # reset ONLY the readout's one-step pairing (drop the cross-boundary
        # readout); the clock / correlograms / c_hat persist (3.7 / Part 10.4).
        self._prev_raw = self._raw_obs()
        self._prev_norm = obs_n
        self.adapter.reset_episode()
        c_used = (
            np.full(self.n_agents, self._last_drv, dtype=np.float64)
            if self._c_oracle
            else self.adapter.c_hat
        )
        return self._augment_obs(obs_n, c_used), self._augment_state(state_n, c_used), avail
