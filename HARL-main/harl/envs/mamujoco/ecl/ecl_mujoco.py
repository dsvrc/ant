"""ECL env shim for HARL / MAMuJoCo (spec Part 4 / 6.1) — envelope [E] only.

Wraps ``MujocoMulti`` and appends the local harm-envelope ``ε_i`` to each agent's
observation (and the shared/critic state). The action, dynamics, reward and env
are **untouched** — there is NO probe (Prohibition 3); the only agent-side change
is one extra observation coordinate. The trainer-side pieces (identifier,
localizer, anchor) live in the ECL runner/buffer and are enabled only for the
off-policy (HASAC) host; the on-policy (HAPPO) host receives this envelope only
(the degraded variant, spec Part 5).

Enabled by ``env_args.ecl: true``; the host actor/critic size themselves from the
augmented spaces declared here.
"""

import numpy as np
from gym.spaces import Box

from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
from harl.envs.mamujoco.ecl.envelope_adapter import EnvelopeAdapter


class EclMujocoMulti(MujocoMulti):
    """``MujocoMulti`` + the passive local envelope obs augmentation."""

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        env_args = kwargs["env_args"]
        cfg = dict(env_args.get("ecl_cfg", {}))

        self.envelope = EnvelopeAdapter(self.n_agents, cfg)

        # effort coordinate = own hip torque (even action index for Ant 4x2);
        # readout = own hip joint velocity in the raw Ant obs. On this env's ant.py
        # the qvel block starts 2 slots earlier than stock Ant, so hips land at
        # [17,19,21,23] and ankles at +1 = [18,20,22,24] (verified by the
        # identifier's scan_best_offset diagnostic).
        self.hip_action_idx = int(cfg.get("hip_action_idx", 0))
        self.ankle_action_idx = int(cfg.get("ankle_action_idx", 1))
        default_ro = [19 + 2 * i for i in range(self.n_agents)]
        self.readout_idx = np.asarray(cfg.get("readout_qvel_idx", default_ro), dtype=np.int64)
        default_ank = [self.readout_idx[i] + 1 for i in range(self.n_agents)]
        self.readout_idx_ankle = np.asarray(
            cfg.get("readout_qvel_idx_ankle", default_ank), dtype=np.int64
        )
        self.use_ankle = bool(cfg.get("use_ankle_channel", True))
        assert self.readout_idx.shape[0] == self.n_agents, (
            "ECL: readout_qvel_idx must have one index per agent."
        )
        self._readout_warned = False

        # C4.1 de-aliased eval: a per-eval-env payload-clock offset (never set for
        # training envs). Set AFTER super().__init__ so it survives construction.
        offset = int(cfg.get("pcr_clock_offset", 0))
        if offset:
            tgt = getattr(self.env, "unwrapped", self.env)
            try:
                tgt._clock = int(offset)
            except Exception:
                print(f"[ECL] WARNING: could not set pcr_clock_offset={offset}.")

        # declare augmented spaces BEFORE HARL sizes actor/critic/buffer:
        # per-agent obs gets its own ε_i; the shared state gets all N envelopes.
        raw_obs = self.get_obs_size()
        raw_share = self.get_state_size()
        self.observation_space = [
            Box(low=-10, high=10, shape=(raw_obs + 1,)) for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            Box(low=-10, high=10, shape=(raw_share + self.n_agents,))
            for _ in range(self.n_agents)
        ]

        self._prev_raw = self._raw_obs()

    # ------------------------------------------------------------------ utils
    def _raw_obs(self):
        try:
            return np.asarray(self.env._get_obs(), dtype=np.float64)
        except Exception:
            return None

    def _readout(self, prev_raw, next_raw, idx):
        """Per-agent own joint-velocity one-step delta at obs indices ``idx``."""
        y = np.zeros(self.n_agents, dtype=np.float64)
        if prev_raw is None or next_raw is None:
            return y
        ncol = next_raw.shape[0]
        if int(np.max(idx)) >= ncol:
            if not self._readout_warned:
                print(
                    f"[ECL] WARNING: readout index {int(np.max(idx))} out of range "
                    f"(obs dim {ncol}); check ecl_cfg.readout_qvel_idx*."
                )
                self._readout_warned = True
            idx = np.clip(idx, 0, ncol - 1)
        return next_raw[idx] - prev_raw[idx]

    def _augment_obs(self, obs_n, eps):
        return [
            np.concatenate([np.asarray(obs_n[i], dtype=np.float64), [eps[i]]])
            for i in range(self.n_agents)
        ]

    def _augment_state(self, state_n, eps):
        return [
            np.concatenate([np.asarray(state_n[i], dtype=np.float64), eps])
            for i in range(self.n_agents)
        ]

    # ------------------------------------------------------------------ step
    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        # own effort at t (Ant: own hip torque). Action is NOT modified.
        x1_hip = actions[:, self.hip_action_idx]

        obs_n, state_n, rewards, dones, infos, avail = super().step(actions)

        next_raw = self._raw_obs()
        y_hip = self._readout(self._prev_raw, next_raw, self.readout_idx)
        y_ank = (self._readout(self._prev_raw, next_raw, self.readout_idx_ankle)
                 if self.use_ankle else None)
        self._prev_raw = next_raw
        # envelope stays hip-only (it demonstrably works; keep it minimal — C1.2)
        eps_norm = self.envelope.update(x1_hip, y_hip)
        eps_raw = self.envelope.eps_raw

        obs_aug = self._augment_obs(obs_n, eps_norm)
        state_aug = self._augment_state(state_n, eps_norm)

        info0 = infos[0]
        info0["ecl_eps_mean"] = float(np.mean(eps_raw))       # v1 continuity (raw)
        info0["ecl_eps_norm_mean"] = float(np.mean(eps_norm))  # C2.3 appended chart
        info0["ecl_eps"] = eps_norm.tolist()
        info0["ecl_eps_raw"] = eps_raw.tolist()
        # RAW (un-normalized) per-agent readout deltas for the trainer's identifier
        # (hip + ankle channels, C1.2). The stored obs are per-timestep normalized,
        # which cancels the coupling signal — the identifier regresses on these.
        info0["ecl_raw_y"] = np.asarray(y_hip, dtype=np.float64).tolist()
        if self.use_ankle:
            info0["ecl_raw_y_ankle"] = np.asarray(y_ank, dtype=np.float64).tolist()
        return obs_aug, state_aug, rewards, dones, infos, avail

    # ----------------------------------------------------------------- reset
    def reset(self, **kwargs):
        obs_n, state_n, avail = super().reset(**kwargs)
        # reset ONLY the readout pairing (skip the cross-boundary readout); the
        # envelope EMAs persist across episodes (§3.6 / Prohibition 7).
        self._prev_raw = self._raw_obs()
        self.envelope.reset_episode()
        eps = self.envelope.eps
        return self._augment_obs(obs_n, eps), self._augment_state(state_n, eps), avail
