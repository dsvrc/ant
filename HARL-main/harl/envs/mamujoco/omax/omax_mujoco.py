"""O-MAX env wrapper — hardwired compensation + privileged obs, all env-side.

This is the ceiling-ladder instrument (spec `OMAX_ceiling_ladder_spec.md`). It is
NOT a deployable method — every rung reads the env's privileged `pcr_*` telemetry.
The whole point is to measure the *reachable* return at sigma=0.45 when a learner
is given both the information and the machinery to use it, so the wrapper does two
things and the runner/algorithm stay stock HAPPO:

1. **Hardwired feed-forward compensation** (`comp_beta > 0`). The env's true
   disturbance for the NEXT action is `info["pcr_d_next"]`; cache it, and before
   the underlying step execute

       u = a − β · d_cache        (clipping happens inside ant.py)

   so the *delivered* torque is `clip(clip(a−d)+d) = clip(a)` on the unsaturated
   set — i.e. **the training env becomes byte-equivalent to the stationary Ant**.
   β = 1 is exact cancellation; the β-grid {0.5,0.75,1} is the O0 sweep. This is
   the mechanic behind rungs O1–O4: HAPPO trains *through* a stationary channel.
   (The env still charges ctrl-cost on the received action `u = a−d`; that is the
   ≤1% wedge documented in the spec §3.4 — expected, not a bug.)

2. **Privileged obs done right** (`aug_actor` / `aug_critic`). Unlike the env's
   `ANT_PCR_ORACLE` (which appends `d` *before* MujocoMulti's per-timestep
   whole-vector normalization, so the policy gets the scrambled `(d−μ_t)/σ_t` —
   the confound that made the user's oracle run underperform blind), this appends
   `[A(t), d·d_scale]` **after** normalization, in torque units. `aug_actor`
   augments each agent's obs with its OWN d slice; `aug_critic` augments the
   central share_obs with ALL of d. Each is an independent list of
   `["payload","d"]`.

Rungs (set by config; the wrapper is agnostic):
  O0  eval-only: a stationary walker + comp_beta (β-grid).      [scripted]
  O1  comp_beta=1, from scratch.                                (the decisive test)
  O2  O1 + aug_actor=[payload,d].
  O3  O2 + aug_critic=[payload,d].
  O4  O3 + warm-start (model_dir) + std_floor 0.05.             (O-MAX)

Enabled by `env_args.omax: true`. Prints an unmissable `[ORACLE ARM][O<k>]` banner.
"""

import numpy as np
from gym.spaces import Box

from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

_BANNER_SHOWN = False


class OmaxMujocoMulti(MujocoMulti):
    """MujocoMulti + hardwired true-d compensation and post-norm privileged obs."""

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        env_args = kwargs["env_args"]
        cfg = dict(env_args.get("omax_cfg", {}))

        self.comp_beta = float(cfg.get("comp_beta", 0.0))
        self.d_scale = float(cfg.get("d_scale", 10.0))
        self.aug_actor = [str(x) for x in cfg.get("aug_actor", [])]
        self.aug_critic = [str(x) for x in cfg.get("aug_critic", [])]
        self.rung = str(cfg.get("rung", "O?"))

        self.n_act = int(sum(sp.shape[0] for sp in self.true_action_space))  # 8 for 4x2
        # per-agent action offsets into the flat 8-dim d vector
        self._offsets = np.cumsum(
            [0] + [sp.shape[0] for sp in self.true_action_space]
        )
        self._d_cache = np.zeros(self.n_act)      # d the NEXT action will face
        self._A_cache = 0.0                       # A(t) payload

        # ---- augmentation dims (declare BEFORE HARL sizes actor/critic/buffer) --
        self._actor_aug = (
            (1 if "payload" in self.aug_actor else 0)
            + (self.true_action_space[0].shape[0] if "d" in self.aug_actor else 0)
        )
        self._critic_aug = (
            (1 if "payload" in self.aug_critic else 0)
            + (self.n_act if "d" in self.aug_critic else 0)
        )
        raw_obs = self.get_obs_size()
        raw_share = self.get_state_size()
        self.observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(raw_obs + self._actor_aug,))
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(raw_share + self._critic_aug,))
            for _ in range(self.n_agents)
        ]

        # C4 de-aliased eval: a per-eval-env payload-clock offset
        offset = int(cfg.get("pcr_clock_offset", 0))
        if offset:
            tgt = getattr(self.env, "unwrapped", self.env)
            try:
                tgt._clock = int(offset)
            except Exception:
                print(f"[OMAX] WARNING: could not set pcr_clock_offset={offset}.")

        self._banner()

    def _banner(self):
        global _BANNER_SHOWN
        if _BANNER_SHOWN:
            return
        _BANNER_SHOWN = True
        print(
            "\n" + "=" * 72
            + f"\n[ORACLE ARM][{self.rung}] comp_beta={self.comp_beta} "
            f"aug_actor={self.aug_actor} aug_critic={self.aug_critic} "
            f"d_scale={self.d_scale}\n"
            "  This is a CEILING-LADDER instrument: it reads the env's privileged\n"
            "  true disturbance to compensate/condition. It is NOT a deployable\n"
            "  method — it measures the reachable ceiling only.\n" + "=" * 72,
            flush=True,
        )

    # ------------------------------------------------------------------ augment
    def _agent_d(self, i):
        """Agent i's own slice of the cached d (torque units)."""
        return self._d_cache[self._offsets[i]:self._offsets[i + 1]]

    def _aug_obs_vec(self, agent_id):
        parts = []
        if "payload" in self.aug_actor:
            parts.append(np.array([self._A_cache], dtype=np.float64))
        if "d" in self.aug_actor:
            parts.append(self._agent_d(agent_id) * self.d_scale)
        return np.concatenate(parts) if parts else np.zeros(0)

    def _aug_state_vec(self):
        parts = []
        if "payload" in self.aug_critic:
            parts.append(np.array([self._A_cache], dtype=np.float64))
        if "d" in self.aug_critic:
            parts.append(self._d_cache * self.d_scale)
        return np.concatenate(parts) if parts else np.zeros(0)

    def _augment_obs(self, obs_n):
        return [
            np.concatenate([np.asarray(obs_n[i], dtype=np.float64), self._aug_obs_vec(i)])
            for i in range(self.n_agents)
        ]

    def _augment_state(self, state_n):
        av = self._aug_state_vec()
        return [
            np.concatenate([np.asarray(state_n[i], dtype=np.float64), av])
            for i in range(self.n_agents)
        ]

    # ------------------------------------------------------------------ step
    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float64)          # (n_agents, adim)
        a_flat = actions.reshape(-1)                             # intended action

        # --- hardwired feed-forward compensation with the TRUE cached d ---------
        # d_used is the d this action WILL face (cached from last step's
        # pcr_d_next); we log it against the env's pcr_d_applied to prove the
        # timing is exact.
        d_used = self._d_cache.copy()
        if self.comp_beta > 0.0:
            d_mat = np.stack(
                [self._d_cache[self._offsets[i]:self._offsets[i + 1]]
                 for i in range(self.n_agents)]
            )
            u = actions - self.comp_beta * d_mat                 # env clips inside
            step_actions = u
        else:
            step_actions = actions
        u_flat = np.asarray(step_actions, dtype=np.float64).reshape(-1)

        obs_n, state_n, rewards, dones, infos, avail = super().step(step_actions)
        info0 = infos[0]

        # ================= DIAGNOSTICS (answer the two questions) ==============
        # Q1 "is the disturbance actually removed from what the policy sees?"
        #    residual = delivered − clip(a_intended). ant delivers
        #    clip(clip(u)+d_applied); the stationary target is clip(a). residual≈0
        #    ⇒ the policy truly experiences the stationary env; residual large ⇒
        #    the disturbance is LEAKING through (comp not delivering the info).
        # Q2 "is the mechanism using the right information?"
        #    timing_err = d_used − pcr_d_applied. Must be ~0 or the comp is
        #    cancelling the wrong d (an off-by-one / reset bug).
        d_applied = np.asarray(
            info0.get("pcr_d_applied", np.zeros(self.n_act)), dtype=np.float64
        ).reshape(-1)
        tau = np.clip(u_flat, -1.0, 1.0)
        delivered = np.clip(tau + d_applied, -1.0, 1.0)
        clip_a = np.clip(a_flat, -1.0, 1.0)
        residual = delivered - clip_a
        info0["omax_d_absmean"] = float(np.mean(np.abs(d_applied)))
        info0["omax_d_absmax"] = float(np.max(np.abs(d_applied)))
        info0["omax_timing_err"] = float(np.max(np.abs(d_used - d_applied)))
        info0["omax_residual_absmean"] = float(np.mean(np.abs(residual)))
        info0["omax_residual_absmax"] = float(np.max(np.abs(residual)))
        info0["omax_u_clip"] = float(np.mean(np.abs(u_flat) > 1.0))     # comp railed
        info0["omax_u_minus_a"] = float(np.sqrt(np.mean((u_flat - a_flat) ** 2)))
        info0["omax_a_absmax"] = float(np.max(np.abs(a_flat)))          # policy saturating?
        info0["omax_beta"] = self.comp_beta

        # --- refresh the cache for the NEXT action (d it will actually face) -----
        if isinstance(info0, dict) and "pcr_d_next" in info0:
            self._d_cache = np.asarray(info0["pcr_d_next"], dtype=np.float64).reshape(-1)
            self._A_cache = float(info0.get("pcr_payload", 0.0))

        return (
            self._augment_obs(obs_n),
            self._augment_state(state_n),
            rewards,
            dones,
            infos,
            avail,
        )

    # ----------------------------------------------------------------- reset
    def reset(self, **kwargs):
        obs_n, state_n, avail = super().reset(**kwargs)
        # new episode: ant.py's reset_model zeroes _d, so the first action faces d=0
        self._d_cache = np.zeros(self.n_act)
        self._A_cache = 0.0
        return self._augment_obs(obs_n), self._augment_state(state_n), avail
