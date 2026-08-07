"""RECON env shim for HARL / MAMuJoCo (spec Part 4) — modeled on ``ecl_mujoco.py``.

Wraps ``MujocoMulti`` and does three things, none of which touch the env:

1. **Declares the augmented spaces** before HARL sizes actor/critic/buffers:
   per-agent obs gets its own ``ℓ̂_i`` (k dims), the shared/critic state gets all
   N of them (k·N). The values themselves are written by the runner (see below);
   this class only fixes the dimensions.
2. **Stashes the raw, un-normalized readout** ``y_i`` — the agent's own hip/ankle
   joint-velocity one-step delta — into ``info`` for the trainer's [ID]. This has
   to happen here: ``MujocoMulti.get_obs()`` normalizes each obs vector by its own
   per-timestep mean/std, which cancels the coupling signal the identifier reads;
   and the vec-env worker auto-resets *after* ``step()`` returns, so a delta
   computed anywhere else would straddle the episode boundary.
3. **C4 de-aliased eval**: pins this env's payload clock to a per-rank offset so
   an eval round is a true cycle-average and not a phase-aliased snapshot.

**The action is not modified here and the filter does not live here.** With
``n_rollout_threads > 1`` HARL runs each env in its own process
(``ShareSubprocVecEnv``), so a *trained* torch module inside this class would be
a pickled copy that never sees a gradient — ECL could keep its adapter env-side
only because it was pure numpy. RECON therefore keeps [F] and [CP] in the runner
(one batched forward over threads×agents, one optimizer, no weight broadcast).
This is a placement decision, not a theory one: [F] still consumes only
``(o_i, u_i(t−1))`` and [CP] only ``ℓ̂_i``, so execution stays exactly as
decentralized as spec §2.6 requires — see ``on_policy_recon_runner.py``.

Enabled by ``env_args.recon: true``.
"""

import os

import numpy as np
from gym.spaces import Box

from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

_BANNER_SHOWN = False       # one banner per process (each env is its own process)


class ReconMujocoMulti(MujocoMulti):
    """``MujocoMulti`` + RECON's obs/share-obs dimensions and readout stash."""

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        env_args = kwargs["env_args"]
        cfg = dict(env_args.get("recon_cfg", {}))

        # k = dim(ℓ_i) = the agent's own action dim (Ant 4x2: own hip + own ankle).
        self.k = int(cfg.get("k", self.action_space[0].shape[0]))
        assert self.k == self.action_space[0].shape[0], (
            f"RECON: k ({self.k}) must equal the per-agent action dim "
            f"({self.action_space[0].shape[0]}) — ℓ_i lives on agent i's own "
            f"channels."
        )

        # Readout indices into the RAW ant obs. **MEASURED, NOT DERIVED.** Stock
        # gym Ant-v2 would put the 8 joint velocities at 19+2i (qpos[2:] is 13
        # wide, then qvel's 6 root dofs) — but on this deployment the block sits
        # 2 slots earlier, so own hips are at [17,19,21,23] and ankles at +1.
        # Two independent scans agree (ECL's, then RECON's own: modal offset -2
        # in 45/50 scans at corr 0.62), and ECL's working configs all override to
        # this map. Trusting the derivation over the measurement cost a 10M run:
        # every agent then regressed its own torque against a NEIGHBOUR's joint,
        # own torque explained ~nothing of the readout, and the clipfit railed at
        # the c-grid ceiling. `_check_scan` in the runner now aborts on a repeat.
        default_ro = [17 + 2 * i for i in range(self.n_agents)]
        self.readout_idx = np.asarray(
            cfg.get("readout_qvel_idx", default_ro), dtype=np.int64
        )
        default_ank = [self.readout_idx[i] + 1 for i in range(self.n_agents)]
        self.readout_idx_ankle = np.asarray(
            cfg.get("readout_qvel_idx_ankle", default_ank), dtype=np.int64
        )
        assert self.readout_idx.shape[0] == self.n_agents, (
            "RECON: readout_qvel_idx must have one index per agent."
        )
        self.stash_dobs = bool(cfg.get("stash_dobs", True))
        self._readout_warned = False

        # C4.1 de-aliased eval: a per-eval-env payload-clock offset (never set for
        # training envs). Set AFTER super().__init__ so it survives construction.
        offset = int(cfg.get("pcr_clock_offset", 0))
        if offset:
            tgt = getattr(self.env, "unwrapped", self.env)
            try:
                tgt._clock = int(offset)
            except Exception:
                print(f"[RECON] WARNING: could not set pcr_clock_offset={offset}.")

        # ---- declare augmented spaces BEFORE HARL sizes actor/critic/buffer ----
        # The runner writes the values into these slots; the env returns raw obs.
        self.recon_raw_obs_size = self.get_obs_size()
        self.recon_raw_share_size = self.get_state_size()
        self.observation_space = [
            Box(low=-10, high=10, shape=(self.recon_raw_obs_size + self.k,))
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            Box(low=-10, high=10, shape=(self.recon_raw_share_size + self.k * self.n_agents,))
            for _ in range(self.n_agents)
        ]

        self._prev_raw = self._raw_obs()
        self._banner(cfg)

    # ------------------------------------------------------------------ banner
    def _banner(self, cfg):
        global _BANNER_SHOWN
        if _BANNER_SHOWN:
            return
        _BANNER_SHOWN = True
        print(
            f"[RECON ENV] k={self.k} raw_obs={self.recon_raw_obs_size} "
            f"-> obs={self.recon_raw_obs_size + self.k} | raw_share="
            f"{self.recon_raw_share_size} -> share={self.recon_raw_share_size + self.k * self.n_agents} "
            f"| readout_hip={self.readout_idx.tolist()} "
            f"| clock_offset={int(cfg.get('pcr_clock_offset', 0))}",
            flush=True,
        )
        if os.environ.get("ANT_PCR_ORACLE", "0") == "1" or \
                os.environ.get("ANT_PCR_CORACLE", "0") == "1":
            print(
                "\n" + "=" * 64 + "\n[ORACLE ARM] ANT_PCR_ORACLE/CORACLE is set — the "
                "env is appending privileged\n             driver state to the "
                "observation. This is a labeled ceiling arm,\n             never the "
                "headline.\n" + "=" * 64, flush=True,
            )

    # ------------------------------------------------------------------ utils
    def _raw_obs(self):
        try:
            return np.asarray(self.env._get_obs(), dtype=np.float64)
        except Exception:
            return None

    def _readout(self, dobs):
        """Own hip/ankle joint-velocity deltas -> (n_agents, 2)."""
        y = np.zeros((self.n_agents, 2), dtype=np.float64)
        if dobs is None:
            return y
        ncol = dobs.shape[0]
        hi, ai = self.readout_idx, self.readout_idx_ankle
        if int(max(hi.max(), ai.max())) >= ncol:
            if not self._readout_warned:
                print(
                    f"[RECON] WARNING: readout index {int(max(hi.max(), ai.max()))} "
                    f"out of range (raw obs dim {ncol}); check "
                    f"recon_cfg.readout_qvel_idx*."
                )
                self._readout_warned = True
            hi = np.clip(hi, 0, ncol - 1)
            ai = np.clip(ai, 0, ncol - 1)
        y[:, 0] = dobs[hi]
        y[:, 1] = dobs[ai]
        return y

    # ------------------------------------------------------------------ step
    def step(self, actions):
        """``actions`` are the EXECUTED actions u (the runner has already applied
        [CP]); the env sees exactly them, and its reward is charged on exactly
        them — [CP] is a reparameterization of the policy's action space, so the
        agent pays for the torque it actually commands, as the blind agent does.
        """
        obs_n, state_n, rewards, dones, infos, avail = super().step(actions)

        next_raw = self._raw_obs()
        dobs = (
            next_raw - self._prev_raw
            if (next_raw is not None and self._prev_raw is not None)
            else None
        )
        self._prev_raw = next_raw

        info0 = infos[0]
        # RAW (un-normalized) per-agent readout for the trainer's identifier:
        # (n_agents, 2) = [own hip Δqvel, own ankle Δqvel].
        info0["recon_raw_y"] = self._readout(dobs).tolist()
        if self.stash_dobs and dobs is not None:
            # full raw-obs delta — read ONLY by the readout-index scan diagnostic
            info0["recon_raw_dobs"] = dobs.tolist()
        return obs_n, state_n, rewards, dones, infos, avail

    # ----------------------------------------------------------------- reset
    def reset(self, **kwargs):
        obs_n, state_n, avail = super().reset(**kwargs)
        # re-pair the readout (the cross-boundary delta is meaningless and is
        # dropped); nothing else in RECON is env-side state.
        self._prev_raw = self._raw_obs()
        return obs_n, state_n, avail
