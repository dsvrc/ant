"""DiagMujocoMulti — the training-arm env shim  [campaign spec §3.2 / F2b].

``PcrDiagMujocoMulti`` (flight recorder) + two things the campaign's arms need
that the env's own ``ANT_PCR_ORACLE`` flag cannot express:

1. **Where** the privileged ``d`` goes: actor obs, critic ``share_obs``, both, or
   neither. ``ANT_PCR_ORACLE`` appends inside ``AntEnv._get_obs()``, which feeds
   *both* the per-agent obs and the shared state, so the spec's **F2b** ("d in
   critic input only") is not expressible without a shim. This is the shim the
   spec's own F2b implementation note calls for.

2. **In what units** it arrives. This is the important one, and it is a finding
   rather than a feature:

       MujocoMulti.get_obs():  obs_i = (obs_i - mean(obs_i)) / std(obs_i)
       MujocoMulti.get_state(): state = (state - mean(state)) / std(state)

   Both normalize the **whole concatenated vector**, and ``ANT_PCR_ORACLE``
   appends d *before* that. So every oracle arm ever run has fed its policy
   ``(d - mean_t)/std_t`` — **not** d in torque units. A feed-forward canceller
   needs d in torque units (it commands ``tau = desired - d``); to recover it the
   net must invert a per-step scale set largely by cfrc_ext contact spikes. That
   is a real, silent handicap on every privileged arm, and it is a live
   alternative explanation for E-5/E-7 ("the oracle failed") that has nothing to
   do with information being toxic.

   This shim appends d **after** normalization, so it arrives in torque units.
   The two schemas are then separable:

       F2   ANT_PCR_ORACLE=1, d_to=none    -> normalized d, actor+critic  (spec arm)
       F2c  ANT_PCR_ORACLE=0, d_to=both    -> RAW d, actor+critic         (control)
       F2b  ANT_PCR_ORACLE=0, d_to=share   -> RAW d, critic only          (spec arm)

   F2 vs F2c isolates the normalization; F2c vs F2b isolates actor-side
   conditioning from critic-side variance reduction; F2b vs F1c isolates the
   critic effect alone. Without F2c, axis V3 cannot tell "info is toxic" (H-C4)
   from "info arrived mangled", and the campaign would hand the method spec a
   confounded reading.

   (Checked: ``use_feature_normalization`` in the tuned config is inert for
   HASAC — its actor is ``SquashedGaussianPolicy`` and its critic is
   ``ContinuousQNet``, both built on ``PlainMLP``, which has no input LayerNorm.
   Only ``MLPBase`` (the on-policy side) honours that flag. So raw d does reach
   the first Linear in torque units.)

3. The C4 de-aliased eval clock offset, applied post-construction exactly as
   ``EclMujocoMulti`` does.

Hygiene: d is read from ``info["pcr_d_next"]`` — the same quantity the ORACLE obs
exposes — and reaches a network **only** on arms whose config says so and whose
banner announces it. Enabled by ``env_args.diag: true``.
"""

import os

import numpy as np
from gym.spaces import Box

from harl.envs.mamujoco.pcr_diag import PcrDiagMujocoMulti

_N_ACT = 8
_D_TO = ("none", "share", "obs", "both")


class DiagMujocoMulti(PcrDiagMujocoMulti):
    """Recorder + privileged-input schema arms + eval clock offset."""

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        env_args = kwargs["env_args"]
        cfg = dict(env_args.get("diag_cfg", {}))

        self.d_to = str(cfg.get("d_to", "none")).lower()
        assert self.d_to in _D_TO, f"diag_cfg.d_to must be one of {_D_TO}"

        if self.d_to != "none" and os.environ.get("ANT_PCR_ORACLE", "0") == "1":
            raise ValueError(
                "diag_cfg.d_to=%r together with ANT_PCR_ORACLE=1 would append d "
                "TWICE (once inside AntEnv._get_obs, normalized; once here, raw). "
                "Pick one: ANT_PCR_ORACLE=1 for the normalized-d arm (F2), or "
                "d_to=... for the raw-d arms (F2b/F2c)." % self.d_to
            )

        # C4.1 de-aliased eval: stratify this eval env's payload clock. Set AFTER
        # super().__init__ so it survives gym's construction-time probe step.
        offset = int(cfg.get("pcr_clock_offset", 0))
        if offset:
            tgt = getattr(self.env, "unwrapped", self.env)
            try:
                tgt._clock = int(offset)
            except Exception:
                print(f"[DIAG] WARNING: could not set pcr_clock_offset={offset}.",
                      flush=True)

        # declare augmented spaces BEFORE HARL sizes actor/critic/buffer
        raw_obs = self.get_obs_size()
        raw_share = self.get_state_size()
        add_obs = _N_ACT if self.d_to in ("obs", "both") else 0
        add_share = _N_ACT if self.d_to in ("share", "both") else 0
        self._add_obs, self._add_share = add_obs, add_share
        if add_obs:
            self.observation_space = [Box(low=-10, high=10, shape=(raw_obs + add_obs,))
                                      for _ in range(self.n_agents)]
        if add_share:
            self.share_observation_space = [
                Box(low=-10, high=10, shape=(raw_share + add_share,))
                for _ in range(self.n_agents)
            ]
        print(f"[DIAG ARM] d_to={self.d_to} (RAW, post-normalization; torque units) "
              f"obs {raw_obs}->{raw_obs + add_obs}  share {raw_share}->"
              f"{raw_share + add_share}  clock_offset={offset}", flush=True)

    # ------------------------------------------------------------------ utils
    def _aug(self, obs_n, state_n, d):
        d = np.asarray(d, dtype=np.float64).reshape(-1)
        if self._add_obs:
            obs_n = [np.concatenate([np.asarray(o, dtype=np.float64), d])
                     for o in obs_n]
        if self._add_share:
            state_n = [np.concatenate([np.asarray(s, dtype=np.float64), d])
                       for s in state_n]
        return obs_n, state_n

    # ------------------------------------------------------------------ step
    def step(self, actions):
        obs_n, state_n, rewards, dones, infos, avail = super().step(actions)
        if self.d_to == "none":
            return obs_n, state_n, rewards, dones, infos, avail
        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
        # pcr_d_next == what the NEXT action will face == exactly what the ORACLE
        # obs appends. Same quantity, different units (see the module docstring).
        d = info0.get("pcr_d_next")
        if d is None:
            d = np.asarray(getattr(self.env, "_d", np.zeros(_N_ACT)))
        obs_n, state_n = self._aug(obs_n, state_n, d)
        return obs_n, state_n, rewards, dones, infos, avail

    # ----------------------------------------------------------------- reset
    def reset(self, **kwargs):
        obs_n, state_n, avail = super().reset(**kwargs)
        if self.d_to == "none":
            return obs_n, state_n, avail
        # reset_model zeroes _d, so the schema's d slots start at exactly 0
        obs_n, state_n = self._aug(obs_n, state_n, np.zeros(_N_ACT))
        return obs_n, state_n, avail
