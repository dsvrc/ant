"""Ant with a Payload-Coupled Chassis-Reaction (PCR) non-stationarity.  [category C, oracle-separable]

Drop-in replacement for the gym ``Ant-v2`` file (``gym/envs/mujoco/ant.py``):
copy this over that file (rename to ``ant.py``) on the run machine to deploy.

The whole effect lives in the TRANSITION (the torque the simulator actually
receives). The reward function is byte-for-byte the original Ant reward — no
penalty term is ever added (constraint 3, dynamics-only).

This is a FRESH instance of the same category-C template as the other envs — it is
NOT the earlier rotation / thermal-misalignment design. Different physics
(transmitted reaction LOAD, not axis rotation), different math (an additive
disturbance, not an orthogonal transform), different driver (a delivery-shift
payload, not a diurnal cosine).

--------------------------------------------------------------------------------
The idea
--------------------------------------------------------------------------------
The four legs are bolted to ONE chassis. By Newton's third law every leg's motor
torque pushes back on that shared floating base; when the base is LOADED (a courier
quadruped carrying cargo) it flexes and transmits a fraction of the OTHER legs'
reaction into each leg's joints as an unmodelled BIAS torque — a steady "tug" the
controller never commanded. Heavier payload  ->  more transmitted reaction.

So each leg i carries a hidden parasitic-load vector  d_i  (one bias per joint). It
is a leaky accumulator fed ONLY by the OTHER legs' commanded torque, with the
feed-rate gated by an exogenous, slowly-drifting PAYLOAD A(t). The bias is ADDED to
the commanded torque inside the dynamics:

    A(t)               = payload over the delivery shift, in [0,1]   (global clock, persists)
    d_i  <-  rho*d_i + (1-rho) * A(t) * SEVERITY * sum_{j!=i} tau_j   (per joint, signed)
    delivered_i = tau_i (commanded) + d_i                            (a DISTURBANCE, not a scaling)
    reward = the ORIGINAL Ant reward                                 (ctrl cost on COMMANDED tau)

Why this is category C (not B, not A)
  * A(t) only ever MULTIPLIES the sum over the OTHER legs — it is never an additive
    term on its own. With no teammates (N=1) the sum is empty, so d_i == 0 for all
    t no matter how the payload drifts: the env reduces EXACTLY to stock Ant.
    -> irreducible / vanishes at N=1.   (Cleanest with agent_conf "4x2": agent == leg.)
  * A(t) depends on a global clock only -> it keeps drifting even with FROZEN
    partners -> a genuine non-stationarity, not a co-learning artefact.
  * The sum EXCLUDES leg i, so a leg cannot set its own liability — the load is
    individually exogenous but collectively endogenous. Tragedy of the commons: the
    harder the team pushes the more it disturbs every member, and because the
    payload drifts, the optimal way to share the effort keeps moving.

Why it is oracle-separable (constraint 7)
  An additive disturbance is FULL-RANK and feed-forward cancelable: a controller
  that KNOWS d_i simply commands  tau_i = desired_i - d_i, so delivered_i =
  desired_i  ->  the EXACT stock-Ant dynamics, at ANY severity. That is the textbook
  "measurable disturbance -> reject it" problem; knowing the driver is (most of) the
  solution. A BLIND controller is not given d_i and cannot infer it (the payload
  phase is hidden and the other legs' commanded torques are not in the observation),
  so every joint is pushed by an unknown, drifting, gait-frequency-varying load: its
  gait decoheres, it is tipped past the height limit, it TERMINATES, and the return
  collapses through the untouched reward. The harm is a mis-direction you can aim out
  of, never a capacity you lose.
  Set ORACLE=True (or env ANT_PCR_ORACLE=1) to append d_i to the observation for the
  oracle / information-recoverability run.
"""

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

# =====================================================================
#  THE ONE KNOB YOU TUNE
# =====================================================================

SEVERITY = 0.9
# Disturbance gain: torque-bias per unit neighbour-effort at the payload peak.
# Higher  -> the blind ant is shoved harder -> deeper, more reliable collapse.
# The ORACLE recovers at ANY value (an additive disturbance is always cancelable),
# so this dial only sets how hard the *blind* problem is. Calibrate by watching the
# `pcr_load` diagnostic (mean |d|): aim for ~0.3-0.5 at the payload peak — enough to
# tip the gait, with headroom below the +/-1 torque limit so the oracle can cancel.

# =====================================================================
#  Oracle ablations (labeled arms only). Env-var controlled so arms launch without
#  editing this file:
#    ANT_PCR_ORACLE=1  -> G2 d-oracle: append the exact per-joint load d (8 dims).
#    ANT_PCR_CORACLE=1 -> G1 c-oracle: append the scalar driver c = A(t)*SEVERITY.
#  Both default OFF (the hard blind task).
# =====================================================================
ORACLE = os.environ.get("ANT_PCR_ORACLE", "0") == "1"
CORACLE = os.environ.get("ANT_PCR_CORACLE", "0") == "1"

# =====================================================================
#  Fixed internals — these set the SHAPE of the non-stationarity, not its strength.
# =====================================================================
_P = 40_000     # delivery-shift period, in this env's steps. Short on purpose, so many
                # shifts occur over any run length (robust) and the metric oscillates
                # visibly (period ~= _P * n_rollout_threads on the total-step axis).
_B = 0.2        # fraction of the shift spent LOADING at the depot (fast load, slow shed).
_RHO = 0.8      # chassis-reaction memory (leaky-integrator retention). Lower -> a
                # higher-frequency load that feedback cannot reject -> stronger collapse.

_N_ACT = 8      # gym Ant: 4 legs x (hip, ankle). Legs are action pairs (0,1)(2,3)(4,5)(6,7).


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # NS state MUST exist before super().__init__ (gym calls step() during init
        # to size the observation).
        self._clock = 0                  # global shift clock — persists across episodes
        self._d = np.zeros(_N_ACT)       # per-joint parasitic load; reset each episode
        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)
        self._clock = 0                  # discard the probe step gym ran during construction
        self._d[:] = 0.0

    def _payload(self):
        """A(t): exogenous payload over the delivery shift, in [0,1]. Trig-free,
        asymmetric smoothstep — loaded quickly at the depot, shed slowly en route."""
        ph = (self._clock % _P) / _P
        x = ph / _B if ph < _B else (1.0 - ph) / (1.0 - _B)
        return x * x * (3.0 - 2.0 * x)    # smoothstep -> smooth peak & troughs

    def step(self, a):
        a = np.asarray(a, dtype=np.float64)
        tau = np.clip(a, -1.0, 1.0)                       # commanded torque (already in [-1,1])

        # --- apply THIS step's parasitic load (prepared at the end of the last step,
        #     so it equals the value exposed to the oracle in the previous obs) ------
        delivered = np.clip(tau + self._d, -1.0, 1.0)

        # --- original Ant transition, now driven by the DISTURBED torque ------------
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(delivered, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        # --- original Ant reward (ctrl cost charged on the COMMANDED torque) ---------
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward  # UNCHANGED
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone

        # --- advance the shift clock; recharge the parasitic load for the NEXT step --
        #     d_i <- rho*d_i + (1-rho) * A(t) * SEVERITY * sum_{j!=i} tau_j  (per joint)
        self._clock += 1
        A = self._payload()
        hip, ank = tau[0::2], tau[1::2]                   # this step's per-leg commands
        s = np.empty_like(tau)
        s[0::2] = hip.sum() - hip                         # sum_{j!=i} hip_j  (category-C channel)
        s[1::2] = ank.sum() - ank                         # sum_{j!=i} ankle_j
        self._d = _RHO * self._d + (1.0 - _RHO) * (A * SEVERITY * s)

        ob = self._get_obs()
        load = np.abs(self._d)
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
                # --- NS diagnostics (NOT part of the reward) ---
                pcr_payload=float(A),          # the exogenous driver A(t)
                pcr_load=float(load.mean()),   # mean |d| — the calibration target
                pcr_loadmax=float(load.max()),
            ),
        )

    def _get_obs(self):
        obs = np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )
        # Oracle ablation: expose the EXACT parasitic load the next action will face,
        # so a driver-conditioned policy can command (desired - d) and recover stock Ant.
        if ORACLE:
            obs = np.concatenate([obs, self._d])
        # c-oracle: expose only the scalar driver c = A(t)*SEVERITY (the ECHO-R
        # "condition on the driver" arm — expected to NOT recover, unlike d).
        if CORACLE:
            obs = np.concatenate([obs, [self._payload() * SEVERITY]])
        return obs

    def reset_model(self):
        # The body starts each episode unloaded (d -> 0); the delivery-shift clock keeps
        # running across episodes (the campaign does not reset).
        self._d[:] = 0.0
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
