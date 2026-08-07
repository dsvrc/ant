"""Ant PCR non-stationarity + DIAGNOSTIC KNOBS  [campaign spec Part 2].

This is ``ant_pcr_v1.py`` (the deployed SEVERITY=0.9 PCR Ant, kept beside this
file as the golden reference) plus a set of env-var-gated diagnostic knobs and
info-only telemetry keys. **With no ``ANT_PCR_*`` env var set, its observable
behavior is byte-identical to ``ant_pcr_v1``** — that equivalence is asserted by
``test_ant_diag.py`` (stage V0) and must pass before any deployment
(Prohibition 5).

Deployment (run machine): copy this file over ``gym/envs/mujoco/ant.py``.
It is the ONLY file ever copied there.

--------------------------------------------------------------------------------
Knobs (all optional; absent => exact v1 behavior)
--------------------------------------------------------------------------------
``ANT_PCR_SEVERITY``  float, default 0.9. The disturbance gain σ. E2b sweeps it
                      to locate the feasibility frontier σ*.
``ANT_PCR_FREEZE_A``  float or unset. If set, ``_payload()`` returns this
                      constant: the payload stops drifting and the env becomes a
                      **stationary** Markov game at effective severity
                      ``c = FREEZE_A · SEVERITY``. This is the frozen-slice
                      family (E1/E2/E3/E4/E6, F1/F2/F3/F4). The clock keeps
                      ticking (harmless — nothing else reads it).
``ANT_PCR_MASK``      both | hip | ankle | off. Which coupling channel stays
                      LIVE: ``hip`` => only the hip channel couples (ankle
                      channel zeroed), ``ankle`` => only the ankle channel
                      couples, ``off`` => neither (stock Ant with a ticking
                      clock — the F0 stationary-walker arm), ``both`` => v1.
``ANT_PCR_DCAP``      float or unset. If set, ``|d|`` is clipped to this cap
                      after each update (Route-B redesign evidence, E2b leg 3).

Info-only telemetry (always on; hygiene-safe — see below)
--------------------------------------------------------------------------------
``pcr_d_applied``  (8,) the d that was added to THIS step's commanded torque
                   (the pre-update value). Equals the previous step's
                   ``pcr_d_next``.
``pcr_d_next``     (8,) ``self._d`` AFTER the update — identical to what the
                   ORACLE obs appends; this is what the NEXT action will face.
``pcr_sat_frac``   fraction of the 8 joints with ``|tau + d_applied| > 1``
                   (the disturbance exceeded actuator authority).
``pcr_clock``      the global shift clock (int), read after this step's tick —
                   i.e. the clock whose payload produced ``pcr_d_next``.

**Hygiene (spec Part 2).** These keys and knobs may be read only by probe
wrappers, recorders and *labeled* diagnostic arms. They must never reach a
policy input outside the already-labeled ``ANT_PCR_ORACLE`` / ``ANT_PCR_CORACLE``
observation flags.

Every process prints one unmissable ``[DIAG ENV]`` banner from the constructor so
run dirs are always classifiable by ``scripts/diag_a0_inventory.py`` — this is a
hard requirement (Prohibition 3), the direct fix for the E-5 ambiguity that
motivated this campaign.
"""

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

# =====================================================================
#  THE ONE KNOB YOU TUNE  (now overridable for the E2b severity sweep)
# =====================================================================

SEVERITY = float(os.environ.get("ANT_PCR_SEVERITY", "0.9"))
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
#
#  NOTE (campaign): both append INSIDE _get_obs(), i.e. *before* MujocoMulti's
#  per-timestep whole-vector normalization  obs <- (obs - mean)/std . The policy
#  therefore receives (d - mean_t)/std_t, NOT d in torque units. That is a real
#  handicap on every oracle arm and a live confound for axis V3; the F2c /
#  DiagMujocoMulti `d_to`+`d_scale` arms isolate it. See diag/README.md.
# =====================================================================
ORACLE = os.environ.get("ANT_PCR_ORACLE", "0") == "1"
CORACLE = os.environ.get("ANT_PCR_CORACLE", "0") == "1"

# =====================================================================
#  Diagnostic knobs (spec Part 2). All default-off; absent => exact v1 behavior.
# =====================================================================
_FREEZE_A = os.environ.get("ANT_PCR_FREEZE_A")          # e.g. "1.0" -> A(t) == 1.0
_MASK = os.environ.get("ANT_PCR_MASK", "both")          # both | hip | ankle | off
_DCAP = os.environ.get("ANT_PCR_DCAP")                  # e.g. "0.5" -> |d| clipped

_FREEZE_A_VAL = None if _FREEZE_A is None else float(_FREEZE_A)
_DCAP_VAL = None if _DCAP is None else float(_DCAP)
assert _MASK in ("both", "hip", "ankle", "off"), (
    f"ANT_PCR_MASK must be one of both|hip|ankle|off (got {_MASK!r})"
)

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

_BANNER_SHOWN = False


def _banner():
    """One unmissable line per process (Prohibition 3 — A0 parses this)."""
    global _BANNER_SHOWN
    if _BANNER_SHOWN:
        return
    _BANNER_SHOWN = True
    print(
        "[DIAG ENV] SEVERITY=%s FREEZE_A=%s MASK=%s DCAP=%s ORACLE=%s CORACLE=%s "
        "RHO=%s P=%s B=%s"
        % (SEVERITY, _FREEZE_A_VAL, _MASK, _DCAP_VAL, int(ORACLE), int(CORACLE),
           _RHO, _P, _B),
        flush=True,
    )


# --------------------------------------------------------------------------
#  Runtime knob setters — DIAGNOSTIC, LABELED ARMS ONLY.
#
#  Each AntEnv snapshots the knobs at construction, so envs built at different
#  settings coexist in one process. Two campaign stages require this and cannot
#  work without it:
#
#    * `scripts/diag_tier0.py` owns the FREEZE_A / SEVERITY grid loops (spec §4,
#      Appendix B: "diag_tier0.py owns the full grid loops; single-cell invocation
#      shown for shape") — one process must build envs at A=0, 0.25, ... 1.0.
#    * F4 (spec §5) trains at FREEZE_A=1 while scoring a 5-episode FREEZE_A=0
#      return every 10k steps — two live envs, two different freezes.
#
#  Module-level constants alone can express neither. The env-var interface is
#  unchanged and remains the ONLY thing a training arm uses (one arm, one process,
#  one setting, one banner); these setters exist for the two scripted stages above
#  and print their own per-cell banner.
# --------------------------------------------------------------------------
def set_freeze_a(value):
    """Freeze A(t) at ``value`` (or None to restore drift) for envs constructed
    AFTER this call. Existing instances keep their snapshot."""
    global _FREEZE_A_VAL
    _FREEZE_A_VAL = None if value is None else float(value)


def set_severity(value):
    """Set sigma for envs constructed AFTER this call (E2b's sweep)."""
    global SEVERITY
    SEVERITY = float(value)


def set_mask(value):
    """Set the live coupling channel for envs constructed AFTER this call (E6)."""
    global _MASK
    assert value in ("both", "hip", "ankle", "off")
    _MASK = value


def set_dcap(value):
    """Set the |d| cap for envs constructed AFTER this call (E2b leg 3 / Route B)."""
    global _DCAP_VAL
    _DCAP_VAL = None if value is None else float(value)


def current_knobs():
    """The module's live knob settings — for banners and artifact provenance."""
    return {"SEVERITY": SEVERITY, "FREEZE_A": _FREEZE_A_VAL, "MASK": _MASK,
            "DCAP": _DCAP_VAL, "ORACLE": ORACLE, "CORACLE": CORACLE,
            "RHO": _RHO, "P": _P, "B": _B}


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        _banner()
        # NS state MUST exist before super().__init__ (gym calls step() during init
        # to size the observation).
        self._clock = 0                  # global shift clock — persists across episodes
        self._d = np.zeros(_N_ACT)       # per-joint parasitic load; reset each episode
        # Snapshot the knobs so envs at different settings can coexist in one
        # process (see the setter block above). With no setter ever called these
        # are exactly the module constants, i.e. exactly the env vars.
        self._freeze_a = _FREEZE_A_VAL
        self._severity = SEVERITY
        self._mask = _MASK
        self._dcap = _DCAP_VAL
        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)
        self._clock = 0                  # discard the probe step gym ran during construction
        self._d[:] = 0.0

    def _payload(self):
        """A(t): exogenous payload over the delivery shift, in [0,1]. Trig-free,
        asymmetric smoothstep — loaded quickly at the depot, shed slowly en route.

        FREEZE_A pins A(t) to a constant => the env is a stationary Markov game at
        effective severity c = FREEZE_A * SEVERITY (the frozen-slice family)."""
        if self._freeze_a is not None:
            return self._freeze_a
        ph = (self._clock % _P) / _P
        x = ph / _B if ph < _B else (1.0 - ph) / (1.0 - _B)
        return x * x * (3.0 - 2.0 * x)    # smoothstep -> smooth peak & troughs

    def step(self, a):
        a = np.asarray(a, dtype=np.float64)
        tau = np.clip(a, -1.0, 1.0)                       # commanded torque (already in [-1,1])

        # --- apply THIS step's parasitic load (prepared at the end of the last step,
        #     so it equals the value exposed to the oracle in the previous obs) ------
        d_applied = self._d.copy()                        # info-only snapshot (pre-update)
        raw = tau + self._d
        delivered = np.clip(raw, -1.0, 1.0)
        sat_frac = float(np.mean(np.abs(raw) > 1.0))      # info-only

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
        # channel mask (E6 attribution / R-c redesign): MASK names the channel that
        # stays LIVE; the other one's coupling is zeroed.
        if self._mask == "ankle":
            s[0::2] = 0.0                                 # hip channel off
        elif self._mask == "hip":
            s[1::2] = 0.0                                 # ankle channel off
        elif self._mask == "off":
            s[:] = 0.0                                    # stock Ant + a ticking clock
        self._d = _RHO * self._d + (1.0 - _RHO) * (A * self._severity * s)
        if self._dcap is not None:
            self._d = np.clip(self._d, -self._dcap, self._dcap)

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
                # --- campaign telemetry (info-only, hygiene-safe; Part 2) ---
                pcr_d_applied=d_applied,       # the d added to THIS step's torque
                pcr_d_next=self._d.copy(),     # what the NEXT action will face
                pcr_sat_frac=sat_frac,         # |tau + d_applied| > 1 fraction
                pcr_clock=int(self._clock),
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
            obs = np.concatenate([obs, [self._payload() * self._severity]])
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
