"""Ant PCR non-stationarity, REPAIRED BENCHMARK (sigma = 0.45) + diagnostic knobs.

This is the campaign's ``ant_diag.py`` (v1 physics + env-var knobs + info-only
telemetry) with the severity default set to the repaired value **0.45** — the
R-a redesign adopted after the Tier-0 campaign measured the feasibility
frontier at sigma* = 0.5 (E2b). It is the deployable for every RECON-era arm.

Deployment (run machine): copy this file over ``gym/envs/mujoco/ant.py``.
It is the ONLY file ever copied there.

**With no ``ANT_PCR_*`` env var set** this env's dynamics are byte-identical to
the plain v1 file edited to SEVERITY=0.45 (the knobs are default-off; the only
additions are info keys and one startup banner). Equivalence chain for the V0
test: this file == ``ant_diag.py`` under ``ANT_PCR_SEVERITY=0.45``, and
``ant_diag.py`` == ``ant_pcr_v1.py`` under no env vars (already asserted by
``test_ant_diag.py``).

--------------------------------------------------------------------------------
Knobs (all optional; absent => the repaired blind benchmark)
--------------------------------------------------------------------------------
``ANT_PCR_SEVERITY``  float, default 0.45 (the repaired benchmark). Override for
                      the P4 stress arm (0.55) and frontier work only.
``ANT_PCR_FREEZE_A``  float or unset. If set, ``_payload()`` returns this
                      constant: the payload stops drifting and the env becomes a
                      **stationary** Markov game at effective severity
                      ``c = FREEZE_A * SEVERITY`` (the frozen-slice family:
                      Stage-0 E2/E3/E5 refits). The clock keeps ticking.
``ANT_PCR_MASK``      both | hip | ankle | off. Which coupling channel stays
                      LIVE (``off`` => stock Ant + ticking clock — the
                      stationary-walker arm). Default ``both``.
``ANT_PCR_DCAP``      float or unset. If set, ``|d|`` is clipped to this cap
                      after each update (Route-B evidence arms only).

Oracle ablations (labeled arms only):
``ANT_PCR_ORACLE=1``  append the exact per-joint load d (8 dims) to the obs
                      (d-oracle ceiling arm).
``ANT_PCR_CORACLE=1`` append the scalar driver c = A(t)*SEVERITY (c-oracle arm).

Info-only telemetry (always on; hygiene-safe)
--------------------------------------------------------------------------------
``pcr_d_applied``  (8,) the d added to THIS step's commanded torque (pre-update).
``pcr_d_next``     (8,) ``self._d`` AFTER the update — what the NEXT action will
                   face; identical to the ORACLE obs slice. RECON's filter
                   true-error column and the Tier-0 cancellation probes read it.
``pcr_sat_frac``   fraction of the 8 joints with ``|tau + d_applied| > 1``.
``pcr_clock``      the global shift clock (int), read after this step's tick.

**Hygiene.** Knobs and info keys may be read only by probe wrappers, recorders,
and labeled diagnostic/oracle arms. They must never reach a policy input outside
the labeled ``ANT_PCR_ORACLE`` / ``ANT_PCR_CORACLE`` flags. (RECON's ``pcr_d_next``
use is a *logging column* — the filter never consumes it.)

Every process prints one unmissable ``[DIAG ENV]`` banner from the constructor
so run dirs are always classifiable from their logs.
"""

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

# =====================================================================
#  THE ONE KNOB — now set by the MEASURED feasibility frontier, not by feel.
# =====================================================================

SEVERITY = float(os.environ.get("ANT_PCR_SEVERITY", "0.45"))
# Repaired benchmark severity (R-a). The Tier-0 campaign measured the
# cancellation-feasibility frontier at sigma* = 0.5: below it, feed-forward
# cancellation with known d recovers 95-100% of the stationary return (E2:
# c=0.45, beta=1 -> 5040 = 95% of B0=5328) and nothing saturates; above it the
# required compensation exceeds the +/-1 torque limit (46% of joints saturated
# at sigma=0.9) and NO controller, with any information, can hold the peak —
# the old "the ORACLE recovers at ANY value" claim is FALSE and was the
# ill-posedness that sank every method before the diagnosis. 0.45 = frontier
# minus margin: the blind zero-shot collapse is still deep (5328 -> 2080 at
# peak, a 61% drop) while the task stays provably recoverable (WP-1 pass).

# =====================================================================
#  Oracle ablations (labeled arms only). Env-var controlled so arms launch
#  without editing this file. Both default OFF (the hard blind task).
#
#  NOTE (campaign): both append INSIDE _get_obs(), i.e. *before* MujocoMulti's
#  per-timestep whole-vector normalization  obs <- (obs - mean)/std . The policy
#  therefore receives (d - mean_t)/std_t, NOT d in torque units. That is a real
#  handicap on every oracle arm; the DiagMujocoMulti `d_to`+`d_scale` arms
#  isolate it. See diag/README.md.
# =====================================================================
ORACLE = os.environ.get("ANT_PCR_ORACLE", "0") == "1"
CORACLE = os.environ.get("ANT_PCR_CORACLE", "0") == "1"

# =====================================================================
#  Diagnostic knobs. All default-off; absent => the repaired blind benchmark.
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
#  Fixed internals — these set the SHAPE of the non-stationarity, not its
#  strength. UNCHANGED by the repair (R-a touches sigma only).
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
    """One unmissable line per process (A0 provenance parses this)."""
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
#  settings coexist in one process (Tier-0 grid loops; F4's two-freeze scoring).
#  The env-var interface is unchanged and remains the ONLY thing a training arm
#  uses (one arm, one process, one setting, one banner).
# --------------------------------------------------------------------------
def set_freeze_a(value):
    """Freeze A(t) at ``value`` (or None to restore drift) for envs constructed
    AFTER this call. Existing instances keep their snapshot."""
    global _FREEZE_A_VAL
    _FREEZE_A_VAL = None if value is None else float(value)


def set_severity(value):
    """Set sigma for envs constructed AFTER this call (frontier sweeps)."""
    global SEVERITY
    SEVERITY = float(value)


def set_mask(value):
    """Set the live coupling channel for envs constructed AFTER this call."""
    global _MASK
    assert value in ("both", "hip", "ankle", "off")
    _MASK = value


def set_dcap(value):
    """Set the |d| cap for envs constructed AFTER this call (Route-B arms)."""
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
        # channel mask (E6 attribution / redesign arms): MASK names the channel
        # that stays LIVE; the other one's coupling is zeroed.
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
                pcr_load=float(load.mean()),   # mean |d| — telemetry
                pcr_loadmax=float(load.max()),
                # --- campaign telemetry (info-only, hygiene-safe) ---
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
        # c-oracle: expose only the scalar driver c = A(t)*SEVERITY.
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
