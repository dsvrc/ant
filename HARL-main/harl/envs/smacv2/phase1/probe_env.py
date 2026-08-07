"""SMACv2ProbeEnv -- the Phase-1 scripted-compensation probe for SMACv2-CWD.

This is the PACT-pipeline Phase-1 instrument (NOT a deployable method).  It wraps
the CWD env with a *privileged, scripted* controller that is handed the true
hidden shove ``d_i`` and cancels it, so we can sweep severity and find sigma-star
-- the largest severity at which compensation still recovers the baseline.

Two compensation laws (declaration #3 of the pipeline; the target-drift row of the
Part-II table), selected by ``comp_mode``:

* ``"continuous"`` -- idealized re-aim.  The env adds only the *uncompensated
  remainder* ``(1 - beta) * d_i`` to the delivered move target instead of the full
  ``d_i``.  With ``beta = 1`` the delivered target equals the commanded target, so
  the game is **byte-identical to stock SMACv2 at every severity**: the CWD harm is
  a pure translation (design doc §5.1) and ``_CWD_DCAP`` guarantees the inverse
  never saturates (§5.4).  This arm is therefore the *invertibility / transparency
  certificate*, not a severity frontier -- it should recover B0 at all sigma.

* ``"discrete"`` -- the realizable frontier, and the sigma-star HEADLINE.  A unit's
  action *selection* stays discrete (design doc §3.1): the controller replaces a
  commanded cardinal MOVE with the available cardinal move (or STOP) whose
  post-shove displacement best matches the intended one:

      believed_delivered(a') = move_vec(a') + beta * d_i     (a' a cardinal move)
      believed_delivered(STOP) = 0                            (STOP carries no target -> undrifted)
      a* = argmin_{a' available} || believed_delivered(a') - move_vec(a_commanded) ||

  ``beta = 0`` reproduces the blind policy exactly (a* == a_commanded); ``beta = 1``
  is full re-aim.  The *bounded resource* here is the discrete-move set itself:
  along-intent shove cannot be undone by picking a different cardinal (you cannot
  command "move less"), so a residual survives and grows with ``|d_i|`` until the
  ``_CWD_DCAP`` cap -- that residual is what pushes the return below the bar.  This
  is a GREEDY per-step re-aim: a conservative (lower-bound) proxy for the discrete
  ceiling (a learned policy could additionally *time* its steps, §5.4).

Freeze knob (Phase-1 §II.1.2): ``freeze`` holds the exogenous driver A(t) at its
PEAK (1.0), so the frozen game has effective coupling c = severity -- the worst
slice, held constant.  Set ``freeze=None`` to run the live bombardment ramp.

Everything is env-side; the host RL and the trained actor are untouched.  The
oracle obs-append is forced OFF (the baseline being defended is the *blind* one).

Wiring: enabled by ``env_args["phase1"] = True``; knobs in ``env_args["phase1_cfg"]``
or set at run time via ``configure_probe()`` (the sweep holds one SC2 process open
and reconfigures between cells).
"""

import numpy as np

from harl.envs.smacv2.smacv2_env import SMACv2Env, _CWD_DCAP


# --- discrete SMAC action convention (standard SMAC / smacv2 get_agent_action) ---
#   0 = no-op (dead)     1 = stop            2 = move NORTH (+y)
#   3 = move SOUTH (-y)  4 = move EAST (+x)  5 = move WEST (-x)   >=6 = attack/heal
# Only the four cardinal MOVES (2..5) compile to a target_world_space_pos and are
# therefore the only actions CWD drifts; STOP/attack/no-op carry no world target.
# *** VERIFY on the run machine *** against the installed smacv2's get_agent_action
# (the axis/sign of N/S/E/W and the index order).  The sweep's low-severity
# "works-when-it-should" gate will fail loudly if this map is wrong.
STOP = 1
_MOVE_ACTIONS = (2, 3, 4, 5)
_CAND_ACTIONS = (1, 2, 3, 4, 5)  # STOP + the four cardinal moves
# unit direction of each cardinal move (scaled by move_amount at run time)
_DIR = {
    1: np.array([0.0, 0.0], dtype=np.float64),   # STOP (undrifted)
    2: np.array([0.0, 1.0], dtype=np.float64),   # NORTH  +y
    3: np.array([0.0, -1.0], dtype=np.float64),  # SOUTH  -y
    4: np.array([1.0, 0.0], dtype=np.float64),   # EAST   +x
    5: np.array([-1.0, 0.0], dtype=np.float64),  # WEST   -x
}

_BANNER_SHOWN = False


class SMACv2ProbeEnv(SMACv2Env):
    """CWD env + scripted (privileged) compensation controller for Phase-1."""

    def __init__(self, args):
        super().__init__(args)
        cfg = dict((self.args or {}).get("phase1_cfg", {}))
        # comp_mode in {"none","discrete","continuous"}; comp_beta the gain
        self.comp_mode = str(cfg.get("comp_mode", "discrete"))
        self.comp_beta = float(cfg.get("comp_beta", 1.0))
        # freeze the driver at its peak (1.0) by default -> stationary worst slice.
        # None runs the live ramp.  env_args["cwd_freeze"] (resolved in seed) is the
        # ground truth; we default the cfg's freeze into args so seed() picks it up.
        if "cwd_freeze" not in self.args and cfg.get("freeze", 1.0) is not None:
            self.args["cwd_freeze"] = float(cfg.get("freeze", 1.0))
        # Phase-1 defends the BLIND baseline -> force both obs-append modes off
        # (the scripted controller runs on the un-augmented blind obs shape).
        self.args["cwd_oracle"] = 0
        self.args["cwd_pact"] = 0
        # per-step telemetry (means over MOVING, alive units), refreshed each step
        self._phase1_residual = 0.0     # ||believed_delivered - intended|| the comp couldn't cancel
        self._phase1_changed = 0.0      # frac of moving units whose discrete action was changed
        self._phase1_n_move = 0

    # ------------------------------------------------------------------ banner
    def _phase1_banner(self):
        global _BANNER_SHOWN
        if _BANNER_SHOWN:
            return
        _BANNER_SHOWN = True
        print(
            "\n" + "=" * 74
            + "\n[PHASE-1 PROBE][SMACv2-CWD] scripted, privileged compensation."
            f"\n  comp_mode={self.comp_mode}  comp_beta={self.comp_beta}  "
            f"severity={self.cwd_severity}  freeze(A)={self.cwd_freeze}"
            "\n  This is a CERTIFICATION instrument (finds sigma-star); it reads the"
            "\n  env's TRUE hidden shove d_i to cancel it.  It is NOT a method.\n"
            + "=" * 74,
            flush=True,
        )

    # ------------------------------------------------------------ live reconfig
    def configure_probe(self, comp_mode=None, comp_beta=None, severity=None,
                        freeze="__keep__"):
        """Reconfigure the probe in place (the sweep holds one SC2 process open and
        changes the cell between rollouts).  Zeroes the accumulated shove so a
        previous cell cannot leak into the next."""
        if comp_mode is not None:
            self.comp_mode = str(comp_mode)
        if comp_beta is not None:
            self.comp_beta = float(comp_beta)
        if severity is not None:
            self.cwd_severity = float(severity)
        if freeze != "__keep__":
            self.cwd_freeze = None if freeze is None else float(freeze)
        if hasattr(self, "_cwd_d"):
            self._cwd_d[:] = 0.0
        self._phase1_banner()

    # ------------------------------------------------- continuous-comp harm hook
    def _cwd_delivered_shift(self, a_id):
        """Offset actually added to unit ``a_id``'s move target this step.

        Continuous re-aim leaves only the uncompensated remainder ``(1-beta)*d``;
        every other mode delivers the full shove and (for discrete) compensates by
        rewriting the action instead.
        """
        if self.comp_mode == "continuous" and self.comp_beta != 0.0:
            return (1.0 - self.comp_beta) * self._cwd_d[a_id]
        return self._cwd_d[a_id]

    # ---------------------------------------------------------- discrete re-aim
    def _discrete_reaim(self, actions):
        """Rewrite each commanded cardinal MOVE to the available discrete action
        whose believed post-shove displacement best matches the intended move.

        Uses ``self._cwd_d`` (the shove THIS step's moves will receive, already set
        at the end of the previous step / at reset) and the current availability
        mask.  Returns a new action array of the same shape as ``actions``.
        """
        orig_shape = np.asarray(actions).shape
        acts = np.asarray(actions).reshape(-1).astype(int).copy()
        avail = self.env.get_avail_actions()  # current (pre-step) availability
        d = self._cwd_d
        M = float(self._cwd_move_amount)
        beta = float(self.comp_beta)

        n_move = 0
        n_changed = 0
        resid_sum = 0.0
        for i in range(self.n_agents):
            a = int(acts[i]) if i < acts.shape[0] else 0
            if a not in _MOVE_ACTIONS:  # only cardinal moves are drifted / re-aimed
                continue
            n_move += 1
            v_star = M * _DIR[a]                      # intended displacement
            di = np.asarray(d[i], dtype=np.float64)   # this unit's shove
            avail_i = np.asarray(avail[i]).reshape(-1)

            best_act, best_cost, best_resid = a, None, 0.0
            for cand in _CAND_ACTIONS:
                if cand >= avail_i.shape[0] or avail_i[cand] == 0:
                    continue
                if cand == STOP:
                    delivered = np.zeros(2, dtype=np.float64)  # STOP is undrifted
                else:
                    delivered = M * _DIR[cand] + beta * di
                resid = float(np.hypot(*(delivered - v_star)))
                cost = resid * resid
                if best_cost is None or cost < best_cost - 1e-9:
                    best_cost, best_act, best_resid = cost, cand, resid
            if best_act != a:
                n_changed += 1
            acts[i] = best_act
            resid_sum += best_resid

        self._phase1_n_move = n_move
        self._phase1_changed = (n_changed / n_move) if n_move else 0.0
        self._phase1_residual = (resid_sum / n_move) if n_move else 0.0
        return acts.reshape(orig_shape)

    # -------------------------------------------------------------------- step
    def step(self, actions):
        do_discrete = self.comp_mode == "discrete" and self.comp_beta != 0.0
        if do_discrete:
            actions = self._discrete_reaim(actions)
        else:
            # continuous / none: no action rewrite; residual is the felt remainder
            self._phase1_n_move = 0
            self._phase1_changed = 0.0
            self._phase1_residual = 0.0

        obs, state, rewards, dones, infos, avail = super().step(actions)

        # ---- Phase-1 telemetry (means for the sweep; never touches the reward) ----
        d_applied = np.asarray(infos[0].get("cwd_d_applied", self._cwd_d))
        if self.comp_mode == "continuous":
            # felt residual = the uncancelled part of the shove on moving units
            rem = (1.0 - self.comp_beta) * d_applied
            self._phase1_residual = float(np.mean(np.hypot(rem[:, 0], rem[:, 1])))
        absd = np.abs(d_applied)
        dsat = float(np.mean(absd >= (_CWD_DCAP - 1e-6))) if absd.size else 0.0

        p1 = {
            "phase1_comp_mode": self.comp_mode,
            "phase1_beta": self.comp_beta,
            "phase1_residual": self._phase1_residual,   # ||uncancelled displacement||
            "phase1_changed_frac": self._phase1_changed,  # discrete: frac actions remapped
            "phase1_dsat_frac": dsat,                   # frac of |d| axes at the DCAP cap
        }
        for info in infos:
            info.update(p1)
        return obs, state, rewards, dones, infos, avail
