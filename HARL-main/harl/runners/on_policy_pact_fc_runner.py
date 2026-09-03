"""Runner for PACT on Formation Congestion.

TRAINING IS BIT-IDENTICAL TO THE HOST.  The whole method lives in the environment
wrappers (``harl/envs/smac/fc/``), so this runner adds exactly one thing: the
diagnostics file ``fc_debug.csv``.  That is deliberate -- PACT_PIPELINE_SPEC 1
says the host RL is never modified, so an arm difference cannot be an algorithm
difference, and the only way to keep that promise is for the runner to be a
logger and nothing else.

READ ``applied_trust`` AND ``delta_nonzero_frac`` BEFORE ANY OTHER NUMBER (9).
They answer "was the method ON AT ALL?" and "is it acting?".  On POWER a silently
disarmed compensator reported ``fit_r2 = 0.9998`` and a healthy learning curve
while being plain PPO.  Nothing looked wrong.  This runner therefore prints those
two numbers at every log interval on every arm -- including the BLIND ones, whose
``fc_*`` columns come from the severity wrapper -- because a silently inert NS is
invisible in exactly the arm you most need to trust.

SCHEMA CHANGES ROLL THE FILE ASIDE (9).  Two runs with different column counts in
one file once misaligned every field in the second segment and produced an
impossible ``cond_psi`` of 0.02, costing a full analysis pass.  ``_open`` compares
the header and renames the old file rather than appending to it.
"""

import csv
import os
import time

import numpy as np

from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner

_COLS = [
    # --- where are we -------------------------------------------------------
    "env_step", "rollout", "wall_s",
    # --- is the DIAL live? (every arm, blind included) ----------------------
    "sigma", "A", "g", "placebo_frac", "dial_ratio",
    # --- is the NS biting? --------------------------------------------------
    "u_mean", "u_max", "delta_mean", "delta_max", "peer_share",
    "stride_mean", "move_frac", "phi_var", "odom_err", "alive",
    # --- 9's table: was the method ON, and is it ACTING? --------------------
    "applied_trust", "delta_nonzero_frac", "delta_abs", "delta_clip_frac",
    "ff_abs", "peer_abs", "ff_share",
    "fit_gain_now", "cond_psi", "trP", "clamp_frac", "own_gain_se", "n_updates",
    "du_da", "floor_frac", "sat_frac", "cmd_mean", "state",
    # --- did it help? -------------------------------------------------------
    "ep_len", "ep_return", "win_rate",
]


def _m(vals):
    """Mean over finite entries only.  NaN when there is nothing finite --
    never an epsilon, and never a sentinel that would be averaged in."""
    a = np.asarray([v for v in vals if isinstance(v, (int, float, np.floating))],
                   dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


class PactFcLogMixin:
    """The diagnostics file.  Mixed into whichever host runner the arm uses."""

    #: set by the concrete subclasses; decides whether the compensator is built
    _pact_on = True

    def __init__(self, args, algo_args, env_args):
        # The wrapper stack is selected by env_args, so `--algo pact` implies it
        # rather than making the user remember two flags.  A blind arm gets the
        # severity layer and NOT the compensator, which is the same object with
        # `pact: 0` -- so the two arms differ by one boolean and nothing else.
        env_args = dict(env_args)
        env_args.setdefault("fc", 1)
        # The algo name is the authority: `pact` / `pact_mappo` build the
        # compensator, `happo_fc` / `mappo_fc` do not.  smac.yaml ships `pact: 0`
        # so that a plain `--algo mappo` gets the DIAL but not the method.
        env_args["pact"] = 1 if self._pact_on else 0
        super().__init__(args, algo_args, env_args)
        self.T = int(self.algo_args["train"]["episode_length"])
        self.nt = int(self.algo_args["train"]["n_rollout_threads"])
        self._t0 = time.time()
        self._rollout = 0
        self._acc = {c: [] for c in _COLS}
        self._eps = []
        self._lens = []
        self._wins = []
        self._ep_r = np.zeros(self.nt)
        self._ep_t = np.zeros(self.nt)
        self._dbg = None
        self._dbg_w = None
        if not self.algo_args["render"]["use_render"]:
            self._open(os.path.join(str(self.run_dir), "fc_debug.csv"))
        if not self._pact_on:
            print("[PACT] blind arm: the severity layer is ON, the compensator is "
                  "OFF.  fc_* columns are still written so an inert NS is visible.")

    # ------------------------------------------------------------------ #
    def _open(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", newline="") as f:
                    head = next(csv.reader(f), None)
            except (OSError, StopIteration):
                head = None
            if head != _COLS:
                # 9: NEVER append across a schema change.
                aside = path + ".%d.bak" % int(time.time())
                os.replace(path, aside)
                print("[PACT] fc_debug.csv schema changed -- rolled the old file to "
                      "%s rather than appending to it." % os.path.basename(aside))
        new = not os.path.exists(path)
        self._dbg = open(path, "a", encoding="utf-8", newline="")
        self._dbg_w = csv.writer(self._dbg)
        if new:
            self._dbg_w.writerow(_COLS)
            self._dbg.flush()

    # ------------------------------------------------------------------ #
    def insert(self, data):
        infos = data[4]
        dones = np.asarray(data[3])
        flat = []
        for th in infos:
            if isinstance(th, (list, tuple, np.ndarray)):
                flat.extend([x for x in th if isinstance(x, dict)])
            elif isinstance(th, dict):
                flat.append(th)
        if flat:
            self._collect(flat)
        # episode bookkeeping
        for t, th in enumerate(infos):
            row = th[0] if isinstance(th, (list, tuple, np.ndarray)) and len(th) \
                else (th if isinstance(th, dict) else None)
            if row is None:
                continue
            if np.all(dones[t]):
                self._wins.append(1.0 if row.get("won", False) else 0.0)
        super().insert(data)

    _KEYMAP = {
        "sigma": "fc_sigma", "A": "fc_A", "g": "fc_g", "dial_ratio": "fc_dial_ratio",
        "placebo_frac": "fc_placebo",
        "u_mean": "fc_u_mean", "u_max": "fc_u_max", "delta_mean": "fc_delta_mean",
        "delta_max": "fc_delta_max", "peer_share": "fc_peer_share",
        "stride_mean": "fc_stride_mean", "move_frac": "fc_move_frac",
        "phi_var": "fc_phi_var", "odom_err": "fc_odom_err", "alive": "fc_alive",
        "applied_trust": "pact_applied_trust",
        "delta_nonzero_frac": "pact_delta_nonzero_frac",
        "delta_abs": "pact_delta_abs", "delta_clip_frac": "pact_delta_clip_frac",
        "ff_abs": "pact_ff_abs", "peer_abs": "pact_peer_abs",
        "fit_gain_now": "pact_fit_gain_now", "cond_psi": "pact_cond_psi",
        "trP": "pact_trP", "clamp_frac": "pact_clamp_frac",
        "own_gain_se": "pact_own_gain_se", "n_updates": "pact_n_updates",
        "du_da": "pact_du_da", "floor_frac": "pact_floor_frac",
        "sat_frac": "pact_sat_frac", "cmd_mean": "pact_cmd_mean",
    }

    def _collect(self, rows):
        for col, key in self._KEYMAP.items():
            vals = [r[key] for r in rows if key in r]
            if vals:
                self._acc[col].append(_m(vals))
        st = [r.get("pact_state") for r in rows if "pact_state" in r]
        if st:
            self._acc["state"].append(st[-1])

    # ------------------------------------------------------------------ #
    def after_update(self):
        # `after_update` runs exactly once per rollout on every on-policy runner,
        # so the debug file has one row per rollout whatever the log interval is.
        self._write_row()
        super().after_update()

    def _write_row(self):
        if self._dbg_w is None:
            return
        self._rollout += 1
        step = self._rollout * self.T * self.nt
        row = {c: float("nan") for c in _COLS}
        row["env_step"] = step
        row["rollout"] = self._rollout
        row["wall_s"] = round(time.time() - self._t0, 1)
        for c in _COLS:
            if c in ("env_step", "rollout", "wall_s", "state", "ff_share",
                     "ep_len", "ep_return", "win_rate"):
                continue
            v = self._acc.get(c) or []
            row[c] = _m(v)
        row["state"] = (self._acc["state"][-1] if self._acc["state"] else "")
        # 7's honesty condition: report the LOCAL / COORDINATION split, always.
        ff, pe = row["ff_abs"], row["peer_abs"]
        row["ff_share"] = (ff / (ff + pe)) if np.isfinite(ff) and np.isfinite(pe) \
            and (ff + pe) > 0 else float("nan")
        row["win_rate"] = _m(self._wins) if self._wins else float("nan")
        row["ep_return"] = _m(self._eps) if self._eps else float("nan")
        row["ep_len"] = _m(self._lens) if self._lens else float("nan")
        self._dbg_w.writerow([row[c] for c in _COLS])
        self._dbg.flush()

        # The two numbers 9 says to read before any other, on the console, on the
        # log interval, on every arm.
        if self._rollout % max(1, int(self.algo_args["train"]["log_interval"])) != 0:
            for c in self._acc:
                self._acc[c] = []
            self._wins, self._eps, self._lens = [], [], []
            return
        print("[PACT] step=%d state=%s applied_trust=%.4f delta_nonzero=%.3f "
              "delta_abs=%.4f ff/peer=%.2f fit_gain=%.4f | sigma=%.2f g=%.3f "
              "delta_mean=%.4f dial_ratio=%.2f | win=%.3f"
              % (step, row["state"], row["applied_trust"], row["delta_nonzero_frac"],
                 row["delta_abs"], row["ff_share"], row["fit_gain_now"],
                 row["sigma"], row["g"], row["delta_mean"], row["dial_ratio"],
                 row["win_rate"]), flush=True)
        self._warn(row)
        for c in self._acc:
            self._acc[c] = []
        self._wins, self._eps, self._lens = [], [], []

    # ------------------------------------------------------------------ #
    def _warn(self, row):
        """The failures the spec paid for, each detected rather than assumed."""
        if row["sigma"] == 0.0 or (np.isfinite(row["sigma"]) and row["sigma"] <= 0):
            print("[PACT][INERT] applied severity is 0 -- the compensator is provably "
                  "inert here, so any arm difference over this stretch is basin luck, "
                  "not method.  Do not compare arms inside a warmup window.")
        if np.isfinite(row["dial_ratio"]) and row["dial_ratio"] <= 0.01 \
                and row["sigma"] > 0:
            print("[PACT][WARN] dial_ratio ~ 0: g == 1 on essentially every step, so "
                  "the NS is not biting.  Gate G0 (liveness) would fail -- check "
                  "ns_severity / ns_knee before reading anything else.")
        if self._pact_on and np.isfinite(row["applied_trust"]) \
                and row["applied_trust"] <= 0.0 and self._rollout > 3:
            print("[PACT][ASLEEP] applied_trust == 0: the coordination term is off. "
                  "Either fit_gain_now has not cleared pact_fit_floor yet, or the "
                  "gate has disarmed a working estimator (8.3).  fit_gain_now=%.4f "
                  "n_updates=%.0f" % (row["fit_gain_now"], row["n_updates"]))
        if np.isfinite(row["delta_clip_frac"]) and row["delta_clip_frac"] > 0.25:
            print("[PACT][RAIL] delta_clip_frac=%.2f -- a rail-pinned delta is a "
                  "CONSTANT BIAS, not a compensation: it stops responding to the "
                  "estimate entirely (6.3)." % row["delta_clip_frac"])
        if np.isfinite(row["phi_var"]) and row["phi_var"] < 0.05:
            print("[PACT][ESCAPE] std(Phi)/mean(Phi)=%.3f < 0.05: the exertion "
                  "functional has gone nearly constant, which makes beta "
                  "unidentifiable (A.5's counter-check).  Check whether the team "
                  "found an escape hatch." % row["phi_var"])
        # 2.4: treat non-finite as a VALUE and test it FIRST.  `if isfinite(c) and
        # c > thr` lets the most degenerate basis possible pass silently.
        if not np.isfinite(row["cond_psi"]) and self._rollout > 3:
            print("[PACT][COND] cond_psi is non-finite: beta may be predictable but "
                  "is NOT decomposable (12.6).  Do not claim to identify beta.")

    def close(self):
        try:
            if self._dbg is not None:
                self._dbg.close()
        finally:
            self._dbg = None
            self._dbg_w = None
        super().close()


class OnPolicyPactFcRunner(PactFcLogMixin, OnPolicyMARunner):
    """``--algo pact``  -- MAPPO host + Formation Congestion + the compensator."""
    _pact_on = True


class OnPolicyPactFcHappoRunner(PactFcLogMixin, OnPolicyHARunner):
    """``--algo pact_happo`` -- the same method on the HAPPO host.

    Shipping both is not indulgence: it is the check that the mechanism is not a
    property of one host's update rule.  The env wrappers are byte-identical
    across the two.
    """
    _pact_on = True


class OnPolicyFcBlindRunner(PactFcLogMixin, OnPolicyMARunner):
    """``--algo mappo_fc`` -- the BLIND baseline, with the same telemetry.

    Identical to ``--algo mappo`` in every trained quantity; it only also writes
    fc_debug.csv, so a silently inert dial is visible in the arm that would
    otherwise report nothing.
    """
    _pact_on = False


class OnPolicyFcBlindHappoRunner(PactFcLogMixin, OnPolicyHARunner):
    """``--algo happo_fc`` -- the blind HAPPO baseline, with the telemetry."""
    _pact_on = False
