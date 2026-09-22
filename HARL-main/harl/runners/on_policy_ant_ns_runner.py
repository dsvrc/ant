"""Runner for PACT-1 on ANT-NS.  An arm selector and a logger, nothing else.

`PACT_NS_SPEC` P-9.1: *"Ship three arms through the identical wrapper, differing
only in the trust term."*  This runner sets the env flags that pick an arm and
writes II.10's instrument panel to ``pact_debug.csv``.  The PPO update, the
clipped surrogate, the entropy bonus, the network and the optimiser are the
host's own, untouched -- an arm difference must not be able to be an algorithm
difference.  ``pact*.yaml`` are ``mappo.yaml`` byte for byte.

THE ARMS (all through the identical layer; the compensator is INSIDE it)
--------------------------------------------------------------------------------
    blind      `--algo happo` (or mappo) inside the dial.  No estimator.
    pactoff    trust forced to 0.  BIT-IDENTICAL to blind: the correction is
               exactly 0.0, the arithmetic is the same arithmetic, and no RNG is
               consumed -- so the executed torques, the trajectory and the
               training data are the blind arm's.  The instrumented baseline.
    pact       trust fixed at 0.9, gated by prediction confidence, after warmup.
    fixed      an alias of pact on this channel (see below).
    oracle     handed the TRUE disturbance -- the ceiling.  On this channel it
               cancels exactly, so its trajectory IS the sigma = 0 trajectory
               and no method can beat it.
    intercept  peer channels deleted, everything else identical.  On (C) it can
               only track a per-agent level; on (B) it is the whole answer.

Learned trust (P-6.2) is not implementable on an invertible action-space
channel -- the correction is applied below the policy, so ``d log pi / d g = 0``
-- which is why ``fixed`` and ``pact`` coincide here.  ``simple_ns`` and
``grf_ns`` have the same limitation and state it; so does the README.

THE PANEL (II.10)
--------------------------------------------------------------------------------
"Is the dial live", "is the NS biting", "is the METHOD working" and "is it
WINNING" are SEPARATE column groups, ordered so the first column that answers
"no" is the one to fix.  The domain metric here is Ant's own return, decomposed
into the host's four reward terms so a fall can be told from a slowdown.
"""

import csv
import os
import time

import numpy as np

from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner

#: arm -> (ns_trust, ns_oracle, ns_intercept_only)
ARMS = {
    "pact": ("fixed", 0, 0),
    "pactoff": ("off", 0, 0),
    "fixed": ("fixed", 0, 0),
    "oracle": ("fixed", 1, 0),
    "intercept": ("fixed", 0, 1),
}

_COLS = [
    "env_step", "rollout", "wall_s",
    # is the dial live?  (every arm)
    "sigma", "A", "amp", "placebo", "dial_live", "harmed", "actuator_clipped",
    # is the NS biting?
    "d", "dmax", "y", "x_hh", "x_aa", "x_cross", "x_std", "tau_rms",
    # is the METHOD working?  (kept apart from "is it winning")
    "fit_gain", "beta_cos", "beta_relerr", "cond_psi", "trust_pol", "trust_app",
    "pred_err", "corr", "clip_frac", "rows", "diverged", "bounded", "corr_clipped",
    # is it WINNING?
    "ep_len", "ep_return", "r_forward", "r_ctrl", "r_contact", "r_survive",
]

#: column -> (info key, transform)
_KEY = {
    "sigma": ("ns_sigma", None), "A": ("ns_A", None), "amp": ("ns_amp", None),
    "placebo": ("ns_placebo", None), "dial_live": ("ns_dial_live", None),
    "harmed": ("ns_harmed", None), "actuator_clipped": ("ns_clipped", None),
    "d": ("ns_d", None), "dmax": ("ns_dmax", None), "y": ("ns_y", abs),
    "x_hh": ("ns_x_hh", None), "x_aa": ("ns_x_aa", None), "x_cross": ("ns_x_cross", None),
    "x_std": ("ns_x_std", None), "tau_rms": ("ns_tau_rms", None),
    "fit_gain": ("ns_fit_gain", None), "beta_cos": ("ns_beta_cos", None),
    "beta_relerr": ("ns_beta_relerr", None), "cond_psi": ("ns_cond_psi", None),
    "trust_app": ("ns_trust", None), "pred_err": ("ns_pred_err", None),
    "corr": ("ns_c", None), "clip_frac": ("ns_clip_frac", None),
    "rows": ("ns_rows", None), "diverged": ("ns_diverged", None),
    "bounded": ("ns_bounded", None), "corr_clipped": ("ns_corr_clipped", None),
    # the host's own reward decomposition -- a fall and a slowdown look the same
    # in the return and completely different here
    "r_forward": ("reward_forward", None), "r_ctrl": ("reward_ctrl", None),
    "r_contact": ("reward_contact", None), "r_survive": ("reward_survive", None),
}


def _m(v):
    a = np.asarray([x for x in v if isinstance(x, (int, float, np.floating, np.integer))],
                   dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


class PactAntMixin(object):
    """Arm selection + the instrument panel.  Mixed into whichever host runs."""

    arm = "pact"

    def __init__(self, args, algo_args, env_args):
        trust, oracle, intercept = ARMS[self.arm]
        env_args = dict(env_args)
        on = env_args.get("ns_on", 1)
        if isinstance(on, str):
            on = on.strip().lower() in ("1", "true", "yes", "on")
        assert int(on) == 1, (
            "a PACT-family arm needs the severity layer; for a B0 reference run use "
            "`--algo happo --env mamujoco_ns --ns_on 0`")
        env_args["ns_pact"] = 1
        env_args["ns_trust"] = trust
        env_args["ns_oracle"] = int(oracle)
        env_args["ns_intercept_only"] = int(intercept)
        if oracle:
            # the ceiling relies FULLY on the true disturbance; pact keeps P-5.1's 0.9
            env_args["ns_g_fixed"] = 1.0
        super().__init__(args, algo_args, env_args)
        self._g_fixed = float(env_args.get("ns_g_fixed", 0.9)) if trust == "fixed" else 0.0
        self.T = int(self.algo_args["train"]["episode_length"])
        self.nt = int(self.algo_args["train"]["n_rollout_threads"])
        self._t0 = time.time()
        self._rollout = 0
        self._acc = {c: [] for c in _COLS}
        self._eps, self._lens = [], []
        self._ep_r = np.zeros(self.nt)
        self._ep_t = np.zeros(self.nt)
        self._dbg = self._dbg_w = None
        if not self.algo_args["render"]["use_render"]:
            self._open(os.path.join(str(self.run_dir), "pact_debug.csv"))
        print("[PACT-ANT] arm=%s  trust=%s (g=%.2f)  oracle=%d  intercept=%d  "
              "channel=EXACT INVERSE (II.6 row 1: identification AND compensation)"
              % (self.arm, trust, self._g_fixed, oracle, intercept))

    def _open(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", newline="") as f:
                    head = next(csv.reader(f), None)
            except (OSError, StopIteration):
                head = None
            if head != _COLS:
                # never append across a schema change
                os.replace(path, path + ".%d.bak" % int(time.time()))
        new = not os.path.exists(path)
        self._dbg = open(path, "a", encoding="utf-8", newline="")
        self._dbg_w = csv.writer(self._dbg)
        if new:
            self._dbg_w.writerow(_COLS)
            self._dbg.flush()

    # ------------------------------------------------------------------ #
    def insert(self, data):
        infos, dones = data[4], np.asarray(data[3])
        flat = []
        for th in infos:
            if isinstance(th, (list, tuple, np.ndarray)):
                flat.extend([x for x in th if isinstance(x, dict)])
            elif isinstance(th, dict):
                flat.append(th)
        for col, (key, fn) in _KEY.items():
            v = [r[key] for r in flat if key in r]
            if fn is not None:
                v = [fn(x) for x in v
                     if isinstance(x, (int, float, np.floating, np.integer))
                     and np.isfinite(x)]
            if v:
                self._acc[col].append(_m(v))
        rew = np.asarray(data[2], dtype=np.float64).reshape(len(infos), -1)
        self._ep_r += rew[:, 0]
        self._ep_t += 1.0
        for t in range(len(infos)):
            if np.all(dones[t]):
                self._eps.append(float(self._ep_r[t]))
                self._lens.append(float(self._ep_t[t]))
                self._ep_r[t] = self._ep_t[t] = 0.0
        super().insert(data)

    def after_update(self):
        self._write_row()
        super().after_update()

    def _write_row(self):
        if self._dbg_w is None:
            return
        self._rollout += 1
        row = {c: float("nan") for c in _COLS}
        row["env_step"] = self._rollout * self.T * self.nt
        row["rollout"] = self._rollout
        row["wall_s"] = round(time.time() - self._t0, 1)
        for c in _COLS:
            if self._acc.get(c):
                row[c] = _m(self._acc[c])
        row["trust_pol"] = self._g_fixed
        row["ep_return"] = _m(self._eps) if self._eps else float("nan")
        row["ep_len"] = _m(self._lens) if self._lens else float("nan")
        self._dbg_w.writerow([row[c] for c in _COLS])
        self._dbg.flush()
        li = max(1, int(self.algo_args["train"]["log_interval"]))
        if self._rollout % li == 0:
            print("[PACT-ANT] step=%d arm=%s | sigma=%.2f A=%.2f placebo=%.2f d=%.4f "
                  "| fit=%.3f bcos=%.3f cond=%.1f trust=%.2f/%.2f pred_err=%.4f "
                  "| dial_live=%.0f harmed=%.0f clip=%.0f | ret=%.1f len=%.0f"
                  % (row["env_step"], self.arm, row["sigma"], row["A"], row["placebo"],
                     row["d"], row["fit_gain"], row["beta_cos"], row["cond_psi"],
                     row["trust_pol"], row["trust_app"], row["pred_err"],
                     row["dial_live"], row["harmed"], row["actuator_clipped"],
                     row["ep_return"], row["ep_len"]), flush=True)
            self._warn(row)
        for c in self._acc:
            self._acc[c] = []
        self._eps, self._lens = [], []

    def _warn(self, row):
        if row["sigma"] > 0 and row["dial_live"] == 0.0 and self._rollout > 2:
            print("[PACT-ANT][REFUSE] severity is %.2f but the dial never produced a "
                  "non-zero disturbance.  This run MUST NOT be reported as a severity "
                  "arm (NS-3.3)." % row["sigma"])
        if np.isfinite(row["x_std"]) and row["x_std"] < 1e-9 and self._rollout > 2:
            print("[PACT-ANT][INERT] the trunk channels are not varying: there is "
                  "nothing to identify (gate 5).")
        if self.arm not in ("pactoff", "oracle", "intercept") \
                and np.isfinite(row["fit_gain"]) and row["fit_gain"] < 0.05 \
                and self._rollout > 25 and row["sigma"] > 0:
            print("[PACT-ANT][GATE 6] rolling fit_gain %.3f is below the floor: the "
                  "reduction is not holding here." % row["fit_gain"])
        if np.isfinite(row["cond_psi"]) and row["cond_psi"] > 1e3 and self._rollout > 25:
            print("[PACT-ANT][GATE 7][WARN] design-matrix condition number %.0f: beta "
                  "can be USED but not DECOMPOSED -- narrow the claim." % row["cond_psi"])
        if np.isfinite(row["diverged"]) and row["diverged"] > 0:
            print("[PACT-ANT][WARN] %.0f non-finite predictions so far -- trust was "
                  "zeroed there (the floor property)." % row["diverged"])
        if self.arm != "pactoff" and np.isfinite(row["trust_app"]) \
                and row["trust_app"] <= 1e-6 and self._rollout > 10:
            print("[PACT-ANT][ASLEEP] applied trust is 0 -- the confidence gate has "
                  "disarmed the compensator; this arm has become the plain host.")

    def close(self):
        try:
            if self._dbg is not None:
                self._dbg.close()
        finally:
            self._dbg = self._dbg_w = None
        super().close()


def _mk(arm, host):
    return type("PactAnt_%s_%s" % (arm, host.__name__), (PactAntMixin, host), dict(arm=arm))


#  EACH ARM'S RUNNER MUST MATCH ITS ACTOR, and that is not bookkeeping.
#
#  ``harl/algorithms/actors/__init__.py`` maps `pact`, `pactoff`, `pact_fixed`,
#  `pact_oracle` and `pact_intercept` to the MAPPO actor, and `pact_happo` /
#  `pact_happo_off` to HAPPO.  Pairing a MAPPO actor with ``OnPolicyHARunner``
#  runs HAPPO's sequential factor update around an actor whose ``update`` never
#  reads the factor: it does not crash, it just quietly trains something that is
#  neither HAPPO nor MAPPO -- and then an arm difference IS an algorithm
#  difference, which P-9.1 exists to prevent.  ``check_plumbing.py`` now asserts
#  this pairing for every arm.
#
#  So: the five spec arms run on the MAPPO host (matching `--algo mappo` as the
#  blind baseline), and `pact_happo` / `pact_happo_off` are the HAPPO-hosted pair
#  (matching `--algo happo`).  Running both pairs is what shows the result is not
#  an artefact of one host.
ANT_ARMS = {
    "pact": _mk("pact", OnPolicyMARunner),
    "pactoff": _mk("pactoff", OnPolicyMARunner),
    "pact_fixed": _mk("fixed", OnPolicyMARunner),
    "pact_oracle": _mk("oracle", OnPolicyMARunner),
    "pact_intercept": _mk("intercept", OnPolicyMARunner),
    "pact_happo": _mk("pact", OnPolicyHARunner),
    "pact_happo_off": _mk("pactoff", OnPolicyHARunner),
}
