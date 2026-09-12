"""Runner for PACT-1 on SMAC-NS.  A logger and an arm selector, nothing else.

`PACT_NS_SPEC` P-9.1: *"Ship three arms through the identical wrapper, differing
only in the trust term."*  So this runner does exactly two things beyond the stock
host: it sets the flags that pick an arm, and it writes II.10's instrument panel.
The PPO update, the clipped surrogate, the entropy bonus and the optimiser are the
host's own, untouched -- an arm difference must not be able to be an algorithm
difference.

THE ARMS
--------------------------------------------------------------------------------
    pactoff    trust forced to 0.  BIT-IDENTICAL to the untouched host: the shift
               function returns the logits unchanged and the trust parameter is
               never created, so the base network is the stock one.
    pact       trust learned through the host's own objective.
    fixed      trust held at a constant.  Not optional politeness -- given the
               inverted prior, it is the only arm that answers "is trust learned,
               or merely well-initialised?"
    oracle     steered on the TRUE excess.  The ceiling; the method cannot beat it.
    intercept  peer channels deleted, everything else identical.  Separates "the
               peer coupling explained something" from "the intercept memorised
               each agent's typical residual".

THE PANEL (II.10)
--------------------------------------------------------------------------------
"Is the method working" and "is it winning" are kept in SEPARATE columns, because
the method can work perfectly and still not win, and that is a statement about how
much headroom the domain has rather than a bug.
"""

import csv
import os
import time

import numpy as np

from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.runners.on_policy_ha_runner import OnPolicyHARunner

#: arm -> (pact_trust, ns_oracle, ns_intercept_only)
ARMS = {
    "pact": ("learned", 0, 0),
    "pactoff": ("off", 0, 0),
    "fixed": ("fixed", 0, 0),
    "oracle": ("learned", 1, 0),
    "intercept": ("learned", 0, 1),
}

_COLS = [
    "env_step", "rollout", "wall_s",
    # is the dial live?  (every arm, baselines included)
    "sigma", "A", "g", "placebo", "harmed_steps", "hp_restored", "debug_fail",
    # is the NS biting?
    "u", "excess", "y", "fire_frac",
    # is the METHOD working?  (II.10 -- kept apart from "is it winning")
    "fit_gain", "pred_gain", "cond_psi", "trust_pol", "trust_app",
    "x_std", "herd_index", "clip_frac", "rows",
    # is it WINNING?
    "ep_len", "ep_return", "win_rate",
]


def _m(v):
    a = np.asarray([x for x in v if isinstance(x, (int, float, np.floating))],
                   dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


class PactNsMixin(object):
    """Arm selection + the instrument panel.  Mixed into whichever host runs."""

    arm = "pact"

    def __init__(self, args, algo_args, env_args):
        trust, oracle, intercept = ARMS[self.arm]
        env_args = dict(env_args)
        env_args["ns_augment"] = 1          # every PACT-family arm, so the host
        env_args["ns_oracle"] = int(oracle)  # network is identical across them
        env_args["ns_intercept_only"] = int(intercept)

        # the trust head's configuration reaches the actor through model args
        from harl.envs.smac.smac_maps import get_map_params
        n_act = int(get_map_params(env_args["map_name"])["n_enemies"]) + 6
        algo_args = {k: dict(v) if isinstance(v, dict) else v
                     for k, v in algo_args.items()}
        algo_args["model"]["pact_n_actions"] = n_act
        algo_args["model"]["pact_trust"] = trust
        algo_args["model"]["pact_kappa"] = float(env_args.get("ns_kappa", 1.0))
        algo_args["model"]["pact_g_max"] = float(env_args.get("ns_g_max", 1.0))
        algo_args["model"]["pact_g_fixed"] = float(env_args.get("ns_g_fixed", 0.9))

        super().__init__(args, algo_args, env_args)
        self.T = int(self.algo_args["train"]["episode_length"])
        self.nt = int(self.algo_args["train"]["n_rollout_threads"])
        self._t0 = time.time()
        self._rollout = 0
        self._acc = {c: [] for c in _COLS}
        self._eps, self._lens, self._wins = [], [], []
        self._ep_r = np.zeros(self.nt)
        self._ep_t = np.zeros(self.nt)
        self._dbg = self._dbg_w = None
        if not self.algo_args["render"]["use_render"]:
            self._open(os.path.join(str(self.run_dir), "pact_debug.csv"))
        print("[PACT-NS] arm=%s  trust=%s  oracle=%d  intercept=%d  n_actions=%d"
              % (self.arm, trust, oracle, intercept, n_act))

    def _open(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", newline="") as f:
                    head = next(csv.reader(f), None)
            except (OSError, StopIteration):
                head = None
            if head != _COLS:
                # never append across a schema change: two runs with different
                # column counts in one file misalign every field in the second.
                os.replace(path, path + ".%d.bak" % int(time.time()))
        new = not os.path.exists(path)
        self._dbg = open(path, "a", encoding="utf-8", newline="")
        self._dbg_w = csv.writer(self._dbg)
        if new:
            self._dbg_w.writerow(_COLS)
            self._dbg.flush()

    # ------------------------------------------------------------------ #
    _KEY = {"sigma": "ns_sigma", "A": "ns_A", "g": "ns_g", "placebo": "ns_placebo",
            "harmed_steps": "ns_harmed", "hp_restored": "ns_restored",
            "debug_fail": "ns_debug_fail", "u": "ns_u", "excess": "ns_excess",
            "y": "ns_y", "fire_frac": "ns_fire_frac", "trust_app": "ns_conf",
            "x_std": "ns_x_std", "clip_frac": "ns_clip_frac", "rows": "ns_rows"}

    def insert(self, data):
        infos, dones = data[4], np.asarray(data[3])
        flat = []
        for th in infos:
            if isinstance(th, (list, tuple, np.ndarray)):
                flat.extend([x for x in th if isinstance(x, dict)])
            elif isinstance(th, dict):
                flat.append(th)
        for col, key in self._KEY.items():
            v = [r[key] for r in flat if key in r]
            if v:
                self._acc[col].append(_m(v))
        rew = np.asarray(data[2], dtype=np.float64).reshape(len(infos), -1)
        self._ep_r += rew[:, 0]
        self._ep_t += 1.0
        for t, th in enumerate(infos):
            row = th[0] if isinstance(th, (list, tuple, np.ndarray)) and len(th) \
                else (th if isinstance(th, dict) else None)
            if np.all(dones[t]):
                self._eps.append(float(self._ep_r[t]))
                self._lens.append(float(self._ep_t[t]))
                self._ep_r[t] = self._ep_t[t] = 0.0
                if isinstance(row, dict):
                    self._wins.append(1.0 if row.get("won", False) else 0.0)
        super().insert(data)

    def after_update(self):
        self._write_row()
        super().after_update()

    def _trust_pol(self):
        """The trust the POLICY has set, as opposed to the confidence gate's.
        P-5.3: report both -- they diverge exactly when the gate misbehaves."""
        import torch
        a = self.actor[0]
        w = getattr(getattr(a, "actor", a), "pact_w", None)
        if w is None:
            m = getattr(getattr(a, "actor", a), "pact_mode", "off")
            return float(getattr(getattr(a, "actor", a), "pact_gfixed", 0.0)) \
                if m == "fixed" else 0.0
        b = float(getattr(getattr(a, "actor", a), "pact_bias", 2.2))
        return float(torch.sigmoid(w.detach() + b).item())

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
        row["trust_pol"] = self._trust_pol()
        row["win_rate"] = _m(self._wins) if self._wins else float("nan")
        row["ep_return"] = _m(self._eps) if self._eps else float("nan")
        row["ep_len"] = _m(self._lens) if self._lens else float("nan")
        self._dbg_w.writerow([row[c] for c in _COLS])
        self._dbg.flush()
        li = max(1, int(self.algo_args["train"]["log_interval"]))
        if self._rollout % li == 0:
            print("[PACT-NS] step=%d arm=%s | sigma=%.2f g=%.3f placebo=%.2f "
                  "u=%.4f excess=%.4f | trust_pol=%.3f trust_app=%.3f "
                  "x_std=%.3f | harmed=%.0f hp=%.0f | win=%.3f"
                  % (row["env_step"], self.arm, row["sigma"], row["g"],
                     row["placebo"], row["u"], row["excess"], row["trust_pol"],
                     row["trust_app"], row["x_std"], row["harmed_steps"],
                     row["hp_restored"], row["win_rate"]), flush=True)
            self._warn(row)
        for c in self._acc:
            self._acc[c] = []
        self._eps, self._lens, self._wins = [], [], []

    def _warn(self, row):
        if row["sigma"] > 0 and row["harmed_steps"] == 0.0 and self._rollout > 2:
            print("[PACT-NS][REFUSE] severity is %.2f but nothing was harmed. "
                  "This run MUST NOT be reported as a severity arm (NS-3.3)."
                  % row["sigma"])
        if np.isfinite(row["debug_fail"]) and row["debug_fail"] > 0:
            print("[PACT-NS][WARN] the engine refused %.0f harm writes -- the dial "
                  "is not fully reaching the records." % row["debug_fail"])
        if np.isfinite(row["x_std"]) and row["x_std"] < 1e-6 and self._rollout > 2:
            print("[PACT-NS][INERT] the channels are not varying: there is nothing "
                  "to identify (gate 5).")
        if self.arm != "pactoff" and np.isfinite(row["trust_pol"]) \
                and row["trust_pol"] <= 1e-6 and self._rollout > 5:
            print("[PACT-NS][ASLEEP] policy trust has collapsed to 0 -- this arm "
                  "has become the plain host.")

    def close(self):
        try:
            if self._dbg is not None:
                self._dbg.close()
        finally:
            self._dbg = self._dbg_w = None
        super().close()


def _mk(arm, host):
    ns = dict(arm=arm)
    return type("PactNs_%s_%s" % (arm, host.__name__), (PactNsMixin, host), ns)


# MAPPO host (shared parameters -- the standard SMAC baseline) and HAPPO host.
PactNsRunner = _mk("pact", OnPolicyMARunner)
PactNsOffRunner = _mk("pactoff", OnPolicyMARunner)
PactNsFixedRunner = _mk("fixed", OnPolicyMARunner)
PactNsOracleRunner = _mk("oracle", OnPolicyMARunner)
PactNsInterceptRunner = _mk("intercept", OnPolicyMARunner)
PactNsHappoRunner = _mk("pact", OnPolicyHARunner)
PactNsHappoOffRunner = _mk("pactoff", OnPolicyHARunner)
