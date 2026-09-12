"""Runner for PACT-1 on SMAC-NS.  An arm selector and a logger, nothing else.

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
    fixed      trust held at a constant.  Given the inverted prior, this is the
               only arm that answers "is trust learned, or well-initialised?"
    oracle     steered on the TRUE excess.  The ceiling; the method cannot beat it.
    intercept  peer channels deleted, everything else identical.

THE DEBUG FILE
--------------------------------------------------------------------------------
Ordered so a first read goes top to bottom:

    did the dial fire?  ->  did the harm reach the game?  ->  is the medium
    loaded?  ->  is the METHOD working?  ->  is it WINNING?

The last group is kept apart from the one before it because the method can work
perfectly and still not win, and that is a statement about the domain's headroom
rather than a bug.

``_check_schema`` compares the declared columns against what the layer actually
emits, once, out loud.  The first run of this pipeline wrote NaN into every
``ns_*`` column because a diagnostic was never plumbed through -- and a silently
missing diagnostic is indistinguishable from a silently inert dial, which is
NS-3.3's failure mode arriving through the instrumentation instead of the physics.
"""

import csv
import os
import time

import numpy as np

from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner

#: arm -> (pact_trust, ns_oracle, ns_intercept_only)
ARMS = {
    "pact": ("learned", 0, 0),
    "pactoff": ("off", 0, 0),
    "fixed": ("fixed", 0, 0),
    "oracle": ("learned", 1, 0),
    "intercept": ("learned", 0, 1),
}

_META = ["env_step", "rollout", "wall_s", "arm"]

#: Every ``ns_*`` key the layer emits, logged without exception.
_NS = [
    # ---- is the DIAL live? ------------------------------------------------
    "ns_sigma", "ns_on", "ns_A", "ns_g", "ns_g_min", "ns_placebo",
    "ns_dial_ratio", "ns_clock",
    # ---- did the harm REACH the game?  (NS-3.2 / NS-3.3) -----------------
    "ns_harmed", "ns_restored", "ns_debug_fail",
    "ns_dmg_dealt", "ns_dmg_wasted", "ns_waste_frac", "ns_waste_frac_cum",
    # ---- is the MEDIUM loaded?  (III.1 Q7: if not, nothing can bite) -----
    "ns_u", "ns_u_max", "ns_excess", "ns_excess_max", "ns_y",
    "ns_fire_frac", "ns_alive", "ns_herd_index", "ns_switch_frac",
    # ---- is the METHOD working? ------------------------------------------
    "ns_fit_gain", "ns_pred_gain", "ns_cond_psi", "ns_conf", "ns_rows",
    "ns_clip_frac", "ns_innov", "ns_trP", "ns_x_std", "ns_x_mean",
    "ns_cost_std",
    "ns_beta0", "ns_beta1", "ns_beta2", "ns_beta3", "ns_x0", "ns_x1", "ns_x2",
]

_TRUST = ["trust_pol"]                       # read off the actor, not the env
_OUT = ["ep_len", "ep_return", "win_rate", "eval_win_rate"]

_COLS = _META + _NS + _TRUST + _OUT


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
        env_args["ns_augment"] = 1           # every PACT-family arm, so the host
        env_args["ns_oracle"] = int(oracle)  # network is identical across them
        env_args["ns_intercept_only"] = int(intercept)

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
        self._schema_checked = False
        self._acc = {c: [] for c in _NS}
        self._eps, self._lens, self._wins = [], [], []
        self._ep_r = np.zeros(self.nt)
        self._ep_t = np.zeros(self.nt)
        self._last_eval_win = float("nan")
        self._dbg = self._dbg_w = None
        if not self.algo_args["render"]["use_render"]:
            self._open(os.path.join(str(self.run_dir), "pact_debug.csv"))
        print("[PACT-NS] arm=%s trust=%s oracle=%d intercept=%d n_actions=%d "
              "sigma=%s" % (self.arm, trust, oracle, intercept, n_act,
                            env_args.get("ns_severity")))

    def _open(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", newline="") as f:
                    head = next(csv.reader(f), None)
            except (OSError, StopIteration):
                head = None
            if head != _COLS:
                # never append across a schema change: two runs with different
                # column counts in one file misalign every field in the second
                os.replace(path, path + ".%d.bak" % int(time.time()))
        new = not os.path.exists(path)
        self._dbg = open(path, "a", encoding="utf-8", newline="")
        self._dbg_w = csv.writer(self._dbg)
        if new:
            self._dbg_w.writerow(_COLS)
            self._dbg.flush()

    # ------------------------------------------------------------------ #
    def _check_schema(self, row):
        """Name any ``ns_*`` key the layer emits that this file does not log,
        and any column that never arrives.  Once, out loud."""
        self._schema_checked = True
        seen = {k for k in row if k.startswith("ns_") and not k.endswith("_i")}
        missing = sorted(seen - set(_NS))
        dead = sorted(set(_NS) - seen)
        if missing:
            print("[PACT-NS][SCHEMA] the layer emits %d key(s) this file does NOT "
                  "log: %s" % (len(missing), ", ".join(missing)), flush=True)
        if dead:
            print("[PACT-NS][SCHEMA] %d column(s) never arrive from the layer and "
                  "will read NaN: %s" % (len(dead), ", ".join(dead)), flush=True)
        if not missing and not dead:
            print("[PACT-NS][SCHEMA] %d ns_* diagnostics, all plumbed."
                  % len(seen), flush=True)

    def insert(self, data):
        infos, dones = data[4], np.asarray(data[3])
        flat = []
        for th in infos:
            if isinstance(th, (list, tuple, np.ndarray)):
                flat.extend([x for x in th if isinstance(x, dict)])
            elif isinstance(th, dict):
                flat.append(th)
        if flat and not self._schema_checked:
            self._check_schema(flat[0])
        # every ns_* column straight through -- no hand-maintained key map to
        # fall out of date, which is how the first run lost all of them
        for col in _NS:
            v = [r[col] for r in flat if col in r]
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

    def eval(self, *a, **k):
        out = super().eval(*a, **k)
        try:
            lg = self.logger
            n = max(1, int(getattr(lg, "eval_episode", 0)
                           or self.algo_args["eval"]["eval_episodes"]))
            self._last_eval_win = float(getattr(lg, "eval_battles_won", 0)) / n
        except Exception:
            pass
        return out

    def after_update(self):
        self._write_row()
        super().after_update()

    def _trust_pol(self):
        """The trust the POLICY has set.  P-5.3 wants policy-set and applied
        reliance side by side; ``ns_conf`` is the estimator's own prediction
        confidence, so both are in the file."""
        import torch
        a = self.actor[0]
        m = getattr(a, "actor", a)
        w = getattr(m, "pact_w", None)
        if w is None:
            return float(getattr(m, "pact_gfixed", 0.0)) \
                if getattr(m, "pact_mode", "off") == "fixed" else 0.0
        b = float(getattr(m, "pact_bias", 2.2))
        return float(torch.sigmoid(w.detach() + b).item())

    def _write_row(self):
        if self._dbg_w is None:
            return
        self._rollout += 1
        r = {c: float("nan") for c in _COLS}
        r["env_step"] = self._rollout * self.T * self.nt
        r["rollout"] = self._rollout
        r["wall_s"] = round(time.time() - self._t0, 1)
        r["arm"] = self.arm
        for c in _NS:
            if self._acc.get(c):
                r[c] = _m(self._acc[c])
        r["trust_pol"] = self._trust_pol()
        r["win_rate"] = _m(self._wins) if self._wins else float("nan")
        r["ep_return"] = _m(self._eps) if self._eps else float("nan")
        r["ep_len"] = _m(self._lens) if self._lens else float("nan")
        r["eval_win_rate"] = float(self._last_eval_win)
        self._dbg_w.writerow([r[c] for c in _COLS])
        self._dbg.flush()

        li = max(1, int(self.algo_args["train"]["log_interval"]))
        if self._rollout % li == 0:
            lines = [
                "[PACT-NS] step=%d arm=%s" % (r["env_step"], self.arm),
                "   dial   sigma=%.2f g=%.4f g_min=%.4f placebo=%.2f "
                "dial_ratio=%.2f" % (r["ns_sigma"], r["ns_g"], r["ns_g_min"],
                                     r["ns_placebo"], r["ns_dial_ratio"]),
                "   harm   harmed=%.0f hp=%.1f fail=%.0f | dealt=%.1f "
                "wasted=%.3f waste_frac=%.4f cum=%.4f"
                % (r["ns_harmed"], r["ns_restored"], r["ns_debug_fail"],
                   r["ns_dmg_dealt"], r["ns_dmg_wasted"], r["ns_waste_frac"],
                   r["ns_waste_frac_cum"]),
                "   medium u=%.4f(max %.4f) excess=%.4f y=%.4f fire=%.2f "
                "herd=%.3f switch=%.2f"
                % (r["ns_u"], r["ns_u_max"], r["ns_excess"], r["ns_y"],
                   r["ns_fire_frac"], r["ns_herd_index"], r["ns_switch_frac"]),
                "   method fit_gain=%.4f pred_gain=%.4f cond=%.1f conf=%.3f "
                "x_std=%.4f beta=[%.3f %.3f %.3f] rows=%.0f"
                % (r["ns_fit_gain"], r["ns_pred_gain"], r["ns_cond_psi"],
                   r["ns_conf"], r["ns_x_std"], r["ns_beta0"], r["ns_beta1"],
                   r["ns_beta2"], r["ns_rows"]),
                "   score  trust_pol=%.3f win=%.3f eval_win=%.3f ret=%.2f len=%.0f"
                % (r["trust_pol"], r["win_rate"], r["eval_win_rate"],
                   r["ep_return"], r["ep_len"]),
            ]
            print("\n".join(lines), flush=True)
            self._warn(r)
        for c in self._acc:
            self._acc[c] = []
        self._eps, self._lens, self._wins = [], [], []

    # ------------------------------------------------------------------ #
    def _warn(self, r):
        """Each of these is a failure the spec paid for, detected rather than
        assumed.  Read them before reading any margin."""
        n = self._rollout
        if r["ns_sigma"] > 0 and r["ns_harmed"] == 0.0 and n > 2:
            print("[PACT-NS][REFUSE] severity is %.2f but nothing was harmed. "
                  "This run MUST NOT be reported as a severity arm (NS-3.3)."
                  % r["ns_sigma"], flush=True)
        if np.isfinite(r["ns_debug_fail"]) and r["ns_debug_fail"] > 0:
            print("[PACT-NS][WARN] the engine refused %.0f harm writes -- the dial "
                  "is not fully reaching the records (NS-3.2)."
                  % r["ns_debug_fail"], flush=True)
        if np.isfinite(r["ns_waste_frac_cum"]) and r["ns_waste_frac_cum"] < 1e-3 \
                and r["ns_sigma"] > 0 and n > 5:
            print("[PACT-NS][WEAK] cumulative waste is %.5f of damage dealt. The "
                  "dial is live but is taking almost nothing -- raise ns_alpha "
                  "(the declared performance coefficient) or check ns_u."
                  % r["ns_waste_frac_cum"], flush=True)
        if np.isfinite(r["ns_u"]) and r["ns_u"] < 1e-3 and n > 5:
            print("[PACT-NS][UNLOADED] u = %.5f: the medium is far below its "
                  "limit, so a capacity loss cannot express itself however large "
                  "sigma grows (III.1 Q7).  Are the agents focus-firing at all? "
                  "herd_index=%.3f" % (r["ns_u"], r["ns_herd_index"]), flush=True)
        if np.isfinite(r["ns_x_std"]) and r["ns_x_std"] < 1e-6 and n > 5:
            print("[PACT-NS][INERT] the channels are not varying: there is nothing "
                  "to identify (gate 5).", flush=True)
        if np.isfinite(r["ns_fit_gain"]) and r["ns_fit_gain"] <= 0.0 and n > 8:
            print("[PACT-NS][NO-REDUCTION] fit_gain=%.4f <= 0: the peer channels "
                  "predict no better than an intercept-only null. The reduction "
                  "does not hold here and a full run cannot fix it (gate 6)."
                  % r["ns_fit_gain"], flush=True)
        if not np.isfinite(r["ns_cond_psi"]) and n > 5:
            print("[PACT-NS][COND] cond_psi is non-finite: beta may be usable but "
                  "is NOT decomposable.  That narrows the claim; it does not break "
                  "the run (gate 7 warns, it does not abort).", flush=True)
        if self.arm != "pactoff" and np.isfinite(r["trust_pol"]) \
                and r["trust_pol"] <= 1e-6 and n > 5:
            print("[PACT-NS][ASLEEP] policy trust has collapsed to 0 -- this arm "
                  "has become the plain host.", flush=True)

    def close(self):
        try:
            env = getattr(self, "envs", None)
            if env is not None and hasattr(env, "ns_close_report"):
                env.ns_close_report()
        except Exception:
            pass
        try:
            if self._dbg is not None:
                self._dbg.close()
        finally:
            self._dbg = self._dbg_w = None
        super().close()


def _mk(arm, host):
    return type("PactNs_%s_%s" % (arm, host.__name__), (PactNsMixin, host),
                dict(arm=arm))


PactNsRunner = _mk("pact", OnPolicyMARunner)
PactNsOffRunner = _mk("pactoff", OnPolicyMARunner)
PactNsFixedRunner = _mk("fixed", OnPolicyMARunner)
PactNsOracleRunner = _mk("oracle", OnPolicyMARunner)
PactNsInterceptRunner = _mk("intercept", OnPolicyMARunner)
PactNsHappoRunner = _mk("pact", OnPolicyHARunner)
PactNsHappoOffRunner = _mk("pactoff", OnPolicyHARunner)
