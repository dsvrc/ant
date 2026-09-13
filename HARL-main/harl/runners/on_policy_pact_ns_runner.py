"""Runner for PACT-1 on SMAC-LANE -- the (C, invertible) cell.  An arm selector and
a logger, nothing else.

`PACT_NS_SPEC` P-9.1: arms go through the identical wrapper and differ only in the
trust term.  On an invertible channel the compensator lives INSIDE the severity
layer (II.6's first row, as in the football instance): it pre-rotates each move
order by the estimated swerve.  So an arm is a set of ENVIRONMENT flags, and every
arm's policy is the stock host network, byte for byte -- an arm difference cannot
be an algorithm difference.

THE ARMS
--------------------------------------------------------------------------------
    pactoff    estimator running, trust 0: identical to blind, order for order
    pact       fixed trust g = 0.9 (P-5.1's prior)
    fixed      the same as pact on this channel (trust is not learnable through a
               deterministic transform of the action: d log pi / d g = 0)
    oracle     compensates with the TRUE swerve, full trust -- the ceiling
    intercept  peer channels deleted: the (B)/(C) probe

THE DEBUG FILE
--------------------------------------------------------------------------------
Ordered so a first read goes top to bottom:  did the dial fire? -> did the swerve
reach the orders? -> is the medium loaded? -> is the METHOD working? -> is it
WINNING?  Worker clocks are de-phased, so the per-rollout means are cycle averages;
the *_peak / *_dry columns split the same sums by the phase the step or episode
was played in, which is where the separation shows inside one run.
"""

import csv
import os
import time

import numpy as np

from harl.envs.smac.smac_lane.layer import NS_KEYS as _LAYER_KEYS
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner

#: arm -> the environment flags that select it
ARMS = {
    "pact": dict(ns_pact=1, ns_g_fixed=0.9, ns_oracle=0, ns_intercept_only=0),
    "pactoff": dict(ns_pact=1, ns_g_fixed=0.0, ns_oracle=0, ns_intercept_only=0),
    "fixed": dict(ns_pact=1, ns_g_fixed=0.9, ns_oracle=0, ns_intercept_only=0),
    "oracle": dict(ns_pact=1, ns_oracle=1, ns_intercept_only=0),
    "intercept": dict(ns_pact=1, ns_g_fixed=0.9, ns_oracle=0, ns_intercept_only=1),
}

_META = ["env_step", "rollout", "wall_s", "arm"]
#: imported from the layer, never hand-maintained
_NS = list(_LAYER_KEYS)
#: ratios of SUMS, never means of per-step ratios
_DERIVED = ["exec_swerve_deg", "design_swerve_deg"]
_PHASE = ["exec_swerve_deg_peak", "exec_swerve_deg_dry", "design_swerve_deg_peak",
          "win_rate_peak", "win_rate_dry", "ep_return_peak", "ep_return_dry",
          "n_ep_peak", "n_ep_dry"]
_OUT = ["ep_len", "ep_return", "win_rate", "eval_win_rate"]
_COLS = _META + _NS + _DERIVED + _PHASE + _OUT


def _m(v):
    a = np.asarray([x for x in v if isinstance(x, (int, float, np.floating))],
                   dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


class PactNsMixin(object):
    """Arm selection + the instrument panel.  Mixed into whichever host runs."""

    arm = "pact"

    def __init__(self, args, algo_args, env_args):
        env_args = dict(env_args)
        env_args.update(ARMS[self.arm])
        algo_args = {k: dict(v) if isinstance(v, dict) else v
                     for k, v in algo_args.items()}
        # the policy is the stock host network in every arm: no tail, no shift
        algo_args["model"]["pact_n_actions"] = 0
        algo_args["model"]["pact_trust"] = "off"

        super().__init__(args, algo_args, env_args)
        self.T = int(self.algo_args["train"]["episode_length"])
        self.nt = int(self.algo_args["train"]["n_rollout_threads"])
        self._t0 = time.time()
        self._rollout = 0
        self._schema_checked = False
        self._acc = {c: [] for c in _NS}
        self._fresh()
        self._ep_r = np.zeros(self.nt)
        self._ep_t = np.zeros(self.nt)
        self._ep_A = np.zeros(self.nt)
        self._ep_wet = np.zeros(self.nt)
        self._last_eval_win = float("nan")
        self._dbg = self._dbg_w = None
        if not self.algo_args["render"]["use_render"]:
            self._open(os.path.join(str(self.run_dir), "pact_debug.csv"))
        print("[PACT-LANE] arm=%s flags=%s sigma=%s map=%s"
              % (self.arm, ARMS[self.arm], env_args.get("ns_severity"),
                 env_args.get("map_name")), flush=True)

    def _fresh(self):
        z = lambda: dict(exec=0.0, design=0.0, moves=0.0, wins=[], rets=[])  # noqa: E731
        self._all = z()
        self._ph = {"peak": z(), "dry": z()}
        self._eps, self._lens, self._wins = [], [], []

    @staticmethod
    def _step_phase(h):
        A, pl = h.get("ns_A", float("nan")), h.get("ns_placebo", float("nan"))
        if pl == 1.0:
            return "dry"
        if np.isfinite(A) and A >= 0.5:
            return "peak"
        return None

    def _open(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", newline="") as f:
                    head = next(csv.reader(f), None)
            except (OSError, StopIteration):
                head = None
            if head != _COLS:
                os.replace(path, path + ".%d.bak" % int(time.time()))
        new = not os.path.exists(path)
        self._dbg = open(path, "a", encoding="utf-8", newline="")
        self._dbg_w = csv.writer(self._dbg)
        if new:
            self._dbg_w.writerow(_COLS)
            self._dbg.flush()

    def _check_schema(self, row):
        self._schema_checked = True
        seen = {k for k in row if k.startswith("ns_") and not k.endswith("_i")}
        missing, dead = sorted(seen - set(_NS)), sorted(set(_NS) - seen)
        if missing or dead:
            print("[PACT-LANE][SCHEMA] not logged: %s | never arrive: %s"
                  % (missing, dead), flush=True)
        else:
            print("[PACT-LANE][SCHEMA] %d ns_* diagnostics, all plumbed." % len(seen),
                  flush=True)

    def insert(self, data):
        infos, dones = data[4], np.asarray(data[3])
        flat, heads = [], []
        for th in infos:
            ds = ([x for x in th if isinstance(x, dict)]
                  if isinstance(th, (list, tuple, np.ndarray))
                  else ([th] if isinstance(th, dict) else []))
            flat.extend(ds)
            heads.append(ds[0] if ds else None)   # one ns_* block per worker
        if flat and not self._schema_checked:
            self._check_schema(flat[0])
        for col in _NS:
            v = [r[col] for r in flat if col in r]
            if v:
                self._acc[col].append(_m(v))
        for h in heads:
            if h is None:
                continue
            num, den = h.get("ns_harm_num", np.nan), h.get("ns_harm_den", np.nan)
            dd = h.get("ns_d_deg", np.nan)
            if not (np.isfinite(num) and np.isfinite(den)) or den <= 0:
                continue
            design = float(np.radians(dd)) * den if np.isfinite(dd) else 0.0
            ph = self._step_phase(h)
            for acc in [self._all] + ([self._ph[ph]] if ph is not None else []):
                acc["exec"] += float(num)
                acc["design"] += design
                acc["moves"] += float(den)
        rew = np.asarray(data[2], dtype=np.float64).reshape(len(infos), -1)
        self._ep_r += rew[:, 0]
        self._ep_t += 1.0
        for t, th in enumerate(infos):
            row = th[0] if isinstance(th, (list, tuple, np.ndarray)) and len(th) \
                else (th if isinstance(th, dict) else None)
            h = heads[t]
            if h is not None:
                A = h.get("ns_A", float("nan"))
                self._ep_A[t] += float(A) if np.isfinite(A) else 0.0
                self._ep_wet[t] += 0.0 if h.get("ns_placebo", 0.0) == 1.0 else 1.0
            if np.all(dones[t]):
                ret, n_t = float(self._ep_r[t]), float(self._ep_t[t])
                self._eps.append(ret)
                self._lens.append(n_t)
                won = None
                if isinstance(row, dict):
                    won = 1.0 if row.get("won", False) else 0.0
                    self._wins.append(won)
                ph = ("dry" if self._ep_wet[t] == 0.0
                      else ("peak" if self._ep_A[t] / max(1.0, n_t) >= 0.5 else None))
                if ph is not None:
                    self._ph[ph]["rets"].append(ret)
                    if won is not None:
                        self._ph[ph]["wins"].append(won)
                self._ep_r[t] = self._ep_t[t] = 0.0
                self._ep_A[t] = self._ep_wet[t] = 0.0
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

        def deg(acc, key):
            return (float(np.degrees(acc[key] / acc["moves"]))
                    if acc["moves"] > 0 else float("nan"))

        r["exec_swerve_deg"] = deg(self._all, "exec")
        r["design_swerve_deg"] = deg(self._all, "design")
        r["exec_swerve_deg_peak"] = deg(self._ph["peak"], "exec")
        r["exec_swerve_deg_dry"] = deg(self._ph["dry"], "exec")
        r["design_swerve_deg_peak"] = deg(self._ph["peak"], "design")
        for p in ("peak", "dry"):
            acc = self._ph[p]
            r["win_rate_" + p] = _m(acc["wins"]) if acc["wins"] else float("nan")
            r["ep_return_" + p] = _m(acc["rets"]) if acc["rets"] else float("nan")
            r["n_ep_" + p] = float(len(acc["rets"]))
        r["win_rate"] = _m(self._wins) if self._wins else float("nan")
        r["ep_return"] = _m(self._eps) if self._eps else float("nan")
        r["ep_len"] = _m(self._lens) if self._lens else float("nan")
        r["eval_win_rate"] = float(self._last_eval_win)
        self._dbg_w.writerow([r[c] for c in _COLS])
        self._dbg.flush()

        li = max(1, int(self.algo_args["train"]["log_interval"]))
        if self._rollout % li == 0:
            print("\n".join([
                "[PACT-LANE] step=%d arm=%s" % (r["env_step"], self.arm),
                "   dial   sigma=%.2f A=%.3f amp=%.3f placebo=%.2f dial_ratio=%.2f"
                % (r["ns_sigma"], r["ns_A"], r["ns_amp"], r["ns_placebo"],
                   r["ns_dial_ratio"]),
                "   swerve design=%.2f deg  executed=%.2f deg  (peak: design %.2f, "
                "executed %.2f | placebo executed %.3f)  clipped=%.0f"
                % (r["design_swerve_deg"], r["exec_swerve_deg"],
                   r["design_swerve_deg_peak"], r["exec_swerve_deg_peak"],
                   r["exec_swerve_deg_dry"], r["ns_corr_clipped"]),
                "   medium moves=%.2f lane_frac=%.2f x_front=%.3f x_flank=%.3f "
                "spread=%.2f" % (r["ns_moves"], r["ns_lane_frac"], r["ns_x_front"],
                                 r["ns_x_flank"], r["ns_spread"]),
                "   method fit_gain=%.3f beta_cos=%.4f relerr=%.3f cond=%.1f conf=%.3f "
                "mem/period=%.3f trust=%.2f"
                % (r["ns_fit_gain"], r["ns_beta_cos"], r["ns_beta_relerr"],
                   r["ns_cond_psi"], r["ns_conf"], r["ns_mem_frac"], r["ns_trust"]),
                "   PHASE  peak: win=%.3f (n=%.0f)  |  placebo: win=%.3f (n=%.0f)"
                % (r["win_rate_peak"], r["n_ep_peak"], r["win_rate_dry"],
                   r["n_ep_dry"]),
                "   score  win=%.3f eval_win=%.3f ret=%.2f len=%.0f"
                % (r["win_rate"], r["eval_win_rate"], r["ep_return"], r["ep_len"]),
            ]), flush=True)
            self._warn(r)
        for c in self._acc:
            self._acc[c] = []
        self._fresh()

    def _warn(self, r):
        n = self._rollout
        comp = self.arm in ("pact", "fixed", "oracle", "intercept")
        if r["ns_sigma"] > 0 and n > 5 and r["ns_dial_ratio"] == 0.0:
            print("[PACT-LANE][REFUSE] sigma=%.2f but the dial never went live (NS-3.3)."
                  % r["ns_sigma"], flush=True)
        if np.isfinite(r["exec_swerve_deg_dry"]) and r["exec_swerve_deg_dry"] > 0.0:
            print("[PACT-LANE][PLACEBO-HARM] %.3g deg of swerve in the placebo, not 0 "
                  "(NS-2.5)." % r["exec_swerve_deg_dry"], flush=True)
        if r["ns_sigma"] > 0 and n > 5 and np.isfinite(r["design_swerve_deg_peak"]) \
                and r["design_swerve_deg_peak"] < 2.0:
            print("[PACT-LANE][WEAK] only %.2f deg of designed swerve at the peak: the "
                  "squad rarely moves through each other's lanes (lane_frac=%.2f). "
                  "Raise ns_severity (labelled beyond-physical)."
                  % (r["design_swerve_deg_peak"], r["ns_lane_frac"]), flush=True)
        if np.isfinite(r["ns_mem_frac"]) and r["ns_mem_frac"] > 0.3 and n > 5:
            print("[PACT-LANE][NOT-TRACKING] estimator memory is %.2f of the period."
                  % r["ns_mem_frac"], flush=True)
        if comp and np.isfinite(r["ns_fit_gain"]) and r["ns_fit_gain"] <= 0.0 and n > 8:
            print("[PACT-LANE][NO-REDUCTION] fit_gain=%.3f <= 0: the lane channels "
                  "predict no better than an intercept." % r["ns_fit_gain"], flush=True)
        if comp and self.arm != "intercept" and np.isfinite(r["ns_beta_cos"]) \
                and r["ns_beta_cos"] < 0.9 and n > 20:
            print("[PACT-LANE][NOT-IDENTIFIED] beta_cos=%.3f: the estimate does not "
                  "point at the true gain, so compensation cannot cancel the swerve."
                  % r["ns_beta_cos"], flush=True)

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
