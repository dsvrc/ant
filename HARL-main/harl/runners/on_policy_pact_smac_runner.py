"""SMAC runner for `--algo pact` — training is BIT-IDENTICAL to HAPPO; it only adds a
detailed debug file (`pact_debug.csv`) so you can see, per rollout, exactly what the
non-stationarity is doing and whether PACT is responding to it.

The env (`StarCraft2_Env.py`, Coupled Weapon Overheat) throttles each unit's fire by a
shared-load drop probability `ell_i = A(t)·σ·x2_i`, and appends the computed load `x2_i`
to the PACT policy's obs.  This runner reads the env's per-step telemetry and writes one
row per rollout with three groups of columns:

  IS THE NS BITING? (severity felt)
    A_mean        driver A(t) — the exogenous engagement tempo, in [0,1]
    ell_mean/max  the drop probability actually applied (mean/max over units).  This is
                  the effective severity — calibrate SEVERITY so ell_peak ~ 0.3-0.6.
    drop_frac     fraction of COMMANDED attacks that were jammed (the real harm)
    x2_mean       the shared overheat load

  IS PACT COORDINATING? (the whole point — does it stagger firing?)
    fire_frac     fraction of live units attacking
    fire_avail    fraction of live units that COULD attack (had a target in range)
    hold_frac     of the units that COULD attack, the fraction that held fire.  THIS
                  is the decision variable; fire_frac alone is diluted by units with
                  nothing to shoot at.
    throughput    fire_frac·(1−drop_frac) — the fraction of units actually LANDING a
                  shot, i.e. the quantity the team is trying to maximize.  A team that
                  is coordinating correctly RAISES this while LOWERING fire_frac.
    hold_gap      hold_frac(peak) − hold_frac(trough).  ***LEGACY — from the DROP
                  channel, where the only response was to fire less.*** The harm is now
                  a target DEFLECTION, which is compensated by re-aiming at zero cost,
                  so a correctly-behaving policy holds fire no more than a blind one and
                  hold_gap should sit at ~0.  Read `drop_frac` (fraction of shots
                  deflected) and the return instead; the compensation ceiling is B0.
    fire_avail    fraction of live units with an enemy in range.  ***WATCH THIS.***  If
                  it collapses (measured 0.89 → 0.22) while ep_len pins at the limit,
                  the team has stopped fighting rather than learned to stagger — the
                  shaped-reward basin, not a coordination result. The runner prints
                  [PACT][DISENGAGED] when it does.
    x2_spread     max−min of x2 over live units.  x2_i differs across agents only by
                  the excluded own-fire term (≤(1−ρ)/(N−1)=0.021 on 3s5z), so this is
                  ~0: x2 says HOW HOT the bus is, never WHO should hold fire — the team
                  must break the symmetry itself.
    x3_mean       own shots jammed lately   } their ratio is the local estimate of ell,
    x3try_mean    own shots attempted lately} i.e. the only decentralized read on the
                  hidden driver.  x3try≈0 means the agent has no evidence right now.

    stagger_gap   fire_lo_load − fire_hi_load.  ***BIASED — DO NOT READ AS
                  COORDINATION.***  x2_i excludes agent i's own fire, so ranking agents
                  by x2_i is very nearly reverse-ranking them by their OWN recent
                  firing, which is autocorrelated with firing now.  On the 20M-step
                  3s5z run it read +0.16 right through the severity-0 warmup, where no
                  NS exists at all.  Kept only for continuity with old runs; use
                  hold_gap.

  ...split at the driver PEAK (A>0.7, where the overheat is worst) and TROUGH (A<0.3),
  plus reward-per-step and episode length.

Win-rate / return are logged by the stock SMAC logger (progress.txt / TensorBoard) —
which now also splits the EVAL win-rate by driver phase.  Read progress.txt's
per-phase columns, not the aggregate: with in-phase eval envs the aggregate is a
single-phase snapshot (see harl/utils/envs_tools.py::_snd_dephase).
"""

import csv
import os

import numpy as np

from harl.runners.on_policy_ha_runner import OnPolicyHARunner

_COLS = [
    "env_step", "rollout", "severity",
    # is the NS biting?
    "A_mean", "ell_mean", "ell_max", "drop_frac", "x2_mean", "x2_spread",
    "x3_mean", "x3try_mean",
    # is PACT coordinating (staggering fire on the shared load)?
    "fire_frac", "fire_avail", "hold_frac", "throughput",
    "fire_hi_load", "fire_lo_load", "stagger_gap",  # stagger_gap is BIASED, see module doc
    # at the driver PEAK (worst overheat) and TROUGH:
    "ell_peak", "drop_peak", "fire_frac_peak", "hold_frac_peak", "throughput_peak",
    "stagger_gap_peak",
    "ell_trough", "fire_frac_trough", "hold_frac_trough", "throughput_trough",
    # THE headline coordination number: hold more when the bus is hot.
    "hold_gap",
    # outcome:
    "r_step_mean", "ep_len_mean",
    # smacv2/CWD leak gate (≡1 for smac CWO):
    "gate_cos", "n_gate",
]


class OnPolicyPactSmacRunner(OnPolicyHARunner):
    """HAPPO on the PACT-augmented obs + a detailed CWO/PACT debug file."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyPactSmacRunner, self).__init__(args, algo_args, env_args)
        self.T = int(self.algo_args["train"]["episode_length"])
        self.nt = int(self.algo_args["train"]["n_rollout_threads"])
        # NaN, not -1: this is only a fallback for steps whose info lacks cwo_sigma, and
        # a sentinel value would be averaged into the severity column (it showed up as a
        # spurious -0.006 once per driver cycle).  _m() drops non-finite entries.
        self._severity = float(env_args.get("snd_severity", float("nan")))

        if not (int(env_args.get("snd_pact", 0)) or int(env_args.get("cwd_pact", 0))):
            print("[PACT][WARN] neither env_args.snd_pact (smac) nor cwd_pact (smacv2) "
                  "is set — obs is NOT augmented, so `--algo pact` == BLIND. Set "
                  "\"snd_pact\": 1 in the config's env_args.", flush=True)

        cfg = dict(env_args.get("pact_cfg", {}))
        self._gate_abort = bool(cfg.get("gate_abort", False))  # off by default (trivial for CWO)
        self._gate_min_cos = float(cfg.get("gate_min_corr", 0.999))
        self._gate_after = int(cfg.get("gate_after_steps", 400000))
        self._gate_min_n = int(cfg.get("gate_min_samples", 2000))
        self._gate_done = False

        self._roll = 0
        self._pact_step = 0
        self._acc = self._fresh()
        self._ep_lens = []
        self._ep_run = np.zeros(self.nt)
        # severity-0 control for the coordination statistic (see _check_zero_severity_control)
        self._zero_ctrl_n = 0
        self._zero_ctrl_sum = 0.0
        self._zero_ctrl_warned = False
        self._diseng = 0            # consecutive rollouts with the team not engaging
        self._diseng_warned = False

        self._dbg = self._dbg_w = None
        if getattr(self, "run_dir", None) is not None:
            p = os.path.join(str(self.run_dir), "pact_debug.csv")
            self._dbg = open(p, "w", newline="", encoding="utf-8")
            self._dbg_w = csv.writer(self._dbg)
            self._dbg_w.writerow(_COLS)
            print(f"[PACT] detailed debug trace -> {os.path.abspath(p)}", flush=True)

    @staticmethod
    def _fresh():
        keys = ["A", "sigma", "ell", "ell_max", "drop", "x2", "x2_spread",
                "x3", "x3try",
                "fire", "avail", "hold", "thru", "fire_hi", "fire_lo",
                "r", "cos", "gate_cos"]
        acc = {k: [] for k in keys}
        for ph in ("peak", "trough"):
            for k in ("ell", "drop", "fire", "hold", "thru", "fire_hi", "fire_lo"):
                acc[f"{k}_{ph}"] = []
        return acc

    def insert(self, data):
        infos, rewards, dones = data[4], data[2], data[3]
        a = self._acc
        for i in range(len(infos)):
            d = infos[i][0] if isinstance(infos[i], (list, tuple, np.ndarray)) else infos[i]
            if not isinstance(d, dict):
                continue
            A = float(d.get("pact_payload", d.get("snd_payload", np.nan)))
            a["sigma"].append(float(d.get("cwo_sigma", self._severity)))
            ell = float(d.get("cwo_ell_mean", d.get("pact_dload", np.nan)))
            fire_hi = float(d.get("cwo_fire_hi_load", np.nan))
            fire_lo = float(d.get("cwo_fire_lo_load", np.nan))
            fire = float(d.get("cwo_fire_frac", np.nan))
            drop = float(d.get("cwo_drop_frac", np.nan))
            hold = float(d.get("cwo_hold_frac", np.nan))
            thru = float(d.get("cwo_throughput", np.nan))
            a["A"].append(A)
            a["ell"].append(ell)
            a["ell_max"].append(float(d.get("cwo_ell_max", np.nan)))
            a["drop"].append(drop)
            a["x2"].append(float(d.get("cwo_x2_mean", d.get("pact_x2load", np.nan))))
            a["x2_spread"].append(float(d.get("cwo_x2_spread", np.nan)))
            a["x3"].append(float(d.get("cwo_x3_mean", np.nan)))
            a["x3try"].append(float(d.get("cwo_x3try_mean", np.nan)))
            a["fire"].append(fire)
            a["avail"].append(float(d.get("cwo_fire_avail", np.nan)))
            a["hold"].append(hold)
            a["thru"].append(thru)
            a["fire_hi"].append(fire_hi)
            a["fire_lo"].append(fire_lo)
            a["r"].append(float(np.asarray(rewards[i]).reshape(-1)[0]))
            # leak gate (smacv2/CWD): only counts where there is real waveform signal
            cos = float(d.get("pact_cos", np.nan))
            x2l = float(d.get("pact_x2load", np.nan))
            if np.isfinite(cos) and np.isfinite(A) and A > 0.3 and x2l > 1e-3:
                a["gate_cos"].append(cos)
            # phase split
            if np.isfinite(A):
                ph = "peak" if A > 0.7 else ("trough" if A < 0.3 else None)
                if ph is not None:
                    a[f"ell_{ph}"].append(ell)
                    a[f"drop_{ph}"].append(drop)
                    a[f"fire_{ph}"].append(fire)
                    a[f"hold_{ph}"].append(hold)
                    a[f"thru_{ph}"].append(thru)
                    a[f"fire_hi_{ph}"].append(fire_hi)
                    a[f"fire_lo_{ph}"].append(fire_lo)

        self._ep_run += 1
        for i in range(self.nt):
            if np.all(dones[i]):
                self._ep_lens.append(float(self._ep_run[i]))
                self._ep_run[i] = 0

        super(OnPolicyPactSmacRunner, self).insert(data)
        self._pact_step += 1
        if self._pact_step % self.T == 0:
            self._write_row()

    @staticmethod
    def _m(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size else float("nan")

    def _gap(self, lo, hi):
        return self._m(lo) - self._m(hi)

    def _write_row(self):
        self._roll += 1
        a, m = self._acc, self._m
        step = self._roll * self.T * self.nt
        gate_cos, n_gate = m(a["gate_cos"]), len(a["gate_cos"])
        sigma = m(a["sigma"])
        # THE coordination number: hold fire MORE at the driver peak than at the trough.
        hold_gap = m(a["hold_peak"]) - m(a["hold_trough"])

        row = [
            step, self._roll, sigma,
            m(a["A"]), m(a["ell"]), m(a["ell_max"]), m(a["drop"]), m(a["x2"]),
            m(a["x2_spread"]), m(a["x3"]), m(a["x3try"]),
            m(a["fire"]), m(a["avail"]), m(a["hold"]), m(a["thru"]),
            m(a["fire_hi"]), m(a["fire_lo"]),
            self._gap(a["fire_lo"], a["fire_hi"]),
            m(a["ell_peak"]), m(a["drop_peak"]), m(a["fire_peak"]),
            m(a["hold_peak"]), m(a["thru_peak"]),
            self._gap(a["fire_lo_peak"], a["fire_hi_peak"]),
            m(a["ell_trough"]), m(a["fire_trough"]),
            m(a["hold_trough"]), m(a["thru_trough"]),
            hold_gap,
            m(a["r"]),
            (self._m(self._ep_lens) if self._ep_lens else float("nan")),
            gate_cos, n_gate,
        ]
        if self._dbg_w is not None:
            self._dbg_w.writerow([round(v, 5) if isinstance(v, float) else v for v in row])
            self._dbg.flush()

        self._check_zero_severity_control(sigma, hold_gap)
        self._check_disengagement(step, m(a["avail"]),
                                  self._m(self._ep_lens) if self._ep_lens else float("nan"))

        if self._roll % 10 == 1:
            print(f"[PACT dbg] roll={self._roll} step={step} sigma={sigma:.2f} | "
                  f"ell(peak)={m(a['ell_peak']):.2f} drop={m(a['drop']):.2f} "
                  f"fire={m(a['fire']):.2f} hold={m(a['hold']):.2f} "
                  f"thru={m(a['thru']):.3f} | HOLD gap(peak-trough)={hold_gap:+.3f} "
                  f"(>0 = holding fire when the bus is hot; ~0 = phase-blind) | "
                  f"r={m(a['r']):.3f} "
                  f"ep_len={round(self._m(self._ep_lens), 0) if self._ep_lens else float('nan')}",
                  flush=True)

        self._check_gate(step, gate_cos, n_gate)
        self._acc = self._fresh()
        self._ep_lens = []

    def _check_disengagement(self, step, avail, ep_len):
        """Detect the team walking away from the fight -- a DIFFERENT failure from
        "PACT did not coordinate", and the one that actually ended the 17.7M-step run.

        SMAC's shaped reward pays for damage dealt.  When firing is mostly futile,
        "do not engage and survive to the time limit" becomes locally better than
        "engage and lose", and it is an absorbing state: once the team disengages the
        shared load goes to ~0, no shots are attempted, no jams are observed, and there
        is no gradient back.  It shows up as fire_avail (the fraction of live units with
        an enemy in range) collapsing while episode length pins at the limit -- measured
        going 0.89 -> 0.22 and 50 -> 141 as severity reached full, taking the win rate
        at the driver trough down with it (0.93 -> 0.01) even though the trough is
        almost unharmed.

        The real fix is env-side (a knee large enough that engaging at the coordinated
        load is genuinely good); this only makes the mode legible instead of looking
        like a method failure."""
        if not np.isfinite(avail):
            return
        self._diseng = self._diseng + 1 if avail < 0.70 else 0
        if self._diseng == 30 and not self._diseng_warned:
            self._diseng_warned = True
            print(f"[PACT][DISENGAGED] at step {step}: only {avail:.0%} of live units "
                  f"have a target in range (ep_len {ep_len:.0f}). The team has stopped "
                  f"fighting rather than learned to stagger -- this is the shaped-reward "
                  f"basin, not a coordination result. Raise the knee so the coordinated "
                  f"load is harm-free, and re-certify with pact.phase1.", flush=True)

    def _check_zero_severity_control(self, sigma, hold_gap):
        """`works-when-it-should` control (pipeline §II.4).  While the curriculum still
        holds severity at 0 the env IS stock SMAC, so any driver-phase-dependent
        behaviour is impossible and hold_gap must read ~0.  A persistent nonzero value
        means the statistic is measuring something other than coordination -- which is
        exactly how the old `stagger_gap` (+0.16 right through the severity-0 warmup on
        the 20M 3s5z run) was found to be an artifact of splitting agents by their own
        x2.  Warn once rather than abort: it is a measurement bug, not a training one."""
        if self._zero_ctrl_warned or sigma > 1e-6 or not np.isfinite(hold_gap):
            return
        self._zero_ctrl_n += 1
        self._zero_ctrl_sum += abs(hold_gap)
        if self._zero_ctrl_n < 50:
            return
        self._zero_ctrl_warned = True
        avg = self._zero_ctrl_sum / self._zero_ctrl_n
        if avg > 0.02:
            print(f"[PACT][CONTROL FAIL] |hold_gap| = {avg:.3f} averaged over the "
                  f"severity-0 warmup, where the env is stock SMAC and no phase-dependent "
                  f"behaviour is possible. The coordination statistic is biased -- fix the "
                  f"metric before reading any coordination claim from this run.", flush=True)
        else:
            print(f"[PACT][CONTROL PASS] |hold_gap| = {avg:.3f} at severity 0 "
                  f"(unbiased: reads ~0 when there is no NS).", flush=True)

    def _check_gate(self, step, cos, n):
        """smacv2/CWD leak gate (trivial ≡1 for smac CWO; off by default)."""
        if self._gate_done or step < self._gate_after or not self._gate_abort:
            return
        if not np.isfinite(cos) or n < self._gate_min_n:
            return
        self._gate_done = True
        if cos < self._gate_min_cos:
            raise AssertionError(
                f"[PACT][GATE FAIL] cos(x2,d)={cos:.4f} < {self._gate_min_cos} at step "
                f"{step}: the waveform is miswired (index/reset/timing). Fix it.")
        print(f"[PACT][GATE PASS] cos(x2,d)={cos:.5f} at step {step}.", flush=True)

    def close(self):
        if self._dbg is not None:
            try:
                self._dbg.close()
            except Exception:
                pass
        super(OnPolicyPactSmacRunner, self).close()
