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
    # --- IS THE ARM LEARNING THE TASK AT ALL?  READ THESE FIRST. -------------------
    # win_rate/timeout_frac are the TRAINING numbers, so during a severity-0 warmup
    # they are the STATIONARY (stock-SMAC) numbers.  progress.txt cannot tell you
    # this: eval envs always run at the FULL severity (make_eval_env sets
    # snd_eval=1 -> _curr_severity returns snd_severity), so the eval win rate during
    # the warmup is a zero-shot-under-the-NS number and is expected to read ~0 --
    # guide II.2 is explicit that this is NOT the baseline.  Without these columns
    # "did the warmup succeed?" is unanswerable from the logs, which is exactly how
    # three 20M runs were burned.
    "win_rate", "timeout_frac", "n_ep", "ep_ret",
    # fighting or farming?  in the timeout basin BOTH stay low while ep_len pins.
    "ally_dead", "enemy_dead",
    # regen_frac = share of r_step_mean that is SHIELD-REGENERATION PAY (stock SMAC's
    # abs() in reward_battle).  A large value means the basin is a reward artifact:
    # the team is paid for standing off and it is paid MORE the longer it stalls.
    "regen_frac",
    # *** THE ESCAPE TEST (guide I.3 -- "the single highest-value diagnostic in the
    # whole programme"): E||ell|| and the load x2 at a MATCHED driver level.  Falls
    # over training => the team found a way to switch the NS off; flat => the load is
    # uncancellable.  NaN while A never enters the band in this rollout.
    "ell_matchedA", "x2_matchedA", "n_matchedA",
    # smacv2/CWD leak gate (≡1 for smac CWO):
    "gate_cos", "n_gate",
    # --- PACT-1 only (nan otherwise) ---
    # p1_ellhat     the deflection each unit PREDICTS from its own beta_hat
    # p1_conf       estimator self-confidence (the trust prior)
    # p1_beta_err   ||beta_hat - beta_true||: is the split being tracked?
    # p1_raw_shift  |s| the channel applied  } their gap is THE result: raw is what
    # p1_net_shift  |s - s_hat| after re-aim } a blind unit eats, net is what is left
    # p1_obs_frac   fraction of units that got a usable reading this step
    "p1_ellhat", "p1_conf", "p1_beta_err", "p1_raw_shift", "p1_net_shift",
    "p1_obs_frac",
    # p1_cancel  fraction of the deflection actually cancelled -- THE headline,
    #            the direct analogue of Ant's pact1_cancel_frac.
    "p1_cancel",
    # --- PACT-1 forensics: which of the four possible faults is it? ---------------
    # p1_frozen     1.0 while the RLS is frozen because the CURRICULUM severity is 0.
    #               Must be 1.0 for the whole warmup and 0.0 afterwards.  If it is 0
    #               during the warmup the estimator is being poisoned with y==0 data
    #               (see pact1_warmup_freeze in StarCraft2_Env._snd_resolve_knobs).
    # p1_n_upd      cumulative RLS updates.  Flat during the warmup (frozen), then it
    #               must CLIMB -- flat afterwards means the sensor never fires and no
    #               estimate is possible however good the maths is.
    # p1_aug_var    *** THE CONTROL FOR THE OBS-AUGMENTATION HYPOTHESIS. ***  The
    #               variance of the whole appended block.  With the freeze on it must
    #               be EXACTLY 0.0 through the warmup: a constant append is
    #               input-equivalent to no append, so any warmup difference against
    #               blind cannot be the augmentation and must be seed/basin.  If it
    #               is nonzero, the arms were never comparable and that IS the bug.
    # p1_aug_absmean  magnitude of the append, to size it against the obs it joins.
    # p1_beta_hat0/1 vs p1_beta_true0/1  per-channel tracking, not just the norm --
    #               a beta_err that looks fine can hide the two channels swapping.
    # p1_psi_cond   cond(E[psi psi^T]) (guide III.6).  Large => only the projection
    #               beta*.psi is identifiable, not the same/cross SPLIT.  Report it
    #               before claiming theta was identified.
    # p1_psi_norm   if this is ~0 the regressor is dead and nothing is estimable.
    # p1_conf_min/max  spread across units; the compensator arms per unit at
    #               pact1_conf_thresh, so the MIN is what gates the slowest unit.
    "p1_frozen", "p1_n_upd", "p1_aug_var", "p1_aug_absmean",
    "p1_beta_hat0", "p1_beta_hat1", "p1_beta_true0", "p1_beta_true1",
    "p1_psi_norm", "p1_psi_cond", "p1_psi_lmin", "p1_conf_min", "p1_conf_max",
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

        if not (int(env_args.get("snd_pact", 0)) or int(env_args.get("cwd_pact", 0))
                or int(env_args.get("snd_pact1", 0))):
            print("[PACT][WARN] none of env_args.snd_pact (smac), snd_pact1 (smac "
                  "PACT-1) or cwd_pact (smacv2) is set — obs is NOT augmented and no "
                  "compensation runs, so this arm == BLIND. Set \"snd_pact1\": 1 in "
                  "the config's env_args.", flush=True)

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
        # true per-episode return (r_step_mean * ep_len_mean is only an approximation
        # and it overestimates whenever episode lengths and rewards correlate, which
        # is exactly what happens in the timeout basin)
        self._ep_rets = []
        self._ep_ret_run = np.zeros(self.nt)
        # cumulative SMAC battle counters per thread, for the TRAIN win / timeout rate
        self._bw = np.zeros(self.nt)      # battles_won, latest seen
        self._bg = np.zeros(self.nt)      # battles_game
        self._bd = np.zeros(self.nt)      # battles_draw (timeouts)
        self._bw0 = np.zeros(self.nt)     # ... at the end of the previous rollout
        self._bg0 = np.zeros(self.nt)
        self._bd0 = np.zeros(self.nt)
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
                "r", "cos", "gate_cos",
                # fighting-or-farming + the escape test
                "ally_dead", "enemy_dead", "regen", "ell_mA", "x2_mA",
                # PACT-1 (absent -> nan -> dropped by _m)
                "p1_ellhat", "p1_conf", "p1_beta_err", "p1_raw", "p1_net", "p1_obs",
                "p1_cancel",
                "p1_frozen", "p1_n_upd", "p1_aug_var", "p1_aug_absmean",
                "p1_bh0", "p1_bh1", "p1_bt0", "p1_bt1",
                "p1_psi_norm", "p1_psi_cond", "p1_psi_lmin",
                "p1_conf_min", "p1_conf_max"]
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
            r_i = float(np.asarray(rewards[i]).reshape(-1)[0])
            a["r"].append(r_i)
            self._ep_ret_run[i] += r_i
            # fighting or farming
            a["ally_dead"].append(float(d.get("cwo_ally_dead", np.nan)))
            a["enemy_dead"].append(float(d.get("cwo_enemy_dead", np.nan)))
            a["regen"].append(float(d.get("cwo_regen_pay", np.nan)))
            # the escape test: E||ell|| and the load at a MATCHED driver level.
            # Averaging over all A confounds "the team walked away from the load"
            # with "the driver happened to be low this rollout" -- guide I.3.
            if np.isfinite(A) and 0.69 <= A <= 0.71:
                a["ell_mA"].append(ell)
                a["x2_mA"].append(float(d.get("cwo_x2_mean", np.nan)))
            # SMAC battle counters (cumulative per env instance)
            if "battles_game" in d:
                self._bw[i] = float(d.get("battles_won", 0.0))
                self._bg[i] = float(d.get("battles_game", 0.0))
                self._bd[i] = float(d.get("battles_draw", 0.0))
            # PACT-1 estimator telemetry
            a["p1_ellhat"].append(float(d.get("p1_ellhat", np.nan)))
            a["p1_conf"].append(float(d.get("p1_conf", np.nan)))
            a["p1_beta_err"].append(float(d.get("p1_beta_err", np.nan)))
            a["p1_raw"].append(float(d.get("p1_raw_shift", np.nan)))
            a["p1_net"].append(float(d.get("p1_net_shift", np.nan)))
            a["p1_obs"].append(float(d.get("p1_obs_frac", np.nan)))
            a["p1_cancel"].append(float(d.get("p1_cancel", np.nan)))
            a["p1_frozen"].append(float(d.get("p1_frozen", np.nan)))
            a["p1_n_upd"].append(float(d.get("p1_n_upd", np.nan)))
            a["p1_aug_var"].append(float(d.get("p1_aug_var", np.nan)))
            a["p1_aug_absmean"].append(float(d.get("p1_aug_absmean", np.nan)))
            a["p1_bh0"].append(float(d.get("p1_beta_hat0", np.nan)))
            a["p1_bh1"].append(float(d.get("p1_beta_hat1", np.nan)))
            a["p1_bt0"].append(float(d.get("p1_beta_true0", np.nan)))
            a["p1_bt1"].append(float(d.get("p1_beta_true1", np.nan)))
            a["p1_psi_norm"].append(float(d.get("p1_psi_norm", np.nan)))
            a["p1_psi_cond"].append(float(d.get("p1_psi_cond", np.nan)))
            a["p1_psi_lmin"].append(float(d.get("p1_psi_lmin", np.nan)))
            a["p1_conf_min"].append(float(d.get("p1_conf_min", np.nan)))
            a["p1_conf_max"].append(float(d.get("p1_conf_max", np.nan)))
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
                self._ep_rets.append(float(self._ep_ret_run[i]))
                self._ep_run[i] = 0
                self._ep_ret_run[i] = 0.0

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

        # TRAIN win / timeout rate over the games that FINISHED in this rollout.
        # During a severity-0 warmup these are the STATIONARY numbers, i.e. the only
        # answer in the whole log to "is this arm learning stock SMAC at all".
        d_bg = float(np.sum(self._bg - self._bg0))
        d_bw = float(np.sum(self._bw - self._bw0))
        d_bd = float(np.sum(self._bd - self._bd0))
        win_rate = (d_bw / d_bg) if d_bg > 0 else float("nan")
        timeout_frac = (d_bd / d_bg) if d_bg > 0 else float("nan")
        self._bw0, self._bg0, self._bd0 = self._bw.copy(), self._bg.copy(), self._bd.copy()
        ep_ret = self._m(self._ep_rets) if self._ep_rets else float("nan")
        r_step = m(a["r"])
        regen = m(a["regen"])
        regen_frac = (regen / r_step) if (np.isfinite(regen) and abs(r_step) > 1e-9) else float("nan")

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
            r_step,
            (self._m(self._ep_lens) if self._ep_lens else float("nan")),
            win_rate, timeout_frac, d_bg, ep_ret,
            m(a["ally_dead"]), m(a["enemy_dead"]), regen_frac,
            m(a["ell_mA"]), m(a["x2_mA"]), len(a["ell_mA"]),
            gate_cos, n_gate,
            m(a["p1_ellhat"]), m(a["p1_conf"]), m(a["p1_beta_err"]),
            m(a["p1_raw"]), m(a["p1_net"]), m(a["p1_obs"]), m(a["p1_cancel"]),
            m(a["p1_frozen"]), m(a["p1_n_upd"]),
            m(a["p1_aug_var"]), m(a["p1_aug_absmean"]),
            m(a["p1_bh0"]), m(a["p1_bh1"]), m(a["p1_bt0"]), m(a["p1_bt1"]),
            m(a["p1_psi_norm"]), m(a["p1_psi_cond"]), m(a["p1_psi_lmin"]),
            m(a["p1_conf_min"]), m(a["p1_conf_max"]),
        ]
        if self._dbg_w is not None:
            self._dbg_w.writerow([round(v, 5) if isinstance(v, float) else v for v in row])
            self._dbg.flush()

        self._check_zero_severity_control(sigma, hold_gap)
        self._check_disengagement(step, m(a["avail"]),
                                  self._m(self._ep_lens) if self._ep_lens else float("nan"),
                                  sigma, win_rate, timeout_frac)
        self._check_aug_inert(step, sigma, m(a["p1_aug_var"]), m(a["p1_frozen"]))

        if self._roll % 10 == 1:
            print(f"[PACT dbg] roll={self._roll} step={step} sigma={sigma:.2f} | "
                  f"WIN={win_rate:.3f} timeout={timeout_frac:.2f} ep_ret={ep_ret:.2f} "
                  f"(n={d_bg:.0f}) | "
                  f"ell(peak)={m(a['ell_peak']):.2f} drop={m(a['drop']):.2f} "
                  f"fire={m(a['fire']):.2f} avail={m(a['avail']):.2f} "
                  f"hold={m(a['hold']):.2f} thru={m(a['thru']):.3f} | "
                  f"HOLD gap(peak-trough)={hold_gap:+.3f} "
                  f"(>0 = holding fire when the bus is hot; ~0 = phase-blind) | "
                  f"e_dead={m(a['enemy_dead']):.2f} regen_frac={regen_frac:.2f} | "
                  f"r={r_step:.3f} "
                  f"ep_len={round(self._m(self._ep_lens), 0) if self._ep_lens else float('nan')}",
                  flush=True)

        self._check_gate(step, gate_cos, n_gate)
        self._acc = self._fresh()
        self._ep_lens = []
        self._ep_rets = []

    def _check_aug_inert(self, step, sigma, aug_var, frozen):
        """*** THE CONTROL THAT SETTLES THE OBS-AUGMENTATION QUESTION. ***

        At severity 0 the env is byte-identical stock SMAC, so the ONLY thing that can
        distinguish this arm from blind HAPPO is the appended obs/state block.  With
        the warmup freeze on (StarCraft2_Env.pact1_warmup_freeze) that block is a
        CONSTANT vector, and a constant input feature is equivalent to no feature --
        so if the arms still diverge during the warmup, the augmentation is innocent
        and the difference is seed/basin, not the method.

        Warn once if the block is NOT constant while sigma is 0: that means the
        comparison against blind was never controlled and the augmentation hypothesis
        is back on the table.  This is a measurement statement, so it warns rather
        than aborting."""
        if getattr(self, "_aug_warned", False) or sigma > 1e-6:
            return
        if not np.isfinite(aug_var):
            return
        if aug_var > 1e-12:
            self._aug_warned = True
            print(f"[PACT][AUG LIVE] at step {step}: severity is 0 (the env is stock "
                  f"SMAC) but the appended obs/state block still VARIES "
                  f"(var={aug_var:.3e}, frozen={frozen:.2f}). This arm is therefore "
                  f"NOT input-equivalent to blind during the warmup, so any warmup "
                  f"difference against blind is confounded by the augmentation. Set "
                  f"env_args.pact1_warmup_freeze=1 (default) to make the block "
                  f"constant.", flush=True)
        elif not getattr(self, "_aug_ok_printed", False) and self._roll >= 5:
            self._aug_ok_printed = True
            print(f"[PACT][AUG CONSTANT] at step {step}: the appended block is "
                  f"constant while severity is 0 (var={aug_var:.1e}), so this arm is "
                  f"input-equivalent to blind for the whole warmup. Any warmup gap "
                  f"against blind is seed/basin, NOT the augmentation.", flush=True)

    def _check_disengagement(self, step, avail, ep_len, sigma=float("nan"),
                             win_rate=float("nan"), timeout_frac=float("nan")):
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
            if np.isfinite(sigma) and sigma <= 1e-6:
                # *** THE IMPORTANT CASE, AND THE ONE THAT KILLED THIS RUN. ***
                print(f"[PACT][DISENGAGED @ SEVERITY 0] at step {step}: only "
                      f"{avail:.0%} of live units have a target in range, ep_len "
                      f"{ep_len:.0f}, train win {win_rate:.3f}, timeouts "
                      f"{timeout_frac:.0%}.\n"
                      f"    THE NON-STATIONARITY IS OFF -- this env is byte-identical "
                      f"stock SMAC, so this is NOT a method failure and NOT an NS "
                      f"failure.\n"
                      f"    It is the stock-SMAC 'farm damage, never finish' timeout "
                      f"basin (reward_battle's abs() pays for enemy shield "
                      f"regeneration,\n"
                      f"    so stalling accrues reward and accrues MORE of it the "
                      f"longer the episode runs -- read the regen_frac column).\n"
                      f"    It has now been recorded on BOTH arms at severity 0, so "
                      f"the warmup is measuring basin luck, not the method.\n"
                      f"    FIX IT AT THE PROTOCOL LEVEL: train one B0 per obs shape "
                      f"at severity 0, check BOTH clear a comparable stationary win "
                      f"rate,\n"
                      f"    then run every arm from those checkpoints with "
                      f"SMAC_SND_WARMUP=0 (guide II.6).", flush=True)
            else:
                print(f"[PACT][DISENGAGED] at step {step}: only {avail:.0%} of live "
                      f"units have a target in range (ep_len {ep_len:.0f}, train win "
                      f"{win_rate:.3f}). The team has stopped fighting rather than "
                      f"learned to stagger -- this is the shaped-reward basin, not a "
                      f"coordination result. Check the escape-test columns "
                      f"(ell_matchedA / x2_matchedA): if they FALL over training the "
                      f"team is switching the NS off. phi=alive should prevent that.",
                  flush=True)

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
