"""Tier 0 — scripted probes  [campaign spec Part 4]. Eval-only; the decisive tier.

Loads a trained HASAC checkpoint, rolls it out through a probe wrapper, records.
**No gradient anywhere.** Everything here runs in hours, and it is what lets the
next method's core mechanism be demonstrated on the real environment *before any
learning code exists*.

Stages
------
``e1``     collapse profile of a competent naive policy: identity probe,
           FREEZE_A in {0, .25, .5, .75, 1}. Reports the **deficit decomposition**
           R = rbar * L  =>  dR = drbar*L0 + rbar_A*dL, i.e. achievement-mediated
           vs termination-mediated loss (feeds WP-3 and the E6 attribution).
``d0``     (= E1b) drifting zero-shot, clock-stratified: the no-adaptation
           reference curve. Any adaptive method must beat it, and its cycle
           average is the honest do-nothing floor. If it already exceeds the
           blind-trained ~3500, NS training actively destroyed competence.
``e2``     privileged cancellation — **the existence experiment**. cancel(beta,
           exact) over a beta grid at A in {0.5, 1}. Gate **V1**.
``e2b``    severity sweep at beta*: sigma in {0.6, .9, 1.0, 1.1, 1.3} (+ an
           optional DCAP leg). The empirical phase-boundary figure / the feasible
           frontier sigma* for the redesign track.
``e3``     the degraded-information frontier: delay / ema / dc / noise / sign_leg.
           Produces (k*, h*, r*) — the spec any future estimator must beat, and
           the retro-diagnosis of every slow-chart method (ECL included).
``e3dob``  the decentralized-feasibility certificate: the E5-fitted filter run in
           closed loop on agent-local features only.
``e4``     the information-free escape: project_sumzero at A in {0, 1}.
``e6``     harm-channel attribution: identity at A=1 with MASK in {hip, ankle}.

Standing rule (spec §4.0), enforced here and not merely documented: **every probe
is first run at FREEZE_A=0 and must reproduce the stationary return within CI.**
A probe that degrades the stock env is confounded and its results are void — fix
the probe, do not reinterpret (abort rule 5). ``project_sumzero`` is the one
exception: its A=0 run *is* the measurement (the cost of the constraint itself).

    python scripts/diag_tier0.py --ckpt <F0>/models --stage e2 --out diag_out/e2
    python scripts/diag_tier0.py --ckpt <F0>/models --probe 'cancel:beta=0.75' \
        --A 1.0 --episodes 40 --out diag_out/custom
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harl.envs.mamujoco.diag.report_io import (  # noqa: E402
    DebugReport, bootstrap_ci, compare, fmt_ci, write_csv)
from harl.envs.mamujoco.diag import probes as P  # noqa: E402
from harl.envs.mamujoco.pcr_diag import PcrDiagMujocoMulti, build_row  # noqa: E402

_P_PERIOD = 40000
_CELL_COLS = ["stage", "cell", "probe", "A", "severity", "mask", "dcap",
              "episodes", "ret_mean", "ret_lo", "ret_hi", "len_mean",
              "fall_rate", "median_time_to_fall", "fwd_vel", "sat_frac",
              "clip_frac_cmd", "r_ctrl", "sumzero_resid",
              "d_hip_common", "d_hip_diff", "d_ank_common", "d_ank_diff",
              "tau_hip_common", "tau_hip_diff", "tau_ank_common", "tau_ank_diff",
              "term_fall_low", "term_fall_high", "term_nonfinite", "term_timeout"]


# ==========================================================================
#  env + actors
# ==========================================================================
def load_run_config(ckpt_dir):
    """The run's config.json (models/ lives inside the run dir)."""
    for cand in (os.path.join(ckpt_dir, "config.json"),
                 os.path.join(os.path.dirname(ckpt_dir.rstrip("/\\")), "config.json")):
        if os.path.exists(cand):
            with open(cand, encoding="utf-8") as f:
                return json.load(f), cand
    raise FileNotFoundError(
        f"no config.json beside {ckpt_dir!r} — Tier 0 must build the actor with the "
        f"EXACT model args it was trained with, and will not guess them.")


def make_env(cfg, out_dir, cell_tag, dump_traj=False):
    """``MujocoMulti`` exactly as HARL eval builds it (same agent_conf, agent_obsk,
    obs pipeline) — via the flight recorder, so every Tier-0 cell writes the §3.1
    CSVs for free."""
    ea = dict(cfg["env_args"])
    for k in ("ecl", "echor", "diag", "ecl_cfg", "diag_cfg"):
        ea.pop(k, None)                      # Tier 0 is probe-only: no method shims
    ea["pcr_diag"] = True
    ea["pcr_diag_cfg"] = {"dir": os.path.join(out_dir, "recorder", cell_tag),
                          "interval": 1, "dump_traj": bool(dump_traj),
                          "dump_every_k_episodes": 1, "dump_max_mb": 2048}
    return PcrDiagMujocoMulti(env_args=ea)


def load_actors(cfg, ckpt_dir, env, device="cpu"):
    import torch
    from harl.algorithms.actors import ALGO_REGISTRY

    algo = cfg["main_args"]["algo"]
    algo = "hasac" if algo in ("hasac_diag", "ecl", "echor_hasac") else algo
    args = {**cfg["algo_args"]["model"], **cfg["algo_args"]["algo"]}
    actors = []
    for i in range(env.n_agents):
        a = ALGO_REGISTRY[algo](args, env.observation_space[i], env.action_space[i],
                                device=torch.device(device))
        a.restore(ckpt_dir, i)
        a.turn_off_grad()
        actors.append(a)
    # obs-schema guard: a blind checkpoint in an ORACLE env (or vice versa) has a
    # silently wrong input dim. This is the E-5 failure one level down.
    want = actors[0].actor.net.mlp[0].in_features
    got = env.observation_space[0].shape[0]
    if want != got:
        raise ValueError(
            f"obs-schema mismatch: the checkpoint's actor expects {want} inputs but "
            f"this env produces {got}. Almost always ANT_PCR_ORACLE / "
            f"ANT_PCR_CORACLE set (or not set) differently from training — check "
            f"the [DIAG ENV] banner above against the checkpoint's run.")
    return actors


# ==========================================================================
#  one cell
# ==========================================================================
def run_cell(cfg, ckpt_dir, out_dir, stage, cell, probe_spec, A=None, severity=None,
             mask=None, dcap=None, episodes=40, dephase=False, dump_traj=False,
             device="cpu", seed=0):
    """Roll ``episodes`` deterministic-actor episodes through ``probe_spec``."""
    import torch
    from harl.envs.mamujoco.diag import knobs as K

    # Set the cell's knobs on the DEPLOYED ant module (see diag/knobs.py — setting
    # them on the repo's ant_diag copy is a different module object and would
    # silently do nothing). Knobs are snapshotted per env instance, so one process
    # can sweep the whole grid.
    knobs = K.apply(freeze_a=A, severity=severity,
                    mask=mask if mask is not None else "both", dcap=dcap)

    tag = f"{stage}_{cell}".replace("/", "_").replace(":", "-").replace(",", "_")
    env = make_env(cfg, out_dir, tag, dump_traj=dump_traj)
    env.seed(seed)
    probe = P.make_probe(probe_spec)
    P.install_probe(env, probe)
    actors = load_actors(cfg, ckpt_dir, env, device)

    eps = []
    for ep in range(episodes):
        if dephase:
            # C4: one episode per phase stratum => the round is a true cycle-average
            inner = getattr(env.env, "unwrapped", env.env)
            inner._clock = (ep * _P_PERIOD) // max(1, episodes)
        obs, _, _ = env.reset()
        rows = []
        done = False
        while not done:
            with torch.no_grad():
                acts = [actors[i].get_actions(np.asarray(obs[i])[None],
                                              stochastic=False).cpu().numpy()[0]
                        for i in range(env.n_agents)]
            obs, _, rewards, dones, infos, _ = env.step(np.asarray(acts))
            info0 = infos[0]
            done = bool(np.all(dones))
            cmd = np.asarray(info0.get("probe_commanded_action",
                                       np.concatenate(acts)), dtype=np.float64)
            tau = np.clip(cmd, -1.0, 1.0)
            d_app = np.asarray(info0.get("pcr_d_applied", np.zeros(8)), float)
            try:
                sv = env.env.state_vector()
                height, finite = float(sv[2]), bool(np.isfinite(sv).all())
            except Exception:
                height, finite = float("nan"), True
            bad = bool(info0.get("bad_transition", False))
            cause = ("" if not done else
                     "timeout" if bad else
                     "nonfinite" if not finite else
                     "fall_low" if height < 0.2 else
                     "fall_high" if height > 1.0 else "other")
            rows.append(build_row(0, len(rows) + 1, done, done and not bad, cause,
                                  tau, d_app, np.clip(tau + d_app, -1, 1), height,
                                  info0, float(np.mean(np.asarray(rewards)))))
        eps.append(rows)
    env.close()

    return _summarize(stage, cell, probe.name, knobs, eps, episodes)


def _summarize(stage, cell, probe_name, knobs, eps, episodes):
    rets = np.array([sum(r["reward"] for r in e) for e in eps])
    lens = np.array([len(e) for e in eps])
    causes = [e[-1]["term_cause"] for e in eps]
    falls = np.array([c in ("fall_low", "fall_high", "nonfinite") for c in causes])
    ttf = lens[falls]
    mean_of = lambda k: float(np.mean([np.nanmean([r[k] for r in e]) for e in eps]))
    p, lo, hi = bootstrap_ci(rets)
    row = {
        "stage": stage, "cell": cell, "probe": probe_name,
        "A": knobs["FREEZE_A"], "severity": knobs["SEVERITY"], "mask": knobs["MASK"],
        "dcap": knobs["DCAP"], "episodes": episodes,
        "ret_mean": round(p, 2), "ret_lo": round(lo, 2), "ret_hi": round(hi, 2),
        "len_mean": round(float(lens.mean()), 1),
        "fall_rate": round(float(falls.mean()), 3),
        "median_time_to_fall": (round(float(np.median(ttf)), 1) if ttf.size
                                else float("nan")),
        "term_fall_low": causes.count("fall_low"),
        "term_fall_high": causes.count("fall_high"),
        "term_nonfinite": causes.count("nonfinite"),
        "term_timeout": causes.count("timeout"),
    }
    for k in ("fwd_vel", "sat_frac", "clip_frac_cmd", "r_ctrl", "sumzero_resid",
              "d_hip_common", "d_hip_diff", "d_ank_common", "d_ank_diff",
              "tau_hip_common", "tau_hip_diff", "tau_ank_common", "tau_ank_diff"):
        row[k] = round(mean_of(k), 5)
    row["_returns"] = rets
    row["_lens"] = lens
    return row


def _row_list(r):
    return [r[c] for c in _CELL_COLS]


def deficit_decomposition(base, arm):
    """R = rbar * L  =>  dR = drbar*L0 + rbar_A*dL  (exactly; spec §4.1).

    Splits the return deficit into an **achievement-mediated** part (the ant moves
    less / fights the load: drbar*L0) and a **termination-mediated** part (it tips
    over and loses the survival stream: rbar_A*dL). WP-3 (graceful harm) wants the
    termination share <= 50% at peak severity — an absorbing-state collapse
    starves the learning signal and makes WP-2 fail for reasons that have nothing
    to do with non-stationarity.
    """
    L0, LA = float(base["len_mean"]), float(arm["len_mean"])
    R0, RA = float(base["ret_mean"]), float(arm["ret_mean"])
    r0, rA = R0 / max(L0, 1e-9), RA / max(LA, 1e-9)
    dR = RA - R0
    ach = (rA - r0) * L0
    term = rA * (LA - L0)
    share = abs(term) / max(abs(ach) + abs(term), 1e-9)
    return {"dR": dR, "achievement": ach, "termination": term,
            "termination_share": share, "check": ach + term - dR}


# ==========================================================================
#  the A=0 control (spec §4.0 standing rule)
# ==========================================================================
def a0_control(cfg, ckpt, out, rep, probe_spec, episodes, b0_returns, device,
               exempt=False):
    """Run this probe at FREEZE_A=0 and require it to reproduce the stationary
    return within CI. Returns (ok, row)."""
    r = run_cell(cfg, ckpt, out, "control", f"A0_{probe_spec}", probe_spec, A=0.0,
                 episodes=episodes, device=device)
    if exempt:
        rep.line(f"  A=0 control [{probe_spec}]: {fmt_ci(r['_returns'])} "
                 f"— EXEMPT from the reproduce-stationary rule: for "
                 f"project_sumzero the A=0 run IS the measurement (the cost of the "
                 f"constraint itself, spec §4.0/§4.4).")
        return True, r
    c = compare(b0_returns, r["_returns"], "identity@A=0 (B0)", f"{probe_spec}@A=0")
    ok = not c["separated"] or c["verdict"] == "<"
    rep.line(f"  A=0 control [{probe_spec}]: {c['summary']}")
    if not ok:
        rep.line(f"  !! PROBE CONFOUNDED: it DEGRADES the stock env at A=0. Per "
                 f"abort rule 5 this invalidates ONLY this probe — fix the probe, "
                 f"do not reinterpret its cells.")
    return ok, r


# ==========================================================================
#  stages
# ==========================================================================
def stage_e1(cfg, ckpt, out, rep, args, cells):
    rep.h2("E1 — collapse profile of a competent naive policy")
    rep.line("Stationary walker, blind, identity probe. The open question is not "
             "*whether* it degrades but **falls vs slowness** at each A.")
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = {}
    for A in grid:
        r = run_cell(cfg, ckpt, out, "e1", f"A{A}", "identity", A=A,
                     episodes=args.episodes, dump_traj=True, device=args.device)
        rows[A] = r
        cells.append(r)
    rep.table(["A", "c = A*sigma", "return [95% CI]", "len", "fall rate",
               "median TTF", "fwd_vel", "sat_frac", "clip_cmd"],
              [[A, round(A * rows[A]["severity"], 3), fmt_ci(rows[A]["_returns"]),
                rows[A]["len_mean"], rows[A]["fall_rate"],
                rows[A]["median_time_to_fall"], rows[A]["fwd_vel"],
                rows[A]["sat_frac"], rows[A]["clip_frac_cmd"]] for A in grid])

    rep.h3("deficit decomposition (spec §4.1) — achievement vs termination")
    dd = []
    for A in grid[1:]:
        d = deficit_decomposition(rows[0.0], rows[A])
        dd.append([A, round(d["dR"], 1), round(d["achievement"], 1),
                   round(d["termination"], 1), f"{d['termination_share']:.0%}",
                   f"{d['check']:.2e}"])
    rep.table(["A", "dR", "achievement (drbar*L0)", "termination (rbar_A*dL)",
               "termination share", "identity check"], dd)
    peak_share = deficit_decomposition(rows[0.0], rows[1.0])["termination_share"]
    rep.verdict("WP-3 graceful harm (termination share <= 50% at peak)",
                peak_share <= 0.5,
                f"termination-mediated share at A=1 is {peak_share:.0%}")

    rep.h3("fall-cause table")
    rep.table(["A", "fall_low", "fall_high", "nonfinite", "timeout"],
              [[A, rows[A]["term_fall_low"], rows[A]["term_fall_high"],
                rows[A]["term_nonfinite"], rows[A]["term_timeout"]] for A in grid])

    b0 = float(rows[0.0]["ret_mean"])
    if float(rows[1.0]["ret_mean"]) >= 6000:
        rep.note("**SURPRISE (spec §4.1).** The naive walker survives A=1 with "
                 "return >= 6000. The entire 'the peak game is hard' premise dies: "
                 "the problem is purely training-induced. Jump the campaign's focus "
                 "to H-C2 (cross-context interference) / H-C5 (SAC miscalibration) "
                 "and read V4 (robust single policy) as the leading axis.")
    rep.kv("B0 (stationary return of this walker)", f"{b0:.1f}")
    return rows


def stage_d0(cfg, ckpt, out, rep, args, cells):
    rep.h2("D0 / E1b — drifting zero-shot: the no-adaptation floor")
    r = run_cell(cfg, ckpt, out, "d0", "drift", "identity", A=None,
                 episodes=max(args.episodes, 40), dephase=True, dump_traj=True,
                 device=args.device)
    cells.append(r)
    rep.line(f"  clock-stratified over one full payload cycle "
             f"({max(args.episodes, 40)} episodes, one per phase stratum)")
    rep.kv("cycle-average return", fmt_ci(r["_returns"]))
    rep.kv("fall rate", r["fall_rate"])
    rep.note("This is the honest **do-nothing floor**: any adaptive method must "
             "beat it. Compare with blind-TRAINED ~3500 (E-2). If this zero-shot "
             "walker already exceeds it, NS training actively DESTROYED competence "
             "— strong H-C2 evidence, and a damning point about every baseline.")
    return r


def stage_e2(cfg, ckpt, out, rep, args, cells, b0_returns, b0):
    rep.h2("E2 — privileged cancellation: THE EXISTENCE EXPERIMENT")
    rep.line("Does a stabilizing feed-forward compensation exist *in practice* at "
             "sigma=0.9 — i.e. was Route A actually sufficient?")
    rep.h3("why a beta grid and not just beta=1 (the terms this measures)")
    rep.line("Under a = pi(o) - beta*d the commanded torque re-enters the other "
             "legs' liability, closing a loop. Mode-wise (M = 11' - I on 4 legs; "
             "eigenvalues +3 common, -1 difference; leak rho=0.8):")
    rep.line("```")
    rep.line("  pole_common = rho - 3(1-rho)c*beta = 0.8 - 0.54*beta   (c=0.9)  "
             "fast, stable for all beta")
    rep.line("  pole_diff   = rho + (1-rho)c*beta  = 0.8 + 0.18*beta  <= 0.98   "
             "stable but SLOW")
    rep.line("  DC gain, tau_des -> d (difference modes)")
    rep.line("              = (1-rho)c*beta / (1 - pole_diff) = 9*beta  at beta=1")
    rep.line("```")
    rep.line("So the slow/DC component of the gait's difference-mode torque is "
             "amplified up to x9 into d, and a = pi - d then rails the actuator "
             "exactly where cancellation is needed most; gait-frequency content "
             "sees only ~x0.2. **Whether the real gait's difference-DC content is "
             "small enough is not derivable — it is measured here.**")

    ok, _ = a0_control(cfg, ckpt, out, rep, "cancel:beta=1.0", args.episodes,
                       b0_returns, args.device)
    best = None
    rows = []
    for A in (0.5, 1.0):
        for beta in (0.25, 0.5, 0.75, 1.0):
            spec = f"cancel:beta={beta}"
            r = run_cell(cfg, ckpt, out, "e2", f"A{A}_b{beta}", spec, A=A,
                         episodes=args.episodes, device=args.device)
            cells.append(r)
            rows.append([A, beta, fmt_ci(r["_returns"]), r["fall_rate"],
                         r["sat_frac"], r["clip_frac_cmd"], r["r_ctrl"],
                         r["tau_hip_common"], r["tau_hip_diff"],
                         r["d_hip_common"], r["d_hip_diff"]])
            if A == 1.0 and (best is None or r["ret_mean"] > best["ret_mean"]):
                best = r
    rep.table(["A", "beta", "return [95% CI]", "fall rate", "sat_frac",
               "clip_cmd", "r_ctrl", "tau_hip_common", "tau_hip_diff",
               "d_hip_common", "d_hip_diff"], rows)

    beta_star = float(str(best["cell"]).split("_b")[-1])
    rep.kv("beta* (best at A=1)", beta_star)
    rep.kv("R(A=1, beta*)", fmt_ci(best["_returns"]))
    rep.kv("0.9 * B0 (the V1 bar)", f"{0.9 * b0:.1f}")
    passed = float(best["ret_mean"]) >= 0.9 * b0
    rep.verdict("V1 existence-control  (max_beta R(A=1) >= 0.9*B0)", passed)
    if not passed:
        rep.note("**V1 FAIL => H-A1/H-A2 are live.** Read the mode columns: if the "
                 "failure coincides with sat_frac spikes driven by difference-DC "
                 "amplification (tau_hip_diff high at the failing betas), then "
                 "existence requires GAIT RESHAPING (H-A2) and **E4 becomes the "
                 "existence test of record**. Per §10.2 ordering rule 1: skip "
                 "F1c/F2/F3/D2 at sigma=0.9 entirely, run e2b to locate sigma*, and "
                 "re-run Tier 0 + F1c at the repaired setting (R-a or R-b). The "
                 "campaign continues on the repaired env and the report records the "
                 "ill-posedness finding (spec §9.3).")
    if not ok:
        rep.note("The A=0 control for `cancel` did NOT reproduce the stationary "
                 "return — every E2/E3 number above is confounded. Fix the probe "
                 "before reading V1 (abort rule 5).")
    return beta_star, best


def stage_e2b(cfg, ckpt, out, rep, args, cells, beta_star, b0=None):
    rep.h2("E2b — severity sweep: the theory figure and the feasibility frontier")
    rep.line("The phase-boundary theorem predicts recovery collapses as c crosses "
             "1 regardless of beta (the difference-mode zero leaves the unit "
             "circle => non-minimum-phase => uncompensable).")
    rep.note("**beta is RE-OPTIMIZED at each sigma.** A fixed beta cannot locate "
             "the frontier: beta* is chosen at sigma=0.9 where cancellation is "
             "*failing*, so it is the LEAST-harmful (smallest) gain, not a working "
             "one. Sweeping sigma at that beta understates sigma* badly — at a "
             "lower sigma a HIGHER beta works. So each sigma here reports "
             "max_beta R over the same {0.25,0.5,0.75,1.0} grid E2 used, which is "
             "the honest existence frontier (spec §4.2 / §9.2).")
    beta_grid = (0.25, 0.5, 0.75, 1.0)
    bar = 0.9 * b0 if b0 else None
    rows = []
    frontier = {}
    # grid extends down to 0.4: the E2 A-sweep shows max_beta cancellation already
    # passing near c=0.45 (A=0.5, beta=1 -> ~5040) and failing by c=0.9, so sigma*
    # sits low and must be bracketed from below.
    for sev in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3):
        best_r, best_b = None, None
        for beta in beta_grid:
            r = run_cell(cfg, ckpt, out, "e2b", f"sev{sev}_b{beta}",
                         f"cancel:beta={beta}", A=1.0, severity=sev,
                         episodes=args.episodes, device=args.device)
            cells.append(r)
            if best_r is None or r["ret_mean"] > best_r["ret_mean"]:
                best_r, best_b = r, beta
        frontier[sev] = float(best_r["ret_mean"])
        gate = ("" if bar is None else
                ("PASS" if best_r["ret_mean"] >= bar else "fail"))
        rows.append([sev, best_b, fmt_ci(best_r["_returns"]), best_r["fall_rate"],
                     best_r["sat_frac"], gate])
    if args.dcap_leg:
        best_r, best_b = None, None
        for beta in beta_grid:
            r = run_cell(cfg, ckpt, out, "e2b", f"sev1.3_dcap0.5_b{beta}",
                         f"cancel:beta={beta}", A=1.0, severity=1.3, dcap=0.5,
                         episodes=args.episodes, device=args.device)
            cells.append(r)
            if best_r is None or r["ret_mean"] > best_r["ret_mean"]:
                best_r, best_b = r, beta
        gate = ("" if bar is None else
                ("PASS" if best_r["ret_mean"] >= bar else "fail"))
        rows.append(["1.3+DCAP0.5", best_b, fmt_ci(best_r["_returns"]),
                     best_r["fall_rate"], best_r["sat_frac"], gate])
    rep.table(["sigma", "best beta", "max_beta R [95% CI]", "fall rate",
               "sat_frac", ">=0.9*B0?"], rows)
    if bar is not None:
        passing = [s for s in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3)
                   if frontier.get(s, -1) >= bar]
        sig_star = max(passing) if passing else None
        rep.kv("0.9*B0 bar", f"{bar:.0f}")
        rep.kv("sigma* (largest sigma with max_beta R >= 0.9*B0)",
               f"{sig_star}" if sig_star is not None
               else "below 0.4 — even the mildest swept severity fails; "
                    "widen the grid down or reconsider the harm channel (E6)")
        if sig_star is not None:
            rep.note(f"**Redesign target (R-a): sigma ~= {sig_star} minus a margin** "
                     f"(spec §9.2). Re-verify the blind collapse is still deep at "
                     f"that sigma (guide constraint 4) before adopting it — a lower "
                     f"sigma both restores compensability AND shrinks the collapse.")
    rep.note("The DCAP leg, if run, is Route-B evidence: a cap that restores "
             "feasibility at a sigma that otherwise fails, keeping the blind "
             "problem hard.")


def stage_e3(cfg, ckpt, out, rep, args, cells, beta_star, exact_ret):
    rep.h2("E3 — the degraded-information frontier")
    rep.line("**The single most design-relevant curve of the campaign.** It turns "
             "'the identifier worked but the method failed' into an engineering "
             "spec: any future estimator must beat (k*, h*, r*).")
    base = float(np.mean(exact_ret))
    bar = 0.9 * base
    rows = []
    results = {}
    grids = ([(f"delay:{k}", "delay", k) for k in (1, 2, 4, 8, 16, 32)]
             + [(f"ema:{h}", "ema", h) for h in (1, 4, 16, 64)]
             + [("dc:64", "dc", 64)]
             + [(f"noise:{r}", "noise", r) for r in (0.1, 0.2, 0.4)]
             + [("sign_leg", "sign_leg", 0)])
    fail_family = set()
    for spec, fam, val in grids:
        # budget: drop finer cells once a coarser one in the same family fails
        if fam in fail_family and fam in ("delay", "ema"):
            rows.append([spec, "skipped", "-", "-",
                         "coarser cell already failed (budget rule, spec §4.3)"])
            continue
        r = run_cell(cfg, ckpt, out, "e3", spec.replace(":", ""),
                     f"cancel:beta={beta_star},transform={spec}", A=1.0,
                     episodes=args.episodes, device=args.device)
        cells.append(r)
        ok = float(r["ret_mean"]) >= bar
        results[spec] = float(r["ret_mean"])
        rows.append([spec, fmt_ci(r["_returns"]), r["fall_rate"], r["sat_frac"],
                     "PASS" if ok else "fail"])
        if not ok and fam in ("delay", "ema"):
            fail_family.add(fam)
    rep.kv("R(exact beta*) — the reference", f"{base:.1f}")
    rep.kv("the 0.9x bar", f"{bar:.1f}")
    rep.table(["transform", "return [95% CI]", "fall rate", "sat_frac",
               ">= 0.9*R(exact)?"], rows)

    def _star(fam, vals):
        good = [v for v in vals if results.get(f"{fam}:{v}", -1) >= bar]
        return max(good) if good else 0

    k_star = _star("delay", [1, 2, 4, 8, 16, 32])
    h_star = _star("ema", [1, 4, 16, 64])
    r_star = max([v for v in (0.1, 0.2, 0.4)
                  if results.get(f"noise:{v}", -1) >= bar], default=0.0)
    rep.h3("V8 — the estimator requirement frontier")
    rep.kv("k* (max tolerable lag, steps)", k_star)
    rep.kv("h* (max tolerable EMA half-life)", h_star)
    rep.kv("r* (max tolerable relative noise)", r_star)
    rep.kv("dc:64 (is a SLOW CHART sufficient at all?)",
           "PASS" if results.get("dc:64", -1) >= bar else "fail")
    rep.verdict("V8 bandwidth budget 'tight' (k* < 4)", k_star < 4,
                f"k*={k_star}")
    rep.note("**Retro-diagnosis of ECL, read directly off this table.** The "
             "envelope's ~15k-step lag and the identifier's window granularity sit "
             "at the far right of the delay axis. If the frontier says '<= 4 steps "
             "of lag or bust', then every slow-chart method — ECL, c-conditioning, "
             "any windowed identifier — was doomed **regardless of accuracy**, and "
             "the method leaf is L1 (fast local estimation). If instead dc:64 "
             "passes, slow charts were sufficient and the failure was OPTIMIZATION "
             "(leaves L2/L3/L4). These two readings point at opposite methods; "
             "this is the table that decides.")
    return {"k_star": k_star, "h_star": h_star, "r_star": r_star,
            "dc64": results.get("dc:64", float("nan")), "exact": base}


def stage_e3dob(cfg, ckpt, out, rep, args, cells, beta_star, exact_ret):
    rep.h2("E3-DOB — the decentralized-feasibility certificate")
    if not args.dob:
        rep.line("  skipped: pass --dob <dob_filter.npz> (produced by "
                 "`python -m harl.envs.mamujoco.diag.sysid --export_dob ...`).")
        rep.note("Abort rule 4 (§10.2): if E5's R^2 < 0.3 everywhere on source (a), "
                 "SKIP this stage and log V6 fail early — do not spend the eval "
                 "pass.")
        return None
    base = float(np.mean(exact_ret))
    r = run_cell(cfg, ckpt, out, "e3dob", "dob",
                 f"cancel:beta={beta_star},transform=dob:{args.dob}", A=1.0,
                 episodes=args.episodes, device=args.device, dump_traj=True)
    cells.append(r)
    bar = 0.85 * base
    passed = float(r["ret_mean"]) >= bar
    rep.kv("R(dob)", fmt_ci(r["_returns"]))
    rep.kv("0.85 * R(exact beta*) — the bar", f"{bar:.1f}")
    rep.verdict("E3-DOB (>= 0.85 * R(exact beta*))", passed)
    if passed:
        rep.note("**A decentralized solution demonstrably exists end-to-end** — "
                 "existence + observability + control, all scripted, no learning. "
                 "This is the strongest certificate the campaign can produce, and "
                 "it makes leaf L1 (MA-RMA/DOB) a build-what-already-works task.")
    else:
        rep.note("If E5's OPEN-LOOP R^2 was high but this closed loop fails, the "
                 "cause is distribution shift: the filter was fit off-policy and "
                 "the corrected dynamics move the states. Spec §4.3 allows ONE "
                 "iteration (twice at most): refit E5 on THIS run's closed-loop "
                 "NPZ dump (a DAgger step — the dump is on, under "
                 "diag_out/*/recorder/e3dob_dob/) and re-run. Persistent failure "
                 "=> local information is insufficient UNDER THE CORRECTED "
                 "DYNAMICS => H-B2 confirmed at the control level.")
    return r


def stage_e4(cfg, ckpt, out, rep, args, cells, b0):
    rep.h2("E4 — the information-free escape (sum-zero gait)")
    rep.line("Zeroing the common mode makes the cross-coupling a PRIVATE, "
             "self-inflicted, predictable gain droop: on the sum-zero manifold "
             "s_i = -tau_i exactly. Uses **no d at all**.")
    rows = {}
    for A in (0.0, 1.0):
        r = run_cell(cfg, ckpt, out, "e4", f"A{A}", "project_sumzero", A=A,
                     episodes=args.episodes, device=args.device)
        cells.append(r)
        rows[A] = r
    rep.table(["A", "return [95% CI]", "fall rate", "sumzero_resid", "clip_cmd",
               "tau_hip_common", "tau_hip_diff"],
              [[A, fmt_ci(rows[A]["_returns"]), rows[A]["fall_rate"],
                rows[A]["sumzero_resid"], rows[A]["clip_frac_cmd"],
                rows[A]["tau_hip_common"], rows[A]["tau_hip_diff"]]
               for A in (0.0, 1.0)])
    rep.line(f"  A=0 measures the COST OF THE CONSTRAINT itself (common-mode hip "
             f"torque may be load-bearing for stock walking): "
             f"{rows[0.0]['ret_mean']:.0f} vs B0 {b0:.0f} "
             f"({rows[0.0]['ret_mean'] / max(b0, 1e-9):.0%} of B0)")
    rep.line(f"  sumzero_resid > 0 means the post-projection clip re-broke the "
             f"manifold — read the number, do not assume it is 0.")
    passed = float(rows[1.0]["ret_mean"]) >= 0.8 * b0
    rep.verdict("E4 information-free near-solution (A=1 >= 0.8*B0)", passed)
    if passed:
        rep.note("**An information-free near-solution exists.** The peak game is "
                 "solvable by coordination / gait reshaping alone; every "
                 "estimation-centric story — including ECHO-R's and ECL's premise — "
                 "was aiming at the wrong bottleneck. The method leaf becomes the "
                 "equilibrium-shaping one (L2/L5 with a sum-zero-biased solution "
                 "concept), and this is a T4-existence validation for the paper.")
    return rows


def stage_e6(cfg, ckpt, out, rep, args, cells, e1_rows):
    rep.h2("E6 — harm-channel attribution (hip vs ankle)")
    rows = []
    got = {}
    base = e1_rows[0.0] if e1_rows else None
    for mask in ("hip", "ankle"):
        r = run_cell(cfg, ckpt, out, "e6", mask, "identity", A=1.0, mask=mask,
                     episodes=args.episodes, device=args.device)
        cells.append(r)
        got[mask] = r
        dd = deficit_decomposition(base, r) if base else None
        rows.append([mask, fmt_ci(r["_returns"]), r["fall_rate"],
                     r["median_time_to_fall"], r["fwd_vel"],
                     f"{dd['termination_share']:.0%}" if dd else "-"])
    if e1_rows:
        r = e1_rows[1.0]
        dd = deficit_decomposition(base, r)
        rows.append(["both (=E1@A=1)", fmt_ci(r["_returns"]), r["fall_rate"],
                     r["median_time_to_fall"], r["fwd_vel"],
                     f"{dd['termination_share']:.0%}"])
    rep.table(["live channel", "return [95% CI]", "fall rate", "median TTF",
               "fwd_vel", "termination share"], rows)
    rep.note("Feeds the redesign kit (spec §9.2): **if falls are hip-channel-driven "
             "and ankle-only still produces a deep but GRACEFUL "
             "(achievement-mediated) collapse, R-c is the leading redesign** — make "
             "ankle-only coupling the design default with sigma retuned up: "
             "propulsion loss instead of posture loss, WP-3 satisfied by "
             "construction, hips (posture) untouched.")
    return got


# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="PCR Tier-0 scripted probes.")
    ap.add_argument("--ckpt", required=True,
                    help="path to a run's models/ dir (F0). config.json must sit "
                         "beside or one level up.")
    ap.add_argument("--stage", default="all",
                    help="e1 | d0 | e2 | e2b | e3 | e3dob | e4 | e6 | all | custom")
    ap.add_argument("--out", default="./diag_out/tier0")
    ap.add_argument("--episodes", type=int, default=40,
                    help="per cell. 40 gives ~+/-150-250 on Ant (return std "
                         "400-800) — sufficient for gates that are all >= 500-point "
                         "contrasts (spec §3.3).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dob", default=None, help="E3-DOB filter npz (from sysid)")
    ap.add_argument("--beta_star", type=float, default=None,
                    help="skip E2's grid and use this beta* for E3/E2b")
    ap.add_argument("--dcap_leg", action="store_true",
                    help="E2b's optional sigma=1.3 + DCAP=0.5 leg (P1, Route-B "
                         "evidence)")
    # --- custom single-cell mode (the shape shown in the spec's Appendix B) ---
    ap.add_argument("--probe", default=None)
    ap.add_argument("--A", type=float, default=None)
    ap.add_argument("--severity", type=float, default=None)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--dcap", type=float, default=None)
    ap.add_argument("--dump_traj", action="store_true")
    args = ap.parse_args(argv)

    cfg, cfg_path = load_run_config(args.ckpt)
    rep = DebugReport(os.path.join(args.out, f"tier0_{args.stage}.md"),
                      title=f"Tier 0 — {args.stage}",
                      subtitle="eval-only scripted probes; no gradient anywhere")
    rep.kv("checkpoint", os.path.abspath(args.ckpt))
    rep.kv("run config", os.path.abspath(cfg_path))
    rep.kv("episodes per cell", args.episodes)
    cells = []

    if args.probe is not None:
        r = run_cell(cfg, args.ckpt, args.out, "custom", "cell", args.probe,
                     A=args.A, severity=args.severity, mask=args.mask,
                     dcap=args.dcap, episodes=args.episodes,
                     dump_traj=args.dump_traj, device=args.device)
        cells.append(r)
        rep.kv("return", fmt_ci(r["_returns"]))
        rep.table(_CELL_COLS, [_row_list(r)])
        write_csv(os.path.join(args.out, "tier0_cells.csv"), _CELL_COLS,
                  [_row_list(c) for c in cells])
        rep.close()
        return 0

    want = ({"e1", "d0", "e2", "e2b", "e3", "e3dob", "e4", "e6"}
            if args.stage == "all" else {args.stage})
    e1_rows, b0, b0_returns, beta_star, exact_ret = {}, None, None, args.beta_star, None

    # E1 establishes B0, which is the denominator of V1/V2/V4/V7, the A=0 control's
    # reference, and E2b's frontier gate — so it runs whenever anything downstream
    # needs it (e2b included: its sigma* gate is R = 0.9*B0).
    if want & {"e1", "e2", "e2b", "e4", "e6"}:
        e1_rows = stage_e1(cfg, args.ckpt, args.out, rep, args, cells)
        b0 = float(e1_rows[0.0]["ret_mean"])
        b0_returns = e1_rows[0.0]["_returns"]
    if "d0" in want:
        stage_d0(cfg, args.ckpt, args.out, rep, args, cells)
    if "e2" in want:
        beta_star, best = stage_e2(cfg, args.ckpt, args.out, rep, args, cells,
                                   b0_returns, b0)
        exact_ret = best["_returns"]
    if "e2b" in want:
        # E2b re-optimizes beta at each sigma now, so it no longer needs a
        # carried-over beta*; run it standalone as `--stage e2b`.
        stage_e2b(cfg, args.ckpt, args.out, rep, args, cells, beta_star, b0=b0)
    if "e3" in want and beta_star is not None:
        if exact_ret is None:
            r = run_cell(cfg, args.ckpt, args.out, "e3", "exact_ref",
                         f"cancel:beta={beta_star}", A=1.0, episodes=args.episodes,
                         device=args.device)
            cells.append(r)
            exact_ret = r["_returns"]
        stage_e3(cfg, args.ckpt, args.out, rep, args, cells, beta_star, exact_ret)
    if "e3dob" in want and beta_star is not None:
        if exact_ret is None:
            # standalone `--stage e3dob` never ran E2, so the exact-cancellation
            # reference the 0.85 bar is measured against does not exist yet —
            # compute it here (mirrors the e3 block), else the stage silently
            # produces zero cells.
            r = run_cell(cfg, args.ckpt, args.out, "e3dob", "exact_ref",
                         f"cancel:beta={beta_star}", A=1.0, episodes=args.episodes,
                         device=args.device)
            cells.append(r)
            exact_ret = r["_returns"]
        stage_e3dob(cfg, args.ckpt, args.out, rep, args, cells, beta_star, exact_ret)
    if "e4" in want:
        stage_e4(cfg, args.ckpt, args.out, rep, args, cells, b0)
    if "e6" in want:
        stage_e6(cfg, args.ckpt, args.out, rep, args, cells, e1_rows)

    if (want & {"e3", "e3dob"}) and beta_star is None:
        rep.note("e3 / e3dob need beta*: run --stage e2 first, or pass "
                 "--beta_star <value>. (e2b no longer needs it — it re-optimizes "
                 "beta per sigma.)")

    path = write_csv(os.path.join(args.out, "tier0_cells.csv"), _CELL_COLS,
                     [_row_list(c) for c in cells])
    rep.h2("artifacts")
    rep.kv("per-cell table", path)
    rep.kv("flight-recorder CSVs", os.path.join(args.out, "recorder", "<cell>/"))
    rep.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
