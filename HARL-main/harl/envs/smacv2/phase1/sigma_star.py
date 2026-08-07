"""Phase-1 sigma-star sweep for SMACv2-CWD (PACT pipeline, Part II).

Load a trained *blind* baseline, freeze the CWD driver at its peak, and roll the
frozen policy through the scripted-compensation probe over a severity x gain grid.
Take the max over gain per severity (Pitfall #1: never fix beta across the sweep)
and report sigma-star = the largest severity at which compensation still holds the
peak to >= 0.90 * B0.

This never trains and never touches the reward.  It reuses ONE StarCraft II process
for the whole sweep (reconfiguring the probe between cells) and drives the eval
rollout itself (mirroring OnPolicyBaseRunner.eval / .render), so it needs no runner.

Usage (on the run machine, with a trained B0 checkpoint):

    python -m harl.envs.smacv2.phase1.sigma_star \
        --load_config results/.../<B0 run>/config.json \
        --model_dir   results/.../<B0 run>/models \
        --episodes 40 \
        --sigmas 0.0,0.25,0.5,0.75,1.0,1.5,2.0 \
        --betas  0.0,0.25,0.5,0.75,1.0 \
        --out    results/smacv2_cwd_phase1

Outputs (under --out): ``sigma_star.csv`` (per-cell) and ``sigma_star_summary.md``.

Read four things, not just sigma-star (pipeline §II.5): the return-vs-sigma
crossing, best_beta vs sigma (the loop-gain fingerprint), the residual/saturation
onset, and win-rate alongside return.  The continuous-re-aim certificate arm should
recover B0 at *every* sigma -- if it does not, suspect a wiring/units bug before
believing any discrete result.
"""

import argparse
import json
import os
import os.path as osp

import numpy as np
import torch

from harl.algorithms.actors import ALGO_REGISTRY
from harl.utils.trans_tools import _t2n


# --------------------------------------------------------------------------- io
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def parse_floats(s):
    return [float(x) for x in str(s).split(",") if x.strip() != ""]


# ---------------------------------------------------------------- actor loading
def build_actors(algo, algo_args, env, device):
    """Mirror OnPolicyBaseRunner's actor construction + restore (actors only)."""
    share_param = algo_args["algo"]["share_param"]
    num_agents = env.n_agents
    merged = {**algo_args["model"], **algo_args["algo"]}
    actors = []
    if share_param:
        agent = ALGO_REGISTRY[algo](
            merged, env.observation_space[0], env.action_space[0], device=device
        )
        actors = [agent for _ in range(num_agents)]
    else:
        for aid in range(num_agents):
            actors.append(
                ALGO_REGISTRY[algo](
                    merged,
                    env.observation_space[aid],
                    env.action_space[aid],
                    device=device,
                )
            )
    # restore
    model_dir = algo_args["train"]["model_dir"]
    for aid in range(num_agents):
        sd = torch.load(
            osp.join(str(model_dir), f"actor_agent{aid}.pt"), map_location=device
        )
        actors[aid].actor.load_state_dict(sd)
        actors[aid].prep_rollout()
    return actors


# --------------------------------------------------------------------- rollouts
@torch.no_grad()
def run_episode(env, actors, num_agents, rec_n, hidden):
    """One deterministic episode on the single probe env; mirrors the eval loop."""
    obs, _state, avail = env.reset()
    obs = np.array(obs, dtype=np.float32)          # (n_agents, obs_dim)
    avail = np.array(avail)                         # (n_agents, n_actions)
    rnn = np.zeros((num_agents, rec_n, hidden), dtype=np.float32)
    masks = np.ones((num_agents, 1), dtype=np.float32)

    ret = 0.0
    length = 0
    resid_sum = 0.0
    dsat_sum = 0.0
    changed_sum = 0.0
    won = False
    while True:
        act_list = []
        for aid in range(num_agents):
            a, r = actors[aid].act(
                obs[aid : aid + 1],
                rnn[aid : aid + 1],
                masks[aid : aid + 1],
                avail[aid : aid + 1],
                deterministic=True,
            )
            rnn[aid] = _t2n(r)[0]
            act_list.append(int(_t2n(a)[0, 0]))
        actions = np.array(act_list, dtype=np.int64).reshape(num_agents, 1)

        obs_l, _state_l, rewards, dones, infos, avail_l = env.step(actions)
        ret += float(rewards[0][0])
        length += 1
        info0 = infos[0]
        resid_sum += float(info0.get("phase1_residual", 0.0))
        dsat_sum += float(info0.get("phase1_dsat_frac", 0.0))
        changed_sum += float(info0.get("phase1_changed_frac", 0.0))
        if bool(dones[0]):
            won = bool(info0.get("battle_won", False))
            break
        obs = np.array(obs_l, dtype=np.float32)
        avail = np.array(avail_l)

    return {
        "ret": ret,
        "won": won,
        "length": length,
        "resid": resid_sum / max(1, length),
        "dsat": dsat_sum / max(1, length),
        "changed": changed_sum / max(1, length),
    }


def _bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def run_cell(env, actors, num_agents, rec_n, hidden, n_eps,
             comp_mode, beta, severity, freeze):
    env.configure_probe(
        comp_mode=comp_mode, comp_beta=beta, severity=severity, freeze=freeze
    )
    rets, wins, resids, dsats, lens, changes = [], [], [], [], [], []
    for _ in range(n_eps):
        r = run_episode(env, actors, num_agents, rec_n, hidden)
        rets.append(r["ret"])
        wins.append(1.0 if r["won"] else 0.0)
        resids.append(r["resid"])
        dsats.append(r["dsat"])
        lens.append(r["length"])
        changes.append(r["changed"])
    ret_lo, ret_hi = _bootstrap_ci(rets)
    win_lo, win_hi = _bootstrap_ci(wins)
    return {
        "comp_mode": comp_mode,
        "sigma": severity,
        "beta": beta,
        "mean_return": float(np.mean(rets)),
        "ret_ci_lo": ret_lo,
        "ret_ci_hi": ret_hi,
        "win_rate": float(np.mean(wins)),
        "win_ci_lo": win_lo,
        "win_ci_hi": win_hi,
        "mean_resid": float(np.mean(resids)),
        "mean_dsat": float(np.mean(dsats)),
        "mean_changed": float(np.mean(changes)),
        "mean_len": float(np.mean(lens)),
        "n_eps": n_eps,
    }


# -------------------------------------------------------------------- reporting
_CSV_COLS = [
    "comp_mode", "sigma", "beta", "mean_return", "ret_ci_lo", "ret_ci_hi",
    "win_rate", "win_ci_lo", "win_ci_hi", "mean_resid", "mean_dsat",
    "mean_changed", "mean_len", "n_eps",
]


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(_CSV_COLS) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in _CSV_COLS) + "\n")


def summarize(rows, b0_ret, b0_win, bar_frac, out_md, args):
    bar_ret = bar_frac * b0_ret
    bar_win = bar_frac * b0_win

    def cells(mode):
        return [r for r in rows if r["comp_mode"] == mode]

    disc = cells("discrete")
    sigmas = sorted({r["sigma"] for r in disc if r["sigma"] > 0})

    lines = []
    lines.append("# SMACv2-CWD — Phase-1 sigma-star certification\n")
    lines.append(
        f"- baseline **B0**: return **{b0_ret:.1f}**, win-rate **{b0_win:.3f}** "
        f"(severity 0, comp off)\n"
        f"- bar (`{bar_frac:g}·B0`): return **{bar_ret:.1f}**, win-rate **{bar_win:.3f}**\n"
        f"- episodes/cell **{args.episodes}**, driver frozen at peak A="
        f"**{args.freeze}** ⇒ effective coupling c = severity\n"
    )

    # per-sigma max over beta (discrete)
    lines.append("\n## Discrete re-aim (the realizable frontier — σ* headline)\n")
    lines.append("| σ | best β (ret) | max_β return | win@bestβ | resid | dsat | ≥bar(ret)? | ≥bar(win)? |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in sigmas:
        cs = [r for r in disc if r["sigma"] == s]
        best = max(cs, key=lambda r: r["mean_return"])
        best_w = max(cs, key=lambda r: r["win_rate"])
        ok_r = best["mean_return"] >= bar_ret
        ok_w = best_w["win_rate"] >= bar_win
        lines.append(
            f"| {s:g} | {best['beta']:g} | {best['mean_return']:.1f} | "
            f"{best['win_rate']:.3f} | {best['mean_resid']:.2f} | "
            f"{best['mean_dsat']:.2f} | {'✅' if ok_r else '❌'} | "
            f"{'✅' if ok_w else '❌'} |"
        )
    # sigma-star = largest sigma with max_beta >= bar, contiguous from the low end
    def contiguous_star(metric_ok):
        star = None
        for s in sigmas:
            cs = [r for r in disc if r["sigma"] == s]
            if metric_ok(cs):
                star = s
            else:
                break
        return star

    ss_ret = contiguous_star(
        lambda cs: max(c["mean_return"] for c in cs) >= bar_ret
    )
    ss_win = contiguous_star(
        lambda cs: max(c["win_rate"] for c in cs) >= bar_win
    )
    lines.append(
        f"\n**σ\\* (return)** = **{ss_ret}**  |  **σ\\* (win-rate)** = **{ss_win}**  "
        f"(largest σ, contiguous from 0, with max_β ≥ bar)\n"
    )
    lines.append(
        "\nbest_β vs σ: a crossover from 1.0 toward 0 marks the loop-gain wall "
        "(over-cancellation hurts). For CWD the binding resource is discrete "
        "expressibility, so also watch `resid` climbing with σ (until the "
        "`_DCAP`=2.0 cap flattens |d|).\n"
    )

    # continuous certificate
    cont = cells("continuous")
    if cont:
        lines.append("\n## Continuous re-aim @ β=1 (invertibility / transparency certificate)\n")
        lines.append("CWD's harm is a pure translation, so β=1 continuous re-aim makes the "
                     "delivered target byte-identical to stock SMACv2 — it should recover "
                     "B0 at **every** σ. Deviations here are a bug, not a frontier.\n")
        lines.append("| σ | return | win-rate | ≈B0? |")
        lines.append("|---|---|---|---|")
        for s in sorted({r["sigma"] for r in cont}):
            c = [r for r in cont if r["sigma"] == s][0]
            near = c["mean_return"] >= bar_ret  # bar-based: should clear 0.9·B0
            lines.append(
                f"| {s:g} | {c['mean_return']:.1f} | {c['win_rate']:.3f} | "
                f"{'✅' if near else '⚠️'} |"
            )

    lines.append("\n## Decision (pipeline §II.6)\n")
    lines.append(
        f"- If your target severity ≤ σ\\* → WELL-POSED: build PACT (Phase 2) at σ ≤ σ\\*.\n"
        f"- Else → redesign (lower σ to σ\\*−0.05, attenuate the channel, or cap the driver) "
        f"and re-run.\n"
    )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return ss_ret, ss_win


# -------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="SMACv2-CWD Phase-1 sigma-star sweep")
    p.add_argument("--load_config", required=True,
                   help="B0 run's config.json (algo_args/env_args/main_args)")
    p.add_argument("--model_dir", required=True,
                   help="dir with actor_agent{0..N-1}.pt from the B0 run")
    p.add_argument("--map_name", default=None, help="override map_name")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--sigmas", default="0.0,0.25,0.5,0.75,1.0,1.5,2.0")
    p.add_argument("--betas", default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--comp_mode", default="discrete",
                   choices=["discrete", "continuous", "none"])
    p.add_argument("--freeze", type=float, default=1.0,
                   help="frozen driver value (peak=1.0); use -1 for the live ramp")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--b0_return", type=float, default=None,
                   help="override measured B0 return (else measured at σ=0)")
    p.add_argument("--b0_win", type=float, default=None)
    p.add_argument("--bar_frac", type=float, default=0.90)
    p.add_argument("--no_certificate", action="store_true",
                   help="skip the continuous-re-aim invertibility certificate")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results/smacv2_cwd_phase1")
    args = p.parse_args()

    cfg = load_config(args.load_config)
    algo = cfg["main_args"]["algo"]
    env_name = cfg["main_args"]["env"]
    assert env_name == "smacv2", f"expected smacv2 config, got {env_name}"
    algo_args = cfg["algo_args"]
    env_args = dict(cfg["env_args"])
    if args.map_name:
        env_args["map_name"] = args.map_name
    algo_args["train"]["model_dir"] = args.model_dir

    seed = args.seed if args.seed is not None else algo_args["seed"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    rec_n = algo_args["model"]["recurrent_n"]
    hidden = algo_args["model"]["hidden_sizes"][-1]
    freeze = None if args.freeze < 0 else args.freeze

    # --- build ONE probe env (single SC2 process) + the frozen baseline actors ---
    env_args["phase1"] = True
    env_args["phase1_cfg"] = {"comp_mode": "none", "comp_beta": 0.0, "freeze": freeze}
    from harl.envs.smacv2.phase1.probe_env import SMACv2ProbeEnv

    env = SMACv2ProbeEnv(env_args)
    env.seed(seed)
    num_agents = env.n_agents
    actors = build_actors(algo, algo_args, env, device)
    print(f"[phase1] map={env_args['map_name']} algo={algo} n_agents={num_agents} "
          f"obs={env.observation_space[0].shape} n_actions={env.action_space[0].n} "
          f"move_amount={env._cwd_move_amount}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    rows = []

    # --- B0: undisturbed baseline (severity 0, comp off) ---
    b0 = run_cell(env, actors, num_agents, rec_n, hidden, args.episodes,
                  "none", 0.0, 0.0, freeze)
    rows.append(b0)
    b0_ret = args.b0_return if args.b0_return is not None else b0["mean_return"]
    b0_win = args.b0_win if args.b0_win is not None else b0["win_rate"]
    print(f"[phase1] B0: return={b0_ret:.1f} win={b0_win:.3f}", flush=True)

    # --- CHECK A (transparency): σ=0 discrete β=1 must reproduce B0 exactly ---
    tA = run_cell(env, actors, num_agents, rec_n, hidden, args.episodes,
                  "discrete", 1.0, 0.0, freeze)
    rows.append(tA)
    if abs(tA["mean_return"] - b0["mean_return"]) > 0.02 * (abs(b0["mean_return"]) + 1):
        print("[phase1][WARN] Transparency check FAILED: σ=0 discrete β=1 "
              f"({tA['mean_return']:.1f}) != B0 ({b0['mean_return']:.1f}). "
              "The probe is corrupting the action independent of the NS.", flush=True)
    else:
        print("[phase1] Transparency ✅ (σ=0 discrete β=1 ≈ B0)", flush=True)

    sigmas = parse_floats(args.sigmas)
    betas = parse_floats(args.betas)

    # --- the discrete σ×β sweep (the headline) ---
    for s in sigmas:
        if s == 0.0:
            continue
        for b in betas:
            c = run_cell(env, actors, num_agents, rec_n, hidden, args.episodes,
                         "discrete", b, s, freeze)
            rows.append(c)
            print(f"[phase1] discrete σ={s:g} β={b:g}: ret={c['mean_return']:.1f} "
                  f"win={c['win_rate']:.3f} resid={c['mean_resid']:.2f} "
                  f"dsat={c['mean_dsat']:.2f}", flush=True)
        write_csv(rows, osp.join(args.out, "sigma_star.csv"))  # incremental

    # --- CHECK B (works-when-it-should): at the lowest σ>0, best-β > blind ---
    lo = min(s for s in sigmas if s > 0)
    lo_cells = [r for r in rows if r["comp_mode"] == "discrete" and r["sigma"] == lo]
    blind_lo = next((r for r in lo_cells if r["beta"] == 0.0), None)
    best_lo = max(lo_cells, key=lambda r: r["mean_return"])
    if blind_lo is not None and best_lo["mean_return"] <= blind_lo["mean_return"] + 1e-6:
        print(f"[phase1][WARN] Works-when-it-should FAILED at σ={lo:g}: comp never "
              "beats blind. Likely the move-direction map (probe_env._DIR) is wrong "
              "for this smacv2 build — VERIFY get_agent_action's N/S/E/W convention.",
              flush=True)
    else:
        print(f"[phase1] Works-when-it-should ✅ (σ={lo:g}: best-β {best_lo['mean_return']:.1f} "
              f"> blind {blind_lo['mean_return']:.1f})", flush=True)

    # --- continuous re-aim invertibility certificate (β=1 across σ) ---
    if not args.no_certificate:
        for s in sigmas:
            if s == 0.0:
                continue
            c = run_cell(env, actors, num_agents, rec_n, hidden, args.episodes,
                         "continuous", 1.0, s, freeze)
            rows.append(c)
            print(f"[phase1] cont(cert) σ={s:g} β=1: ret={c['mean_return']:.1f} "
                  f"win={c['win_rate']:.3f} (should ≈ B0)", flush=True)

    write_csv(rows, osp.join(args.out, "sigma_star.csv"))
    ss_ret, ss_win = summarize(
        rows, b0_ret, b0_win, args.bar_frac,
        osp.join(args.out, "sigma_star_summary.md"), args
    )
    print("\n" + "=" * 60)
    print(f"[phase1] σ* (return)   = {ss_ret}")
    print(f"[phase1] σ* (win-rate) = {ss_win}")
    print(f"[phase1] wrote {osp.join(args.out, 'sigma_star.csv')} and "
          f"{osp.join(args.out, 'sigma_star_summary.md')}")
    print("=" * 60, flush=True)
    env.close()


if __name__ == "__main__":
    main()
