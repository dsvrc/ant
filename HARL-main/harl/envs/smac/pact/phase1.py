"""PHASE 1 for SMAC-CWO: certify that a compensation solution EXISTS, and find sigma*.

`diagnose.py` answers "does the NS block a good policy?".  This answers the question
the pipeline actually gates on: **at severity sigma, can a PRIVILEGED, SCRIPTED
controller -- handed the true liability, no learning -- hold the harmed win-rate at
>= 0.90 * B0?**  If it cannot, no learner can either, and a PACT training run at that
sigma measures nothing (pipeline Pitfall 5: never conclude "unsolvable" from a failed
TRAINING run).  sigma* = the largest sigma that clears the bar; run Phase 2 below it.

WHY THIS MATTERS HERE.  CWO's harm is a stochastic shot DROP, so the "channel inverse"
is not a reparameterization -- you cannot un-drop a shot.  The only lever is to HOLD
FIRE so the shared bus cools.  That makes the whole method contingent on an arithmetic
fact: holding fire has to be worth more than the damage it costs.  With per-unit
throughput T(f) = f*(1 - ell(f)) at team firing fraction f, the concave branch has an
interior optimum, but once ell saturates at _LMAX the branch T = f*(1-_LMAX) is
strictly INCREASING -- greedy wins and there is nothing to certify.  (That was the
state of the 20M-step 3s5z run: SEVERITY=2.0 with _LMAX=0.6 made greedy f=1 worth
3.2x the "stagger" optimum, so the firing fraction at the driver peak and trough came
out 0.335 vs 0.345 -- no modulation, because none was ever profitable.)  The env now
prints this headroom in its banner; this script MEASURES it end to end.

THE SCRIPTED LAW (declaration #3 -- the same law Phase 2 has to learn)

  --law target  (DEFAULT)  p_hold_i = clip(beta * (1 - f*(A)/max(x2_i, eps)), 0, 1)
                steer the team's shared load onto the analytic optimum f*(A) -- the
                knee when c*KNEE >= 1, else 1/(2c) + KNEE/2, with c = A*sigma/(1-KNEE).
                A regulator: it settles at a load, which is what "coordinating" means.
                Its fixed point sits ABOVE f* for small beta, so sweep beta well past
                1 (the default grid goes to 5) or it never reaches the target.
  --law prop    p_hold_i = clip(beta * ell_i, 0, 1)
                hold fire in proportion to how jammed you would be -- the cruder law a
                policy might plausibly discover.  NOT the default: it is proportional
                feedback around a loop with ~1/(1-RHO) = 7 steps of lag and slope
                d(ell)/d(x2) = sigma/(1-KNEE) > 5, so it oscillates (hold nearly
                everything -> bus cools -> release -> bus reheats) instead of settling.
                It never occupies the coordinated operating point and so under-reports
                what compensation can do.
  beta=0 is the blind policy exactly, for both laws (the transparency check).

Both are driven by the TRUE per-unit liability read straight off the env (native
units, never piggybacked through a normalized obs vector -- Pitfall 3).  The shim is
memoryless, so no per-episode state can leak across resets (Pitfall 2).

READ THE METRIC NOTE BEFORE TRUSTING A NEGATIVE RESULT.  The bar defaults to RETURN,
not win rate.  SMAC 3s5z is a MIRROR match, so by Lanchester's square law the win/loss
outcome is a CLIFF in relative DPS: measured on a B0 checkpoint, a 27% shot-drop rate
took the win rate 0.600 -> 0.000 with episode length FALLING 45.7 -> 37.3 (the team
dies faster, it does not time out).  With no partial band, a 0.9*B0 WIN bar can only be
met by a controller that restores essentially all the lost damage -- and holding fire
buys damage with damage, so it structurally cannot.  A win-rate sweep therefore reports
"ILL-POSED at every severity" for a property of the map, not of the NS.  Return
degrades gracefully (18.2 -> 11.5 -> 3.4 over the same sweep) and has resolution.

PROTOCOL (pipeline II.3)
  bar = 0.90 * B0 ;  for each sigma: freeze the driver at its PEAK, sweep beta,
  take max_beta.  ***beta is re-optimized per sigma, never fixed*** (Pitfall 1) --
  the optimal gain falls as sigma rises and a fixed gain understates sigma*.
  sigma* = the largest sigma whose max_beta result clears the bar.

  EVERY CELL IS PRINTED, not just the per-sigma argmax.  Printing only the argmax is
  useless exactly when the sweep fails: with the metric 0 in every cell, argmax returns
  the FIRST cell (beta=0, i.e. blind), so the table silently shows uncompensated
  numbers at every severity -- held=0.000 and fire unchanged from B0 are the tell.
  Each sigma block also reports the lowest load the shim actually achieved against the
  target f*: if it never got there, that cell measured a controller that failed to
  control, not the environment's frontier.

  Two mandatory confounding checks run first (pipeline II.4):
    CHECK A transparency      severity 0, beta 0  must reproduce B0.
    CHECK B works-when-it-should  at LOW sigma compensation must actually help.
  Only if both pass is a high-sigma failure a property of the environment.

USAGE
    python -m harl.envs.smac.pact.phase1 \
        --load_config results/.../<B0 run>/config.json \
        --model_dir   results/.../<B0 run>/models \
        --sigmas 1.0,1.8,2.5,3.2 --episodes 40

B0 must be a checkpoint trained with the NS OFF (severity 0), and it must be STRONG.
A frontier measured against a marginal baseline is not a frontier: a policy that only
wins 0.60 has no margin, loses to any handicap at all, and drives sigma* to ~0 for
reasons that have nothing to do with the NS.  The script warns when B0 < 0.85.  B0 is
severity independent, so one checkpoint serves every cell -- nothing is retrained.
"""

import argparse
import json

import numpy as np
import torch

from harl.algorithms.actors import ALGO_REGISTRY
from harl.utils.trans_tools import _t2n


def build_actors(algo, algo_args, env, device):
    """Load the B0 actors.  Mirrors diagnose.py so both scripts stay interchangeable."""
    share = algo_args["algo"]["share_param"]
    merged = {**algo_args["model"], **algo_args["algo"]}
    n = env.n_agents
    if share:
        ag = ALGO_REGISTRY[algo](
            merged, env.observation_space[0], env.action_space[0], device=device
        )
        actors = [ag] * n
    else:
        actors = [
            ALGO_REGISTRY[algo](
                merged, env.observation_space[i], env.action_space[i], device=device
            )
            for i in range(n)
        ]
    md = algo_args["train"]["model_dir"]
    for i in range(n):
        actors[i].actor.load_state_dict(
            torch.load(f"{md}/actor_agent{i}.pt", map_location=device)
        )
        actors[i].prep_rollout()
    return actors


def reaim(acts, avail, env, beta, n_no_attack):
    """The scripted compensation law: PRE-SHIFT the commanded target so the channel's
    deflection lands it where the policy actually aimed.

    The env shifts the delivered target by s = round(ell_i*(K-1)) places along unit i's
    K attackable enemies, so commanding `desired - s_hat` cancels it exactly when
    s_hat = s (i.e. beta = 1).  This is a permutation inverse: it costs NOTHING, unlike
    the hold-fire law the drop channel forced, which bought damage with damage and
    could never return the team to B0 (see the HISTORY note in StarCraft2_Env.py).

    `env._cwo_ell` is the true liability the env will apply THIS step, read straight off
    the env rather than through obs so the controller cannot be handed a
    per-step-rescaled "oracle" (Pitfall 3).  Returns the number of shots re-aimed.
    Memoryless, so no per-episode state can leak across resets (Pitfall 2)."""
    if beta <= 0.0:
        return 0, 0  # identity shim: byte-for-byte the blind policy
    ell = np.asarray(env._cwo_ell, dtype=np.float64)
    n_re = n_shots = 0
    for i in range(len(acts)):
        if acts[i] < n_no_attack:
            continue
        n_shots += 1
        tg = np.where(np.asarray(avail[i][n_no_attack:]) > 0)[0]
        k = int(tg.size)
        if k <= 1 or ell[i] <= 0.0:
            continue
        s = int(round(beta * float(ell[i]) * (k - 1)))
        if s <= 0:
            continue
        cur = int(acts[i] - n_no_attack)
        where = np.where(tg == cur)[0]
        if where.size == 0:
            continue
        acts[i] = n_no_attack + int(tg[(int(where[0]) - s) % k])
        n_re += 1
    return n_re, n_shots


@torch.no_grad()
def eval_cell(env, actors, n, rec_n, hidden, n_eps, severity, freeze, beta,
              n_no_attack):
    """One (sigma, beta) cell: roll the frozen B0 policy through the probe shim."""
    env.snd_severity = float(severity)
    env.snd_freeze = freeze  # None (live) or a fixed A in [0,1]
    env._snd_payload = env._snd_driver_value()  # so step 1 of the cell sees the new A
    wins, lens, rets, drops, fires, x2s, holds = [], [], [], [], [], [], []
    for _ in range(n_eps):
        obs, _st, avail = env.reset()
        obs = np.array(obs, dtype=np.float32)
        avail = np.array(avail)
        rnn = np.zeros((n, rec_n, hidden), dtype=np.float32)
        masks = np.ones((n, 1), dtype=np.float32)
        ret = length = 0.0
        won = False
        drop_acc = fire_acc = x2_acc = 0.0
        n_held = n_wanted = 0
        while True:
            acts = []
            for i in range(n):
                a, r = actors[i].act(
                    obs[i : i + 1], rnn[i : i + 1], masks[i : i + 1],
                    avail[i : i + 1], deterministic=True,
                )
                rnn[i] = _t2n(r)[0]
                acts.append(int(_t2n(a)[0, 0]))

            # --- the probe shim: intercept -> apply the law -> step ------------
            n_re, n_sh = reaim(acts, avail, env, beta, n_no_attack)
            n_held += n_re
            n_wanted += n_sh
            # ------------------------------------------------------------------

            obs_l, _s, rew, dones, infos, avail_l = env.step(
                np.array(acts, dtype=np.int64)
            )
            ret += float(np.asarray(rew[0]).reshape(-1)[0])
            length += 1
            drop_acc += float(infos[0].get("cwo_drop_frac", 0.0))
            fire_acc += float(infos[0].get("cwo_fire_frac", 0.0))
            x2_acc += float(infos[0].get("cwo_x2_mean", 0.0))
            if bool(np.asarray(dones).reshape(-1)[0]):
                won = bool(infos[0].get("won", False))
                break
            obs = np.array(obs_l, dtype=np.float32)
            avail = np.array(avail_l)
        wins.append(1.0 if won else 0.0)
        lens.append(length)
        rets.append(ret)
        drops.append(drop_acc / max(1, length))
        fires.append(fire_acc / max(1, length))
        x2s.append(x2_acc / max(1, length))
        holds.append(n_held / max(1, n_wanted))
    fire_m, drop_m = float(np.mean(fires)), float(np.mean(drops))
    return dict(
        win=float(np.mean(wins)), ep_len=float(np.mean(lens)),
        ret=float(np.mean(rets)), drop=drop_m,
        fire=fire_m, x2=float(np.mean(x2s)), hold=float(np.mean(holds)),
        thru=fire_m * (1.0 - drop_m),
        ret_ci=float(1.96 * np.std(rets) / max(1.0, np.sqrt(len(rets)))),
        win_ci=float(1.96 * np.std(wins) / max(1.0, np.sqrt(len(wins)))),
    )


def main():
    p = argparse.ArgumentParser(description="SMAC-CWO Phase 1: certify sigma*")
    p.add_argument("--load_config", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--map_name", default=None)
    p.add_argument("--sigmas", default="0.4,0.7,1.0,1.4,2.0")
    p.add_argument("--betas", default="0,0.5,1.0,2.0,3.0,5.0",
                   help="hold gains. Needs to reach high: `target` is a proportional "
                        "law whose fixed point sits above f* for small beta, so a grid "
                        "stopping at 1.0 never actually reaches the coordinated load.")
    p.add_argument("--law", default="reaim", choices=("reaim",),
                   help="the channel inverse: pre-shift the target so the deflection "
                        "lands it where the policy aimed. beta=1 is exact.")
    p.add_argument("--bar", type=float, default=0.90,
                   help="fraction of B0 the compensated arm must reach. 0.90 is the "
                        "pipeline default; lower it if you are targeting a specific "
                        "absolute score. Read RECOVERY (fraction of the blind->B0 gap "
                        "closed) alongside it -- that is the scale-free number.")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--metric", default="ret", choices=("win", "ret"),
                   help="which quantity the 0.9*B0 bar is applied to. DEFAULT 'ret'. "
                        "Win rate is a poor frontier observable on a MIRROR-match map "
                        "(3s5z vs 3s5z): Lanchester's square law makes the outcome a "
                        "cliff in relative DPS, so it goes 0.6 -> 0.0 for a 27% shot "
                        "drop with NO partial band, and a 0.9*B0 WIN bar can then only "
                        "be met by a controller that restores essentially all the lost "
                        "damage -- which holding fire structurally cannot do, since it "
                        "buys damage with damage. Return degrades gracefully (18.2 -> "
                        "11.5 -> 3.4 on the same sweep) and has resolution.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    betas = [float(b) for b in args.betas.split(",") if b.strip()]

    cfg = json.load(open(args.load_config, encoding="utf-8"))
    algo, algo_args = cfg["main_args"]["algo"], cfg["algo_args"]
    assert cfg["main_args"]["env"] == "smac"
    env_args = dict(cfg["env_args"])
    if args.map_name:
        env_args["map_name"] = args.map_name
    # B0 is a BLIND checkpoint: stock obs, no augmentation, or the shapes will not load.
    env_args["snd_oracle"] = 0
    env_args["snd_pact"] = 0
    # A PROBE ENV MUST BYPASS THE WARMUP CURRICULUM.  _curr_severity() ramps the applied
    # severity from 0 over the first _WARMUP (=500k) steps of an env's life and this
    # script runs a few thousand, so without snd_eval every cell would silently execute
    # at severity 0 and the whole sweep would be vacuous.
    env_args["snd_eval"] = 1
    env_args["snd_phase"] = 0.0   # the probe freezes the driver anyway; keep it explicit
    algo_args["train"]["model_dir"] = args.model_dir

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    rec_n = algo_args["model"]["recurrent_n"]
    hidden = algo_args["model"]["hidden_sizes"][-1]

    from harl.envs.smac import StarCraft2_Env as CWO
    from harl.envs.smac.StarCraft2_Env import StarCraft2Env

    env = StarCraft2Env(env_args)
    env.seed(args.seed)
    actors = build_actors(algo, algo_args, env, dev)
    n = env.n_agents
    n_no_attack = int(env.n_actions_no_attack)
    knee = float(CWO._KNEE)

    def cell(sev, frz, beta):
        return eval_cell(env, actors, n, rec_n, hidden, args.episodes, sev, frz,
                         beta, n_no_attack)

    key = args.metric
    print(f"[phase1] map={env_args['map_name']} law={args.law} eps/cell={args.episodes} "
          f"knee={knee} lmax={CWO._LMAX} metric={key}\n")

    # ---- B0 + the two mandatory confounding checks (pipeline II.4) -------------
    b0 = cell(0.0, None, 0.0)
    bar_frac = float(args.bar)
    bar = bar_frac * b0[key]
    print(f"B0 (severity 0, no shim): win={b0['win']:.3f}+-{b0['win_ci']:.2f} "
          f"ret={b0['ret']:.1f}+-{b0['ret_ci']:.1f} ep_len={b0['ep_len']:.1f} "
          f"fire={b0['fire']:.3f} x2={b0['x2']:.3f}")
    print(f"BAR = {bar_frac:.2f} * B0[{key}] = {bar:.3f}")
    if b0["win"] < 0.85:
        print(f"[phase1] *** WARNING: B0 wins only {b0['win']:.2f}. A frontier measured "
              f"against a marginal baseline is not a frontier -- a policy with no margin "
              f"loses to ANY handicap, so sigma* collapses to ~0 for reasons that have "
              f"nothing to do with the NS. Train B0 to a strong stationary win rate "
              f"(HAPPO reaches ~0.95 on 3s5z) before reading anything below.")
    print(f"[phase1] measured greedy load x2={b0['x2']:.3f} at fire={b0['fire']:.3f} -- "
          f"set _GREEDY_LOAD in StarCraft2_Env.py to this, and size _KNEE against it "
          f"(_KNEE sets the CEILING: the coordinated team keeps throughput ~_KNEE).\n")

    trans = cell(0.0, 0.0, max(betas) if betas else 1.0)
    lo_sigma = min([s for s in sigmas if s > 0.0], default=0.4)
    blind_lo = cell(lo_sigma, 1.0, 0.0)
    best_lo = max((cell(lo_sigma, 1.0, b) for b in betas if b > 0),
                  key=lambda r: r[key], default=blind_lo)

    print("CHECK A  transparency (severity 0, shim ON): "
          f"{trans[key]:.3f} vs B0 {b0[key]:.3f} -> "
          + ("PASS (the shim is inert with no NS)"
             if abs(trans[key] - b0[key]) <= max(0.05, 0.15 * abs(b0[key]))
             else "FAIL -- the shim corrupts the action independently of the NS"))
    print(f"CHECK B  works-when-it-should (sigma={lo_sigma}, peak): blind "
          f"{blind_lo[key]:.3f} -> compensated {best_lo[key]:.3f} -> "
          + ("PASS (compensation recovers where it should)"
             if best_lo[key] > blind_lo[key] + 1e-9
             else "FAIL -- compensation NEVER helps; the law or the signal is wrong, "
                  "or the NS has no coordination solution at all (check the env banner's "
                  "headroom line)"))
    print()

    # ---- the sweep: EVERY cell is printed, then max over beta PER sigma --------
    # Printing only the argmax row is useless exactly when the sweep fails: if the
    # metric is 0 in every cell, argmax returns the FIRST cell, which is beta=0 --
    # i.e. the blind row -- so the table silently reports uncompensated numbers for
    # every severity (held=0.000, fire unchanged from B0).  Show the whole grid.
    hdr = (f"{'sigma':>6} {'beta':>6} | {'win':>6} {'+-':>5} {'ret':>8} {'+-':>6} "
           f"{'ep_len':>7} | {'x2':>6} {'defl':>6} {'fire':>6} {'thru':>6} {'reaim':>6}")
    print(hdr)
    print("-" * len(hdr))
    sigma_star = None
    summary = []
    for sig in sigmas:
        results = [(b, cell(sig, 1.0, b)) for b in betas]   # driver FROZEN AT PEAK
        for b, r in results:
            print(f"{sig:>6.2f} {b:>6.2f} | {r['win']:>6.3f} {r['win_ci']:>5.2f} "
                  f"{r['ret']:>8.1f} {r['ret_ci']:>6.1f} {r['ep_len']:>7.1f} | "
                  f"{r['x2']:>6.3f} {r['drop']:>6.3f} {r['fire']:>6.3f} "
                  f"{r['thru']:>6.3f} {r['hold']:>6.3f}")
        best_b, r = max(results, key=lambda kv: kv[1][key])
        blind = dict(results)[0.0] if 0.0 in dict(results) else results[0][1]
        ok = r[key] >= bar
        if ok:
            sigma_star = sig
        # RECOVERY is the number that matters for a permutation channel: how much of
        # the blind->B0 gap the scripted inverse closes.  1.0 means conjugacy holds
        # exactly at this severity (re-aiming restores the stationary game).
        gap = b0[key] - blind[key]
        rec = (r[key] - blind[key]) / gap if abs(gap) > 1e-9 else float("nan")
        summary.append((sig, best_b, r, ok, blind, rec))
        verdict = "CLEARS" if ok else "below"
        print(f"{'':>6} {'BEST':>6} | beta={best_b:.2f}  {key}={r[key]:.3f} "
              f"(blind {blind[key]:.3f} -> B0 {b0[key]:.3f});  RECOVERY "
              f"{rec:.0%} of the gap;  bar {bar:.3f} -> {verdict}")
        print("-" * len(hdr))
    env.close()

    print("\n[phase1] VERDICT:")
    print(f"  {'sigma':>6} {'blind':>9} {'best':>9} {'B0':>9} {'recovery':>9} {'>=bar':>6}")
    for sig, bb, r, ok, blind, rec in summary:
        print(f"  {sig:>6.2f} {blind[key]:>9.3f} {r[key]:>9.3f} {b0[key]:>9.3f} "
              f"{rec:>8.0%} {('YES' if ok else 'no'):>6}")
    good = [s for s, _b, _r, _ok, _bl, rec in summary if np.isfinite(rec) and rec >= 0.75]
    if sigma_star is not None:
        print(f"\n  sigma* = {sigma_star} on '{key}' at the {bar_frac:.0%} bar. "
              f"WELL-POSED at or below it -- run Phase 2 at sigma = "
              f"{max(0.0, sigma_star - 0.05):.2f}.")
    elif good:
        print(f"\n  Nothing clears the {bar_frac:.0%} bar, BUT the scripted inverse "
              f"recovers >=75% of the blind->B0 gap at sigma={good}. For a permutation "
              f"channel that is the meaningful certificate: conjugacy is close to "
              f"holding, and the shortfall is the coarseness of a discrete re-aim (the "
              f"shift is an integer over K attackable enemies), not a missing solution. "
              f"Pick the largest such sigma, or lower the bar with --bar if you are "
              f"targeting a specific absolute score rather than 0.9*B0.")
    else:
        print(f"\n  No sigma recovers even 75% of the gap.  Check first that the "
              f"channel is really invertible here (CHECK B should show a large jump) "
              f"and that beta=1 is in --betas -- beta=1 IS the exact inverse, so if it "
              f"does not recover, the shim and the env disagree about the shift. Only "
              f"then redesign: lower sigma, or cap the driver.")


if __name__ == "__main__":
    main()
