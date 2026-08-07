"""Decisive CWO debugging experiment -- does a KNOWN-GOOD policy still WIN through CWO?

Loads a trained checkpoint (your B0, which wins ~0.67 on stationary 3s5z) and evaluates
it -- NO training -- through the CWO env across a grid of (severity, frozen driver A),
reporting win rate, episode length, drop rate and return per cell.  This separates the
three hypotheses cleanly and runs in minutes:

  (A) severity=0                     -> should reproduce B0 (~0.67, ep_len ~40).
  (B) severity>0, A=0 (frozen trough)-> ell=0 => NO drops => must EQUAL (A) exactly.
        If win collapses here, a CODE BUG breaks the game even with zero drops.
  (C) severity>0, A=1 (frozen peak)  -> ell=peak, ~1/3 shots dropped.
        win still decent  -> CWO only slows a good policy; the 0-win training result is
                             a from-scratch TRAINING-STEERING trap (fix: warm-start B0,
                             or sparse reward so damage-farming isn't rewarded).
        win ~0            -> CWO makes winning impossible for ANY policy (the NS itself
                             must change -- it converts wins to timeouts, not "less DPS").
  (D) severity>0, live A               -> the training-time condition, for reference.

    python -m harl.envs.smac.pact.diagnose \
        --load_config results/.../<B0 run>/config.json \
        --model_dir   results/.../<B0 run>/models  --episodes 40
"""

import argparse
import json

import numpy as np
import torch

from harl.algorithms.actors import ALGO_REGISTRY
from harl.utils.trans_tools import _t2n


def build_actors(algo, algo_args, env, device):
    share = algo_args["algo"]["share_param"]
    merged = {**algo_args["model"], **algo_args["algo"]}
    n = env.n_agents
    if share:
        ag = ALGO_REGISTRY[algo](merged, env.observation_space[0], env.action_space[0], device=device)
        actors = [ag] * n
    else:
        actors = [ALGO_REGISTRY[algo](merged, env.observation_space[i], env.action_space[i], device=device)
                  for i in range(n)]
    md = algo_args["train"]["model_dir"]
    for i in range(n):
        actors[i].actor.load_state_dict(torch.load(f"{md}/actor_agent{i}.pt", map_location=device))
        actors[i].prep_rollout()
    return actors


@torch.no_grad()
def eval_cell(env, actors, n, rec_n, hidden, n_eps, severity, freeze):
    env.snd_severity = float(severity)          # override the resolved knobs per cell
    env.snd_freeze = freeze                      # None (live) or a fixed A in [0,1]
    wins, lens, rets, drops = [], [], [], []
    for _ in range(n_eps):
        obs, _st, avail = env.reset()
        obs = np.array(obs, dtype=np.float32); avail = np.array(avail)
        rnn = np.zeros((n, rec_n, hidden), dtype=np.float32)
        masks = np.ones((n, 1), dtype=np.float32)
        ret = length = 0.0; won = False; drop_acc = 0.0
        while True:
            acts = []
            for i in range(n):
                a, r = actors[i].act(obs[i:i+1], rnn[i:i+1], masks[i:i+1], avail[i:i+1], deterministic=True)
                rnn[i] = _t2n(r)[0]; acts.append(int(_t2n(a)[0, 0]))
            obs_l, _s, rew, dones, infos, avail_l = env.step(np.array(acts, dtype=np.int64))
            ret += float(np.asarray(rew[0]).reshape(-1)[0]); length += 1
            drop_acc += float(infos[0].get("cwo_drop_frac", 0.0))
            if bool(np.asarray(dones).reshape(-1)[0]):
                won = bool(infos[0].get("won", False)); break
            obs = np.array(obs_l, dtype=np.float32); avail = np.array(avail_l)
        wins.append(1.0 if won else 0.0); lens.append(length)
        rets.append(ret); drops.append(drop_acc / max(1, length))
    return dict(win=np.mean(wins), ep_len=np.mean(lens), ret=np.mean(rets), drop=np.mean(drops))


def main():
    p = argparse.ArgumentParser(description="CWO decisive debugging experiment")
    p.add_argument("--load_config", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--map_name", default=None)
    p.add_argument("--severity", type=float, default=2.0)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = json.load(open(args.load_config, encoding="utf-8"))
    algo, algo_args = cfg["main_args"]["algo"], cfg["algo_args"]
    assert cfg["main_args"]["env"] == "smac"
    env_args = dict(cfg["env_args"])
    if args.map_name:
        env_args["map_name"] = args.map_name
    env_args["snd_oracle"] = 0; env_args["snd_pact"] = 0  # B0 was trained BLIND (stock obs)
    # A PROBE ENV MUST BYPASS THE WARMUP CURRICULUM.  _curr_severity() ramps the
    # applied severity from 0 over the first _WARMUP (=500k) steps of an env's life,
    # and this script runs a few thousand steps -- so without snd_eval the cells below
    # all execute at severity 0 no matter what --severity says, and every verdict is
    # vacuous.  snd_eval=1 is exactly the "use the full severity" flag.
    env_args["snd_eval"] = 1
    algo_args["train"]["model_dir"] = args.model_dir

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    rec_n = algo_args["model"]["recurrent_n"]; hidden = algo_args["model"]["hidden_sizes"][-1]

    from harl.envs.smac.StarCraft2_Env import StarCraft2Env
    env = StarCraft2Env(env_args); env.seed(args.seed)
    actors = build_actors(algo, algo_args, env, dev)
    n = env.n_agents
    print(f"[diag] map={env_args['map_name']} algo={algo} n={n} eps/cell={args.episodes} "
          f"severity={args.severity}\n")

    s = args.severity
    grid = [
        ("A) severity=0 (stock)",        0.0, None),
        ("B) severity>0, A=0 (no drops)", s,   0.0),
        ("C) severity>0, A=1 (peak)",     s,   1.0),
        ("D) severity>0, live A",         s,   None),
    ]
    print(f"{'cell':<32} {'win':>6} {'ep_len':>7} {'drop':>6} {'return':>7}")
    print("-" * 62)
    rows = {}
    for name, sev, fA in grid:
        r = eval_cell(env, actors, n, rec_n, hidden, args.episodes, sev, fA)
        rows[name[0]] = r
        print(f"{name:<32} {r['win']:>6.3f} {r['ep_len']:>7.1f} {r['drop']:>6.3f} {r['ret']:>7.1f}")
    env.close()

    print("\n[diag] VERDICT:")
    if rows["B"]["win"] < rows["A"]["win"] - 0.15:
        print("  B << A: a CODE BUG breaks the game at severity>0 even with ZERO drops "
              "(A=0 => ell=0 => should equal stock). Investigate the env, not the NS.")
    elif rows["C"]["win"] > 0.2:
        print("  B == A and C still wins: NO bug and the NS does NOT prevent winning. The "
              "0-win TRAINING result is a from-scratch steering trap -> WARM-START from B0 "
              "(or train with reward_sparse=true so damage-farming isn't rewarded).")
    else:
        print("  B == A but C ~0: no code bug, but CWO converts a good policy's WINS into "
              "TIMEOUTS at this severity -> lower severity, and/or the NS must change so it "
              "throttles winning-capability gracefully rather than blocking the finish.")


if __name__ == "__main__":
    main()
