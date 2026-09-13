"""The sigma ladder against a COMPETENT controller: a trained B0 policy, frozen.

StarCraft II ships no scripted controller that kites like a trained policy, so
sigma cannot be calibrated offline.  Train B0 first (stock ``--algo mappo
--ns_on 0``), then evaluate that checkpoint inside the dial at each sigma, in
every arm:

    blind      B0 inside the dial, no compensator
    pactoff    estimator running, trust 0            (must equal blind)
    pact       compensated with the ESTIMATE, g=0.9   (the method, no retraining)
    oracle     compensated with the TRUE swerve       (the ceiling: B0 itself)
    intercept  peer channels deleted                  (the (B)/(C) probe)

The OPERATING POINT is the smallest sigma at which blind has lost >= 20 points
while the oracle is still within 5 of B0.  Commit it, with the table, to
smac.yaml.  sigma > 1 is beyond-physical and printed so.

    python -m harl.envs.smac.smac_lane.calibrate \\
        --run_dir results/smac/3s_vs_5z/mappo/b0/seed-00001-<stamp> \\
        --sigmas 0,1,2,3,4,6 --episodes 64 --threads 8 --phase peak --out ladder.json

``--phase peak`` pins every env's driver at its maximum; ``--phase cycle``
de-phases threads across the cycle (what training sees).
"""

import argparse
import json
import os
import time

import numpy as np
import torch

ARMS = {
    "blind": dict(ns_pact=0),
    "pactoff": dict(ns_pact=1, ns_g_fixed=0.0),
    "pact": dict(ns_pact=1, ns_g_fixed=0.9),
    "oracle": dict(ns_pact=1, ns_oracle=1),
    "intercept": dict(ns_pact=1, ns_g_fixed=0.9, ns_intercept_only=1),
}
_MEAN_KEYS = ("ns_d_deg", "ns_exec_deg", "ns_lane_frac", "ns_beta_cos", "ns_fit_gain")


def load_run(run_dir):
    with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_dir = os.path.join(run_dir, "models")
    assert os.path.exists(os.path.join(model_dir, "actor_agent0.pt")), (
        "no actor checkpoint under %s" % model_dir)
    return cfg["main_args"]["algo"], cfg["algo_args"], cfg["env_args"], model_dir


def make_vec(env_args, n_threads, phase, seed):
    from harl.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
    from harl.utils.envs_tools import make_smac_env

    period = int(env_args["ns_period"])
    peak = int(period * float(env_args.get("ns_guard_frac", 0.5)) / 2)
    fns = []
    for rank in range(n_threads):
        def fn(rank=rank):
            ea = dict(env_args)
            s = seed * 50000 + rank * 10000       # the SAME stream in every arm
            if phase == "peak":
                ea["ns_phase0"] = peak
                return make_smac_env(ea, 0, 1, s)
            return make_smac_env(ea, rank, n_threads, s)
        fns.append(fn)
    return ShareSubprocVecEnv(fns) if n_threads > 1 else ShareDummyVecEnv(fns)


def build_actors(algo, algo_args, envs, n_agents, device, model_dir):
    from harl.algorithms.actors import ALGO_REGISTRY

    actors = []
    for i in range(n_agents):
        ac = ALGO_REGISTRY[algo]({**algo_args["model"], **algo_args["algo"]},
                                 envs.observation_space[i], envs.action_space[i],
                                 device=device)
        sd = torch.load(os.path.join(model_dir, "actor_agent%d.pt" % i),
                        map_location=device)
        ac.actor.load_state_dict(sd)
        ac.prep_rollout()
        actors.append(ac)
    return actors


@torch.no_grad()
def evaluate(actors, envs, n_agents, n_threads, episodes, algo_args):
    from harl.utils.trans_tools import _t2n

    hid = int(algo_args["model"]["hidden_sizes"][-1])
    rn = int(algo_args["model"]["recurrent_n"])
    obs, _, avail = envs.reset()
    rnn = np.zeros((n_threads, n_agents, rn, hid), dtype=np.float32)
    masks = np.ones((n_threads, n_agents, 1), dtype=np.float32)
    wins, lens, done_eps = [], [], 0
    ep_t = np.zeros(n_threads)
    acc = {k: [] for k in _MEAN_KEYS}
    clipped = np.zeros(n_threads)
    while done_eps < episodes:
        acts = []
        for i in range(n_agents):
            a, r = actors[i].act(obs[:, i], rnn[:, i], masks[:, i], avail[:, i],
                                 deterministic=True)
            rnn[:, i] = _t2n(r)
            acts.append(_t2n(a))
        actions = np.array(acts).transpose(1, 0, 2)
        obs, _, rew, dones, infos, avail = envs.step(actions)
        ep_t += 1.0
        for t in range(n_threads):
            row = infos[t][0] if isinstance(infos[t], (list, tuple, np.ndarray)) else infos[t]
            if isinstance(row, dict):
                for k in _MEAN_KEYS:
                    if k in row and np.isfinite(row[k]):
                        acc[k].append(float(row[k]))
                clipped[t] = max(clipped[t], float(row.get("ns_corr_clipped", 0.0)))
        denv = np.all(dones, axis=1)
        rnn[denv] = 0.0
        masks = np.ones((n_threads, n_agents, 1), dtype=np.float32)
        masks[denv] = 0.0
        for t in range(n_threads):
            if denv[t]:
                row = infos[t][0] if isinstance(infos[t], (list, tuple, np.ndarray)) else infos[t]
                wins.append(1.0 if row.get("won", False) else 0.0)
                lens.append(float(ep_t[t]))
                ep_t[t] = 0.0
                done_eps += 1
    out = dict(win=float(np.mean(wins)), win_se=float(np.std(wins) / np.sqrt(len(wins))),
               ep_len=float(np.mean(lens)), episodes=len(wins),
               corr_clipped=float(clipped.sum()))
    for k in _MEAN_KEYS:
        out[k[3:]] = float(np.mean(acc[k])) if acc[k] else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="a B0 run dir (config.json, models/)")
    ap.add_argument("--sigmas", default="0,1,2,3,4,6")
    ap.add_argument("--arms", default="blind,pactoff,pact,oracle,intercept")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--phase", choices=["peak", "cycle"], default="peak")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cuda", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from harl.envs.smac.smac_maps import get_map_params

    algo, algo_args, env_args, model_dir = load_run(a.run_dir)
    env_args = dict(env_args)
    env_args["ns_on"] = 1
    env_args.setdefault("ns_period", 100 * int(get_map_params(env_args["map_name"])["limit"]))
    n_agents = int(get_map_params(env_args["map_name"])["n_agents"])
    device = torch.device("cuda" if a.cuda and torch.cuda.is_available() else "cpu")
    sigmas = [float(s) for s in a.sigmas.split(",")]
    arms = [s.strip() for s in a.arms.split(",")]
    print("[calibrate] B0 %s from %s | map %s | phase %s | %d episodes x %d threads"
          % (algo, model_dir, env_args["map_name"], a.phase, a.episodes, a.threads),
          flush=True)

    table, actors = {}, None
    for s in sigmas:
        for arm in arms:
            ea = dict(env_args)
            ea["ns_severity"] = float(s)
            ea.update(dict(ns_pact=0, ns_g_fixed=0.9, ns_oracle=0, ns_intercept_only=0))
            ea.update(ARMS[arm])
            t0 = time.time()
            envs = make_vec(ea, a.threads, a.phase, a.seed)
            if actors is None:
                actors = build_actors(algo, algo_args, envs, n_agents, device, model_dir)
            r = evaluate(actors, envs, n_agents, a.threads, a.episodes, algo_args)
            envs.close()
            table[(s, arm)] = r
            print("  sigma=%-4g %-9s win=%.3f+-%.3f len=%5.1f | design %.1f deg, executed "
                  "%.1f deg, lane_frac %.2f, clipped %.0f | beta_cos %.3f | %.0fs"
                  % (s, arm, r["win"], r["win_se"], r["ep_len"], r["d_deg"], r["exec_deg"],
                     r["lane_frac"], r["corr_clipped"], r["beta_cos"], time.time() - t0),
                  flush=True)

    print("\nTHE LADDER (win rate of the frozen B0 policy; sigma > 1 is BEYOND-PHYSICAL)")
    print("  %-6s" % "sigma" + "".join("%11s" % arm for arm in arms))
    b0 = table[(sigmas[0], "blind")]["win"] if "blind" in arms else float("nan")
    rec = None
    for s in sigmas:
        row = "  %-6g" % s + "".join("%11.3f" % table[(s, arm)]["win"] for arm in arms)
        print(row + ("   <- beyond-physical" if s > 1.0 else ""))
        if rec is None and s > 0 and "blind" in arms and "oracle" in arms:
            if table[(s, "blind")]["win"] <= b0 - 0.20 \
                    and table[(s, "oracle")]["win"] >= b0 - 0.05:
                rec = s
    if "blind" in arms and "pactoff" in arms:
        same = all(abs(table[(s, "blind")]["win"] - table[(s, "pactoff")]["win"]) < 1e-9
                   for s in sigmas)
        print("\n  pactoff == blind at every sigma: %s" % same)
    if rec is not None:
        print("\n  RECOMMENDED OPERATING POINT: sigma = %g (blind lost >= 20 points, oracle "
              "within 5 of B0 = %.3f).  Commit it to smac.yaml with this table." % (rec, b0))
    else:
        print("\n  No sigma satisfies (blind <= B0-0.20, oracle >= B0-0.05): extend "
              "--sigmas upward, or report the smaller margin as it is.")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"%g:%s" % k: v for k, v in table.items()}, f, indent=2)
        print("  wrote %s" % a.out)


if __name__ == "__main__":
    main()
