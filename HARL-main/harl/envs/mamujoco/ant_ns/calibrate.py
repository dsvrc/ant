"""The sigma ladder against a COMPETENT controller: a trained B0 policy.

Porting brief, traps 1 and 2: *"Calibrate sigma against a competent controller,
never random actions... If no competent scripted controller exists, you cannot
calibrate offline -- train B0 first and say so."*  Ant ships no scripted gait and
a random policy scores about -60, where a disturbance has nothing to take away.
So this script loads the checkpoint of a stock ``--ns_on 0`` run (B0) and
evaluates it, FROZEN, inside the dial at each sigma, in every arm:

    blind      the B0 policy inside the dial, no compensator
    pactoff    same, estimator running, trust 0     (must equal blind exactly)
    pact       same, fixed trust                     (the method, no retraining)
    oracle     same, handed the true disturbance     (the ceiling: B0 itself)
    intercept  same, peer channels deleted           (the (B)/(C) probe)

Pick the OPERATING POINT from the table -- the smallest sigma at which the blind
arm has lost a clear margin while the oracle still recovers B0 -- and commit it
to ``mamujoco_ns.yaml`` with the table.  Every sigma > 1 is a beyond-physical
stress test and is labelled so in the output.

    python -m harl.envs.mamujoco.ant_ns.calibrate \\
        --run_dir results/mamujoco_ns/Ant-v2-4x2/happo/b0/seed-00001-<stamp> \\
        --sigmas 0,0.5,1,2,3 --episodes 20 --threads 10 --phase peak --out ladder.json

``--phase peak`` pins every eval env's clock at the driver's maximum (the
operating point is about the peak); ``--phase cycle`` de-phases the threads
across the cycle and reports the cycle average, which is what training sees.

MuJoCo is deterministic given a seed, so every arm faces the identical initial
states and the ladder is a paired comparison.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from harl.algorithms.actors import ALGO_REGISTRY
from harl.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from harl.utils.trans_tools import _t2n

from . import make_ant_ns_env
from .driver import DialParams

ARMS = {
    "blind": dict(ns_pact=0),
    "pactoff": dict(ns_pact=1, ns_trust="off"),
    "pact": dict(ns_pact=1, ns_trust="fixed"),
    "oracle": dict(ns_pact=1, ns_trust="fixed", ns_oracle=1, ns_g_fixed=1.0),
    "intercept": dict(ns_pact=1, ns_trust="fixed", ns_intercept_only=1),
}

_MEAN_KEYS = ("ns_d", "ns_fit_gain", "ns_beta_cos", "ns_pred_err", "ns_trust", "ns_conf",
              "ns_x_std", "ns_tau_rms", "reward_forward", "reward_ctrl",
              "reward_contact", "reward_survive")
_COUNT_KEYS = ("ns_dial_live", "ns_harmed", "ns_clipped", "ns_corr_clipped",
               "ns_diverged")


def load_run(run_dir):
    with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_dir = os.path.join(run_dir, "models")
    assert os.path.exists(os.path.join(model_dir, "actor_agent0.pt")), (
        "no actor checkpoint under %s" % model_dir)
    return cfg["main_args"]["algo"], cfg["algo_args"], cfg["env_args"], model_dir


def make_vec(env_args, n_threads, phase, seed, peak):
    fns = []
    for rank in range(n_threads):
        def fn(rank=rank):
            ea = dict(env_args)
            if phase == "peak":
                ea["ns_phase0"] = int(peak)
                env = make_ant_ns_env(ea, 0, 1)
            else:
                env = make_ant_ns_env(ea, rank, n_threads)
            env.seed(seed * 50000 + rank * 10000)
            return env
        fns.append(fn)
    return ShareSubprocVecEnv(fns) if n_threads > 1 else ShareDummyVecEnv(fns)


def build_actors(algo, algo_args, envs, n_agents, device, model_dir):
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
    hid = int(algo_args["model"]["hidden_sizes"][-1])
    rn = int(algo_args["model"]["recurrent_n"])
    obs, _, avail = envs.reset()
    rnn = np.zeros((n_threads, n_agents, rn, hid), dtype=np.float32)
    masks = np.ones((n_threads, n_agents, 1), dtype=np.float32)
    rets, lens, done_eps = [], [], 0
    ep_r = np.zeros(n_threads)
    ep_t = np.zeros(n_threads)
    acc = {k: [] for k in _MEAN_KEYS}
    counts = {k: np.zeros(n_threads) for k in _COUNT_KEYS}
    while done_eps < episodes:
        acts = []
        for i in range(n_agents):
            a, r = actors[i].act(obs[:, i], rnn[:, i], masks[:, i],
                                 avail[:, i] if avail[0] is not None else None,
                                 deterministic=True)
            rnn[:, i] = _t2n(r)
            acts.append(_t2n(a))
        actions = np.array(acts).transpose(1, 0, 2)
        obs, _, rew, dones, infos, avail = envs.step(actions)
        ep_r += np.asarray(rew, dtype=np.float64).reshape(n_threads, -1)[:, 0]
        ep_t += 1.0
        for t in range(n_threads):
            row = infos[t][0] if isinstance(infos[t], (list, tuple, np.ndarray)) else infos[t]
            if isinstance(row, dict):
                for k in _MEAN_KEYS:
                    if k in row and np.isfinite(row[k]):
                        acc[k].append(float(row[k]))
                for k in _COUNT_KEYS:
                    if k in row:
                        counts[k][t] = max(counts[k][t], float(row[k]))
        denv = np.all(dones, axis=1)
        rnn[denv] = 0.0
        masks = np.ones((n_threads, n_agents, 1), dtype=np.float32)
        masks[denv] = 0.0
        for t in range(n_threads):
            if denv[t]:
                rets.append(float(ep_r[t]))
                lens.append(float(ep_t[t]))
                ep_r[t] = ep_t[t] = 0.0
                done_eps += 1
    out = dict(ret=float(np.mean(rets)),
               ret_se=float(np.std(rets) / max(np.sqrt(len(rets)), 1.0)),
               ep_len=float(np.mean(lens)), episodes=len(rets))
    for k in _MEAN_KEYS:
        out[k.replace("ns_", "")] = float(np.mean(acc[k])) if acc[k] else float("nan")
    for k in _COUNT_KEYS:
        out[k.replace("ns_", "")] = float(counts[k].sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="a B0 run dir (config.json, models/)")
    ap.add_argument("--sigmas", default="0,0.5,1,2,3")
    ap.add_argument("--arms", default="blind,pactoff,pact,oracle,intercept")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--phase", choices=["peak", "cycle"], default="peak")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cuda", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    algo, algo_args, env_args, model_dir = load_run(a.run_dir)
    env_args = dict(env_args)
    env_args["ns_on"] = 1
    p = DialParams(period=int(env_args.get("ns_period", 20000)),
                   warm_fraction=float(env_args.get("ns_warm_fraction", 0.5)))
    peak = int(p.period * p.warm_fraction / 2)
    device = torch.device("cuda" if a.cuda and torch.cuda.is_available() else "cpu")
    sigmas = [float(s) for s in a.sigmas.split(",")]
    arms = [s.strip() for s in a.arms.split(",")]
    print("[calibrate] B0 policy: %s from %s | partition %s | phase=%s | %d episodes "
          "x %d threads" % (algo, model_dir, env_args.get("agent_conf"), a.phase,
                            a.episodes, a.threads), flush=True)

    table = {}
    actors = None
    n_agents = None
    for s in sigmas:
        for arm in arms:
            ea = dict(env_args)
            ea["ns_severity"] = float(s)
            ea.update(ARMS[arm])
            t0 = time.time()
            envs = make_vec(ea, a.threads, a.phase, a.seed, peak)
            if actors is None:
                n_agents = int(envs.n_agents)
                actors = build_actors(algo, algo_args, envs, n_agents, device, model_dir)
            r = evaluate(actors, envs, n_agents, a.threads, a.episodes, algo_args)
            envs.close()
            table[(s, arm)] = r
            print("  sigma=%-4g %-9s ret=%8.1f+-%5.1f len=%6.1f | d=%.3f dial=%.0f "
                  "harmed=%.0f clip=%.0f corrclip=%.0f | fit=%.3f bcos=%.3f "
                  "pred_err=%.4f trust=%.2f | fwd=%.3f ctrl=%.3f | %.0fs"
                  % (s, arm, r["ret"], r["ret_se"], r["ep_len"], r["d"], r["dial_live"],
                     r["harmed"], r["clipped"], r["corr_clipped"], r["fit_gain"],
                     r["beta_cos"], r["pred_err"], r["trust"], r["reward_forward"],
                     r["reward_ctrl"], time.time() - t0), flush=True)

    # ---- the ladder, and the recommendation ---------------------------------
    print("")
    print("THE LADDER (return of the FROZEN B0 policy; sigma > 1 is BEYOND-PHYSICAL)")
    print("  %-6s" % "sigma" + "".join("%12s" % arm for arm in arms)
          + "%14s" % "corr_clipped")
    b0 = table[(sigmas[0], "blind")]["ret"] if "blind" in arms else float("nan")
    rec = None
    for s in sigmas:
        row = "  %-6g" % s + "".join("%12.1f" % table[(s, arm)]["ret"] for arm in arms)
        if "oracle" in arms:
            row += "%14.0f" % table[(s, "oracle")]["corr_clipped"]
        if s > 1.0:
            row += "   <- beyond-physical"
        print(row)
        if rec is None and s > 0 and "blind" in arms and "oracle" in arms:
            if (table[(s, "blind")]["ret"] <= 0.75 * b0
                    and table[(s, "oracle")]["ret"] >= 0.95 * b0):
                rec = s
    if "blind" in arms and "pactoff" in arms:
        same = all(abs(table[(s, "blind")]["ret"] - table[(s, "pactoff")]["ret"]) < 1e-6
                   for s in sigmas)
        print("")
        print("  pactoff == blind at every sigma: %s%s"
              % (same, "" if same else "   <-- NOT IDENTICAL: the floor property is broken"))
    if "pact" in arms and "intercept" in arms:
        print("  the (C) signature -- deleting the peer channels must COST:")
        for s in sigmas:
            if s == 0:
                continue
            print("    sigma=%-4g pact=%8.1f  intercept=%8.1f  (%s)"
                  % (s, table[(s, "pact")]["ret"], table[(s, "intercept")]["ret"],
                     "peer channels load-bearing"
                     if table[(s, "pact")]["ret"] > table[(s, "intercept")]["ret"]
                     else "they are NOT -- report that"))
    print("")
    if rec is not None:
        print("  RECOMMENDED OPERATING POINT: sigma = %g  (blind fell below 75%% of "
              "B0 = %.1f while the oracle held above 95%%).  Commit it to "
              "mamujoco_ns.yaml with this table." % (rec, b0))
    else:
        print("  No sigma in the ladder satisfies (blind <= 0.75*B0 and oracle >= "
              "0.95*B0); extend --sigmas upward, or accept a smaller margin and say "
              "so in the paper.")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"%g:%s" % k: v for k, v in table.items()}, f, indent=2)
        print("  wrote %s" % a.out)


if __name__ == "__main__":
    main()
