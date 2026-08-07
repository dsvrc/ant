"""F4 — the controlled interference probe  [campaign spec §5 / H-C2]. 0.2M steps.

Protocol (verbatim from the spec):

    load F0; collect a 512-state trough probe bank; train 200k steps at
    FREEZE_A=1 (plain continuation, blind); every 10k steps log probe action-MSE
    vs the F0 policy and a 5-episode FREEZE_A=0 return.

Output: **the forgetting curve** — how fast peak training destroys trough
competence. That curve *is* the retention budget any future method must cover,
and it is what turns leaf L3 ("ride the ramp + explicit retention") from a slogan
into a sized engineering requirement. Axis **V7**.

**F4b** (P2): the same run with 25% of every minibatch drawn from a frozen buffer
of F0 **trough** transitions. It isolates what *pure rehearsal* — ECL's anchor,
with the identifier, localizer and envelope all removed — buys against that curve.
This is the cleanest read the campaign gets on whether ECL's [A] component was
ever doing anything, so it is run as two honest phases rather than approximated:

    # phase 1: collect a real trough pool with the F0 policy at A=0
    ANT_PCR_FREEZE_A=0.0 python scripts/diag_f4.py --config .../f4.json \\
        --model_dir <F0>/models --collect_pool diag_out/f4/trough_pool.npz \\
        --pool_steps 200000
    # phase 2: peak training with 25% trough rehearsal
    ANT_PCR_FREEZE_A=1.0 python scripts/diag_f4.py --config .../f4.json \\
        --model_dir <F0>/models --rehearse_pool diag_out/f4/trough_pool.npz \\
        --rehearse 0.25 --exp_name diag_f4b

Two phases because the env var is read at import, once, per process: the
collection envs must exist at A=0 and the training envs at A=1, and they cannot
be the same processes.

Plain F4 (no rehearsal) is one process:

    ANT_PCR_FREEZE_A=1.0 python scripts/diag_f4.py \\
        --config tuned_configs/mamujoco/Ant-v2-4x2/diag/f4.json \\
        --model_dir <F0>/models --exp_name diag_f4 --seed 1

Why the A=0 scoring works at all
--------------------------------
F4 trains at FREEZE_A=1 while **evaluating at FREEZE_A=0** — two live envs at two
different freezes. ``ant_diag`` snapshots its knobs per env instance, so the
training envs (built in subprocesses from the env var, at A=1) are unaffected when
this script flips the *deployed module's* knob to build its own in-process eval
env at A=0. The flip goes through ``diag.knobs``, which resolves the module gym
actually instantiates — flipping the repo's ``ant_diag`` copy would silently do
nothing.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harl.envs.mamujoco.diag.report_io import DebugReport, write_csv  # noqa: E402
from harl.envs.mamujoco.diag import knobs as K  # noqa: E402

_COLS = ["env_step", "probe_action_mse", "trough_return_5ep", "trough_return_std",
         "train_reward_mean", "rehearse_frac"]


# ==========================================================================
#  the trough pool (F4b)
# ==========================================================================
def dump_pool(buf, path, n_agents):
    """Freeze the buffer's occupied slots as an F0-trough rehearsal pool."""
    n = int(buf.cur_size)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    d = {"n": n, "n_agents": n_agents,
         "share_obs": buf.share_obs[:n], "next_share_obs": buf.next_share_obs[:n],
         "rewards": buf.rewards[:n], "dones": buf.dones[:n], "terms": buf.terms[:n]}
    for i in range(n_agents):
        d[f"obs{i}"] = buf.obs[i][:n]
        d[f"next_obs{i}"] = buf.next_obs[i][:n]
        d[f"actions{i}"] = buf.actions[i][:n]
        d[f"valid{i}"] = buf.valid_transitions[i][:n]
    np.savez_compressed(path, **d)
    return n


def load_pool(buf, path, n_agents):
    """Write the pool into the buffer's OLDEST slots and return its slot range.

    Placing it at [0, n) — before any peak data — means the n-step walk
    (``next()``, which strides by n_rollout_threads and stops at ``end_flag``)
    stays inside pool episodes, so a rehearsed transition's n-step target is built
    from trough data only. That is the point: rehearsal must replay the trough, not
    a trough state stitched onto a peak future.
    """
    z = np.load(path, allow_pickle=False)
    n = int(z["n"])
    assert n <= buf.buffer_size, f"pool ({n}) exceeds buffer_size ({buf.buffer_size})"
    buf.share_obs[:n] = z["share_obs"]
    buf.next_share_obs[:n] = z["next_share_obs"]
    buf.rewards[:n] = z["rewards"]
    buf.dones[:n] = z["dones"]
    buf.terms[:n] = z["terms"]
    for i in range(n_agents):
        buf.obs[i][:n] = z[f"obs{i}"]
        buf.next_obs[i][:n] = z[f"next_obs{i}"]
        buf.actions[i][:n] = z[f"actions{i}"]
        buf.valid_transitions[i][:n] = z[f"valid{i}"]
    buf.idx = n % buf.buffer_size
    buf.cur_size = max(buf.cur_size, n)
    if hasattr(buf, "payload_diag"):
        buf.payload_diag[:n] = 0.0        # trough
    return n


def install_rehearsal(buf, pool_n, frac):
    """Force ``frac`` of every minibatch to come from the pool slots [0, pool_n).

    This IS replay shaping — deliberately, inside a labeled diagnostic arm. It is
    the one place the campaign touches the sampler, and it exists precisely to
    measure what ECL's anchor bought in isolation (spec §5, F4b). It is never on
    for any other arm.
    """
    base_sample = buf.sample

    def sample():
        out = list(base_sample())
        n = out[4].shape[0]                       # batch size, from sp_reward
        k = int(round(frac * n))
        if k <= 0 or pool_n <= 0:
            return tuple(out)
        idx = np.random.randint(0, pool_n, size=k)
        pool = buf.gather(idx)
        for j, (a, b) in enumerate(zip(out, pool)):
            if a is None or b is None:
                continue
            a = np.array(a, copy=True)
            if a.ndim >= 2 and a.shape[0] == buf.num_agents and j in (1, 2, 6, 9):
                a[:, -k:] = b[:, :k]              # (n_agents, batch, dim)
            else:
                a[-k:] = b[:k]                    # (batch, dim)
            out[j] = a
        return tuple(out)

    buf.sample = sample


# ==========================================================================
@torch.no_grad()
def probe_mse(actors, ref_actions, bank):
    """Mean squared action deviation from the F0 policy on the frozen bank."""
    tot = []
    for i, a in enumerate(actors):
        cur = a.get_actions(bank[i], stochastic=False).cpu().numpy()
        tot.append(float(np.mean((cur - ref_actions[i]) ** 2)))
    return float(np.mean(tot))


def make_trough_env(all_cfg, out_dir, seed=0, restore_a=1.0):
    """Build ONE in-process env pinned at FREEZE_A=0 and keep it for the whole run.

    This is the payoff of ``ant_diag`` snapshotting its knobs per instance: the
    env captures A=0 at construction and holds it forever, so the knob can be put
    straight back to the training slice and never touched again. No flipping
    inside the loop, no chance of a stray A=0 leaking into training, and one
    recorder CSV instead of one per log point.
    """
    from scripts.diag_tier0 import make_env

    K.apply(freeze_a=0.0, mask="both", dcap=None)
    env = make_env(all_cfg, out_dir, "f4_trough_eval")
    env.seed(seed)
    K.apply(freeze_a=restore_a, mask="both", dcap=None)   # back to the training slice
    return env


@torch.no_grad()
def collect_bank(env, actors, n_states, seed=0):
    """The 512-state TROUGH probe bank: states F0 actually visits at A=0, under
    F0's own deterministic policy.

    Not the replay buffer's warmup states — those are collected with RANDOM
    actions in the PEAK env, so they are neither trough states nor on-policy, and
    a drift measured on them would be measuring the wrong thing.
    """
    banks = [[] for _ in range(env.n_agents)]
    obs, _, _ = env.reset()
    while len(banks[0]) < n_states:
        for i in range(env.n_agents):
            banks[i].append(np.asarray(obs[i], dtype=np.float32))
        acts = [actors[i].get_actions(np.asarray(obs[i])[None],
                                      stochastic=False).cpu().numpy()[0]
                for i in range(env.n_agents)]
        obs, _, _, dones, _, _ = env.step(np.asarray(acts))
        if np.all(dones):
            obs, _, _ = env.reset()
    return [np.asarray(b[:n_states]) for b in banks]


@torch.no_grad()
def trough_return(env, runner, episodes=5):
    """A 5-episode FREEZE_A=0 return on the pinned A=0 env."""
    rets = []
    for _ in range(episodes):
        obs, _, _ = env.reset()
        R, done = 0.0, False
        while not done:
            acts = [runner.actor[i].get_actions(np.asarray(obs[i])[None],
                                                stochastic=False).cpu().numpy()[0]
                    for i in range(env.n_agents)]
            obs, _, rewards, dones, _, _ = env.step(np.asarray(acts))
            R += float(np.mean(np.asarray(rewards)))
            done = bool(np.all(dones))
        rets.append(R)
    return float(np.mean(rets)), float(np.std(rets))


def build_runner(all_cfg, exp_name, seed, model_dir, steps=None):
    main_args = dict(all_cfg["main_args"])
    main_args["exp_name"] = exp_name
    algo_args = copy.deepcopy(all_cfg["algo_args"])
    env_args = copy.deepcopy(all_cfg["env_args"])
    algo_args["seed"]["seed"] = seed
    algo_args["train"]["model_dir"] = model_dir
    algo_args["eval"]["use_eval"] = False          # F4 does its own A=0 eval
    if steps is not None:
        algo_args["train"]["num_env_steps"] = int(steps)
    from harl.runners import RUNNER_REGISTRY

    return RUNNER_REGISTRY[main_args["algo"]](main_args, algo_args, env_args), algo_args


# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="F4 — the forgetting curve.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model_dir", required=True, help="F0's models/ dir")
    ap.add_argument("--exp_name", default="diag_f4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="./diag_out/f4")
    ap.add_argument("--log_every", type=int, default=10000, help="env steps")
    ap.add_argument("--bank_size", type=int, default=512)
    ap.add_argument("--eval_episodes", type=int, default=5)
    ap.add_argument("--collect_pool", default=None,
                    help="F4b phase 1: dump an F0-trough pool here and exit. "
                         "Launch with ANT_PCR_FREEZE_A=0.0.")
    ap.add_argument("--pool_steps", type=int, default=200000)
    ap.add_argument("--rehearse_pool", default=None, help="F4b phase 2: the pool")
    ap.add_argument("--rehearse", type=float, default=0.0,
                    help="F4b: fraction of every minibatch drawn from the pool "
                         "(spec: 0.25)")
    args = ap.parse_args(argv)

    K.require_deployed()
    with open(args.config, encoding="utf-8") as f:
        all_cfg = json.load(f)

    # ---- F4b phase 1: collect the trough pool ---------------------------
    if args.collect_pool:
        if os.environ.get("ANT_PCR_FREEZE_A") not in ("0", "0.0"):
            print("[F4] REFUSING: --collect_pool must run at ANT_PCR_FREEZE_A=0.0, "
                  "else the 'trough pool' is not trough data. "
                  f"Got {os.environ.get('ANT_PCR_FREEZE_A')!r}.", flush=True)
            return 2
        runner, aa = build_runner(all_cfg, args.exp_name + "_pool", args.seed,
                                  args.model_dir, steps=args.pool_steps)
        print(f"[F4] collecting {args.pool_steps} F0 trough transitions "
              f"(FREEZE_A=0, no gradient steps) ...", flush=True)
        obs, share_obs, avail = runner.envs.reset()
        n_thr = aa["train"]["n_rollout_threads"]
        for _ in range(args.pool_steps // n_thr):
            actions = runner.get_actions(obs, available_actions=avail,
                                         add_random=True)
            (new_obs, new_share_obs, rewards, dones, infos,
             new_avail) = runner.envs.step(actions)
            runner.insert((share_obs, obs.transpose(1, 0, 2),
                           actions.transpose(1, 0, 2), None, rewards, dones, infos,
                           new_share_obs.copy(), new_obs.copy(), None))
            obs, share_obs, avail = new_obs, new_share_obs, new_avail
        n = dump_pool(runner.buffer, args.collect_pool, runner.num_agents)
        print(f"[F4] trough pool: {n} transitions -> "
              f"{os.path.abspath(args.collect_pool)}", flush=True)
        runner.close()
        return 0

    # ---- F4 / F4b phase 2 ------------------------------------------------
    if os.environ.get("ANT_PCR_FREEZE_A") != "1.0":
        print("[F4] WARNING: ANT_PCR_FREEZE_A is not '1.0'. F4 must TRAIN at the "
              "peak; the training envs read the env var at import, in their own "
              "processes. Launch with ANT_PCR_FREEZE_A=1.0.", flush=True)

    runner, algo_args = build_runner(all_cfg, args.exp_name, args.seed,
                                     args.model_dir)
    rep = DebugReport(os.path.join(args.out, f"f4_{args.exp_name}.md"),
                      title="F4 — the controlled interference probe"
                            + ("b (rehearsal)" if args.rehearse else ""),
                      subtitle=f"train at FREEZE_A=1 from F0; score trough "
                               f"competence every {args.log_every} steps")
    rep.kv("F0 init", os.path.abspath(args.model_dir))
    rep.kv("steps", algo_args["train"]["num_env_steps"])
    rep.kv("rehearsal fraction", args.rehearse)

    rep.h2("setup")
    # The A=0 env is built ONCE and pinned (see make_trough_env). It supplies both
    # the trough probe bank and every trough return, while training runs at A=1.
    teval = make_trough_env(all_cfg, args.out, seed=args.seed)
    bank = collect_bank(teval, runner.actor, args.bank_size, seed=args.seed)
    ref = [runner.actor[i].get_actions(bank[i], stochastic=False).cpu().numpy()
           for i in range(runner.num_agents)]
    rep.line(f"  probe bank frozen: {args.bank_size} TROUGH states x "
             f"{runner.num_agents} agents, visited by F0 at FREEZE_A=0 under its "
             f"own deterministic policy, labelled with F0's actions")
    obs, share_obs, avail = runner.warmup()
    rep.line(f"  replay warmed up with {algo_args['train']['warmup_steps']} steps "
             f"at the PEAK (FREEZE_A=1)")

    pool_n = 0
    if args.rehearse > 0:
        if not args.rehearse_pool:
            rep.line("  !! --rehearse without --rehearse_pool: run phase 1 first.")
            return 2
        pool_n = load_pool(runner.buffer, args.rehearse_pool, runner.num_agents)
        install_rehearsal(runner.buffer, pool_n, args.rehearse)
        rep.line(f"  F4b: {pool_n} F0-trough transitions loaded into slots "
                 f"[0, {pool_n}); {args.rehearse:.0%} of every minibatch is drawn "
                 f"from them")
        rep.note("This is the ONLY arm in the campaign that shapes the sampler. It "
                 "is a labeled diagnostic (spec §5, F4b), not a method component: "
                 "it exists to measure what ECL's anchor bought once the "
                 "identifier, localizer and envelope are removed.")

    rep.h2("the forgetting curve")
    steps = (algo_args["train"]["num_env_steps"]
             // algo_args["train"]["n_rollout_threads"])
    n_thr = algo_args["train"]["n_rollout_threads"]
    update_num = int(algo_args["train"]["update_per_train"]
                     * algo_args["train"]["train_interval"])
    rows, train_rew, last_log = [], [], 0
    csv_path = os.path.join(args.out, f"f4_{args.exp_name}.csv")

    for step in range(1, steps + 1):
        actions = runner.get_actions(obs, available_actions=avail, add_random=True)
        (new_obs, new_share_obs, rewards, dones, infos,
         new_avail) = runner.envs.step(actions)
        train_rew.append(float(np.mean(np.asarray(rewards))))
        runner.insert((share_obs, obs.transpose(1, 0, 2), actions.transpose(1, 0, 2),
                       None, rewards, dones, infos, new_share_obs.copy(),
                       new_obs.copy(), None))
        obs, share_obs, avail = new_obs, new_share_obs, new_avail
        if step % algo_args["train"]["train_interval"] == 0:
            for _ in range(update_num):
                runner.train()
        env_step = algo_args["train"]["warmup_steps"] + step * n_thr
        if env_step - last_log >= args.log_every:
            last_log = env_step
            mse = probe_mse(runner.actor, ref, bank)
            tr, trs = trough_return(teval, runner, args.eval_episodes)
            rows.append([env_step, round(mse, 6), round(tr, 2), round(trs, 2),
                         round(float(np.mean(train_rew[-2000:])), 4), args.rehearse])
            rep.line(f"  step {env_step:>7}  probe-MSE {mse:.5f}  "
                     f"trough-return {tr:8.1f} +/- {trs:5.1f}")
            write_csv(csv_path, _COLS, rows)

    rep.h2("V7 — retention")
    if rows:
        b0, final = rows[0][2], rows[-1][2]
        rep.table(_COLS, rows)
        rep.kv("trough return at the first log (F0's, the reference)", f"{b0:.1f}")
        rep.kv("trough return after 200k peak steps", f"{final:.1f}")
        easy = final >= 0.8 * b0
        rep.verdict("V7 retention 'easy' (trough >= 0.8*B0 after 200k peak steps)",
                    easy)
        if not easy:
            xs = np.array([r[0] for r in rows], float)
            ys = np.array([r[2] for r in rows], float)
            t10 = xs[np.flatnonzero(ys <= 0.9 * b0)[0]] if np.any(ys <= 0.9 * b0) \
                else float("nan")
            t50 = xs[np.flatnonzero(ys <= 0.5 * b0)[0]] if np.any(ys <= 0.5 * b0) \
                else float("nan")
            cls = ("cliff" if np.isfinite(t10) and t10 - xs[0] <= 2 * args.log_every
                   else "fast" if np.isfinite(t50) and t50 - xs[0] <= 50000
                   else "slow")
            rep.kv("slope class", cls)
            rep.kv("steps to lose 10% of trough competence", t10)
            rep.kv("steps to lose 50%", t50)
            rep.note(f"**This is the retention budget**, and it is now a number "
                     f"rather than an intuition: any method that enters the peak "
                     f"must restore trough competence faster than '{cls}'. Overlay "
                     f"on any leaf (spec §8): V7 hard => add a retention module "
                     f"(KL-anchor / snapshot / modular heads). Compare F4 vs F4b to "
                     f"see how much of that budget pure rehearsal covers.")
    teval.close()
    runner.close()
    rep.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
