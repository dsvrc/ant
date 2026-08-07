"""X1 — conditioned-representation existence (offline distillation)  [spec §5.2].

Torch allowed here: this is offline, supervised, and diagnostic — **not** a method
component. It asks one question RL cannot answer on its own:

    Can ONE conditioned network represent the whole policy path  c -> pi*_c  ?

If yes (**V5 pass**) then H-C3 is confirmed *constructively*: representation is
not the bottleneck, RL optimization is — and "train slices, then distill" is a
certified method skeleton (leaf L2). If no, the memoryless c-conditioned class is
insufficient even under supervision, and the d-transient must be observed (weight
shifts to E5 / E3-DOB, leaf L1, or redesign R-e).

Pipeline (spec §5.2)
--------------------
1. **Data**: 200k on-policy steps from each frozen expert (F1a/F1b/F1c at its own
   A), deterministic actors + Gaussian jitter sigma=0.05 for coverage, stored as
   (per-agent obs, expert action, c-label = A*sigma).
   The *label* is the expert's deterministic action at the visited state; the
   *executed* action carries the jitter. Labelling the jittered action instead
   would teach the student the noise.
2. **Student**: per-agent net with the SAME architecture/sizes as the HASAC actor
   (literally ``SquashedGaussianPolicy``, so the tanh squash and widths match),
   input = own obs (+) c, output = action, MSE loss, 3 offline seeds.
3. **Eval**: (i) the frozen grid including the held-out A in {0.25, 0.75}
   (interpolation test); (ii) the drifting env, C4-stratified, with c supplied.

**Gate V5**: X1's drifting cycle-average >= 0.9 * PC.

----------------------------------------------------------------------------
How ``c`` reaches the student, and why the default is not ``ANT_PCR_CORACLE=1``
----------------------------------------------------------------------------
The spec says to eval "(ii) the drifting env with ``ANT_PCR_CORACLE=1`` providing
the c input". That flag appends c inside ``AntEnv._get_obs()`` — i.e. *before*
``MujocoMulti.get_obs()`` applies ``obs = (obs - mean(obs)) / std(obs)`` to the
whole concatenated vector. Two things then go wrong at once:

  * the c coordinate arrives as ``(c - mean_t)/std_t``, a **different function of
    c at every timestep** (the scale is set largely by cfrc_ext contact spikes),
    so it is no longer a usable label; and
  * appending a coordinate changes ``mean_t``/``std_t`` and therefore **every
    other coordinate too**, so the student's obs distribution no longer matches
    the blind-env obs its experts were trained and distilled on.

A V5 failure under that schema would say nothing about representational capacity
— it would be a measurement artifact. So the default here is ``--c_source info``:
run the **blind** env (obs identical to the training distribution) and take c from
``info["pcr_payload"] * severity`` — exact, unnormalized, and a clearly-labeled
privileged read of the kind §2's hygiene rule explicitly permits for diagnostic
arms. ``--c_source coracle`` reproduces the spec's literal wording for comparison;
the report prints both if you run both, and the gap is itself a measurement of
what the normalization costs a conditioned policy.

    python scripts/diag_distill.py --out diag_out/x1 \\
        --expert f1a:<run>/models@0.0 --expert f1b:<run>/models@0.5 \\
        --expert f1c:<run>/models@1.0 --pc 6100
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harl.envs.mamujoco.diag.report_io import (  # noqa: E402
    DebugReport, bootstrap_ci, fmt_ci, write_csv)
from harl.envs.mamujoco.diag import knobs as K  # noqa: E402
from scripts.diag_tier0 import load_run_config, make_env, load_actors  # noqa: E402

_P_PERIOD = 40000
_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
_HELD_OUT = (0.25, 0.75)


def parse_expert(s):
    name, _, rest = s.partition(":")
    path, _, a = rest.partition("@")
    return {"name": name, "path": path, "A": float(a)}


# ==========================================================================
#  1. data
# ==========================================================================
def collect(cfg, ckpt, out, A, n_steps, jitter, device, seed=0):
    """Roll the expert at its own A; label with its DETERMINISTIC action."""
    K.apply(freeze_a=A, mask="both", dcap=None)
    env = make_env(cfg, out, f"x1_collect_A{A}")
    env.seed(seed)
    actors = load_actors(cfg, ckpt, env, device)
    sev = K.current_knobs()["SEVERITY"]
    rng = np.random.default_rng(seed)
    obs_l, act_l, c_l = [], [], []
    obs, _, _ = env.reset()
    for t in range(n_steps):
        with torch.no_grad():
            det = [actors[i].get_actions(np.asarray(obs[i])[None],
                                         stochastic=False).cpu().numpy()[0]
                   for i in range(env.n_agents)]
        obs_l.append(np.asarray(obs, dtype=np.float32))          # (N, obs_dim)
        act_l.append(np.asarray(det, dtype=np.float32))          # (N, act_dim)
        c_l.append(np.float32(A * sev))
        exec_a = np.clip(np.asarray(det) + jitter * rng.standard_normal(
            (env.n_agents, len(det[0]))), -1.0, 1.0)
        obs, _, _, dones, _, _ = env.step(exec_a)
        if np.all(dones):
            obs, _, _ = env.reset()
    env.close()
    return (np.asarray(obs_l), np.asarray(act_l), np.asarray(c_l))


# ==========================================================================
#  2. student
# ==========================================================================
def make_student(cfg, obs_dim, act_space, device):
    """Same architecture/sizes as the HASAC actor — literally the same class, so
    the widths, activations and tanh squash match by construction rather than by
    a comment claiming they do."""
    from gym.spaces import Box
    from harl.models.policy_models.squashed_gaussian_policy import (
        SquashedGaussianPolicy)

    args = {**cfg["algo_args"]["model"], **cfg["algo_args"]["algo"]}
    return SquashedGaussianPolicy(args, Box(-10, 10, (obs_dim + 1,)), act_space,
                                  torch.device(device))


def train_student(data, cfg, act_space, device, seed, epochs, batch, lr, rep):
    torch.manual_seed(seed)
    obs, act, c = data
    T, N, od = obs.shape
    students, losses = [], []
    for i in range(N):
        X = np.concatenate([obs[:, i, :], c[:, None]], axis=-1).astype(np.float32)
        Y = act[:, i, :].astype(np.float32)
        s = make_student(cfg, od, act_space, device)
        opt = torch.optim.Adam(s.parameters(), lr=lr)
        Xt = torch.as_tensor(X, device=device)
        Yt = torch.as_tensor(Y, device=device)
        last = float("nan")
        for ep in range(epochs):
            perm = torch.randperm(T, device=device)
            tot, nb = 0.0, 0
            for b in range(0, T, batch):
                idx = perm[b:b + batch]
                pred, _ = s(Xt[idx], stochastic=False, with_logprob=False)
                loss = torch.nn.functional.mse_loss(pred, Yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.item())
                nb += 1
            last = tot / max(nb, 1)
        students.append(s)
        losses.append(last)
    rep.line(f"  seed {seed}: final per-agent MSE = "
             f"{[round(x, 5) for x in losses]}")
    return students, losses


# ==========================================================================
#  3. eval
# ==========================================================================
@torch.no_grad()
def eval_student(students, cfg, out, A, episodes, device, c_source, tag, seed=0):
    """A=None => the drifting env, clock-stratified (C4)."""
    use_coracle = (c_source == "coracle" and A is None)
    if use_coracle:
        os.environ["ANT_PCR_CORACLE"] = "1"
    K.apply(freeze_a=A, mask="both", dcap=None)
    sev = K.current_knobs()["SEVERITY"]
    env = make_env(cfg, out, f"x1_eval_{tag}")
    env.seed(seed)
    rets, lens = [], []
    for ep in range(episodes):
        if A is None:
            inner = getattr(env.env, "unwrapped", env.env)
            inner._clock = (ep * _P_PERIOD) // max(1, episodes)
        obs, _, _ = env.reset()
        R, L, done = 0.0, 0, False
        c = (A * sev) if A is not None else 0.0
        while not done:
            acts = []
            for i in range(env.n_agents):
                o = np.asarray(obs[i], dtype=np.float32)
                if use_coracle:
                    x = o[None]              # c already inside (normalized!)
                else:
                    x = np.concatenate([o, [np.float32(c)]])[None]
                a, _ = students[i](torch.as_tensor(x, device=device),
                                   stochastic=False, with_logprob=False)
                acts.append(a.cpu().numpy()[0])
            obs, _, rewards, dones, infos, _ = env.step(np.asarray(acts))
            if A is None and isinstance(infos[0], dict):
                c = float(infos[0].get("pcr_payload", 0.0)) * sev
            R += float(np.mean(np.asarray(rewards)))
            L += 1
            done = bool(np.all(dones))
        rets.append(R)
        lens.append(L)
    env.close()
    if use_coracle:
        os.environ.pop("ANT_PCR_CORACLE", None)
    return np.asarray(rets), np.asarray(lens)


# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="X1 — offline conditioned distillation.")
    ap.add_argument("--expert", action="append", required=True,
                    help="'name:path/to/models@A' — repeatable (f1a@0.0, f1b@0.5, "
                         "f1c@1.0)")
    ap.add_argument("--out", default="./diag_out/x1")
    ap.add_argument("--steps_per_expert", type=int, default=200000)
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pc", type=float, default=None,
                    help="the path ceiling PC from diag_crosseval (needed for V5)")
    ap.add_argument("--c_source", default="info", choices=("info", "coracle"),
                    help="how c reaches the student on the drifting eval. "
                         "'info' (default) = blind env + exact c from info — the "
                         "schema the student was trained in. 'coracle' = the spec's "
                         "literal ANT_PCR_CORACLE=1, where c arrives normalized "
                         "(see the module docstring).")
    args = ap.parse_args(argv)

    rep = DebugReport(os.path.join(args.out, "x1_distill.md"),
                      title="X1 — conditioned-representation existence",
                      subtitle="offline supervised distillation; gate V5")
    experts = [parse_expert(s) for s in args.expert]
    rep.kv("experts", ", ".join(f"{e['name']}@A={e['A']}" for e in experts))
    rep.kv("c source on the drifting eval", args.c_source)
    if args.c_source == "coracle":
        rep.note("`--c_source coracle` reproduces the spec's literal wording, but c "
                 "then arrives as (c - mean_t)/std_t and the whole obs vector is "
                 "rescaled relative to the student's training distribution. A "
                 "failure here does NOT refute representational capacity. Prefer "
                 "`--c_source info` for the V5 reading; run this one only as the "
                 "contrast.")

    # ---- 1. data ---------------------------------------------------------
    rep.h2("1. data")
    cfg0 = None
    parts = []
    for e in experts:
        cfg, _ = load_run_config(e["path"])
        cfg0 = cfg0 or cfg
        n = args.steps_per_expert
        rep.line(f"  collecting {n} steps from {e['name']} at A={e['A']} "
                 f"(deterministic + N(0, {args.jitter}) jitter for coverage)")
        parts.append(collect(cfg, e["path"], args.out, e["A"], n, args.jitter,
                             args.device))
    obs = np.concatenate([p[0] for p in parts])
    act = np.concatenate([p[1] for p in parts])
    c = np.concatenate([p[2] for p in parts])
    rep.kv("dataset", f"obs {obs.shape}, act {act.shape}, c in "
                      f"{sorted(set(np.round(c, 4).tolist()))}")
    np.savez_compressed(os.path.join(args.out, "x1_data_meta.npz"),
                        c_values=np.unique(c), n=len(c))

    # ---- 2. train --------------------------------------------------------
    rep.h2("2. student (same architecture/sizes as the HASAC actor)")
    tmp_env = make_env(cfg0, args.out, "x1_spaces")
    act_space = tmp_env.action_space[0]
    tmp_env.close()
    all_students = []
    for s in range(args.seeds):
        st, _ = train_student((obs, act, c), cfg0, act_space, args.device, s,
                              args.epochs, args.batch, args.lr, rep)
        all_students.append(st)

    # ---- 3. eval ---------------------------------------------------------
    rep.h2("3. eval — frozen grid (incl. held-out A) + drifting")
    rows = []
    frozen = {A: [] for A in _GRID}
    drift = []
    for si, st in enumerate(all_students):
        for A in _GRID:
            r, _ = eval_student(st, cfg0, args.out, A, args.episodes, args.device,
                                args.c_source, f"s{si}_A{A}")
            frozen[A].append(float(np.mean(r)))
            rows.append(["frozen", si, A, round(float(np.mean(r)), 2),
                         round(float(np.std(r)), 2), args.episodes,
                         A in _HELD_OUT])
        r, _ = eval_student(st, cfg0, args.out, None, args.episodes, args.device,
                            args.c_source, f"s{si}_drift")
        drift.append(r)
        rows.append(["drift", si, None, round(float(np.mean(r)), 2),
                     round(float(np.std(r)), 2), args.episodes, False])

    rep.table(["A", "held out?", "student (mean over seeds)", "per-seed"],
              [[A, "YES" if A in _HELD_OUT else "",
                f"{np.mean(frozen[A]):.0f}",
                ", ".join(f"{v:.0f}" for v in frozen[A])] for A in _GRID])
    all_drift = np.concatenate(drift)
    rep.kv("drifting cycle-average (C4-stratified, pooled over seeds)",
           fmt_ci(all_drift))
    rep.line("  per-seed: " + ", ".join(f"{np.mean(d):.0f}" for d in drift))

    # ---- V5 --------------------------------------------------------------
    rep.h2("V5 — conditioned-representation capacity")
    if args.pc is None:
        rep.line("  --pc not given: run diag_crosseval.py first and pass its PC. "
                 "V5 cannot be read without it.")
        rep.verdict("V5 conditioned-representation", False,
                    "UNDETERMINED — PC missing, not a failure")
    else:
        got = float(np.mean(all_drift))
        bar = 0.9 * args.pc
        passed = got >= bar
        rep.kv("drifting cycle-avg", f"{got:.1f}")
        rep.kv("0.9 * PC — the bar", f"{bar:.1f}")
        rep.verdict("V5 conditioned-representation (drift cycle-avg >= 0.9*PC)",
                    passed)
        if passed:
            rep.note("**H-C3 confirmed constructively.** One network holds the "
                     "whole path; RL — not representation — is the bottleneck. "
                     "'Train slices, then distill' is a certified method skeleton "
                     "(leaf L2). If v2's [L]/[A] with perfect tags failed while "
                     "this passes, the telemetry will show WHY localized RL != "
                     "supervised distillation — that contrast is itself a paper "
                     "section.")
        else:
            ho = np.mean([np.mean(frozen[a]) for a in _HELD_OUT])
            tr = np.mean([np.mean(frozen[a]) for a in _GRID if a not in _HELD_OUT])
            if ho < 0.8 * tr:
                rep.note(f"**Fail mode: held-out-A dips only** (held-out "
                         f"{ho:.0f} vs trained-slice {tr:.0f}). The path has "
                         f"curvature the 3 experts cannot span => add A=0.25/0.75 "
                         f"expert runs (P2) and re-distill (spec §5.2).")
            else:
                rep.note("**Fail mode: uniform.** Within-slice POMDP-ness matters — "
                         "the memoryless c-conditioned class is insufficient even "
                         "under SUPERVISION, so no amount of RL fixes it. The "
                         "d-transient must be observed: weight shifts to E5 / "
                         "E3-DOB (leaf L1) or to redesign R-e (spec §5.2).")

    write_csv(os.path.join(args.out, "x1_eval.csv"),
              ["protocol", "seed", "A", "ret_mean", "ret_std", "episodes",
               "held_out"], rows)
    rep.kv("x1_eval.csv", os.path.join(args.out, "x1_eval.csv"))
    rep.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
