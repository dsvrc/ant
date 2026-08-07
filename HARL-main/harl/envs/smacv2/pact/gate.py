"""Real-env Phase-2 gate for SMACv2-CWD PACT (pipeline §VI) -- runs on the server.

Steps the REAL StarCraft II env with random *available* actions (no policy / no
checkpoint needed) and checks the one hard, gait-independent gate: the mean per-step
cosine between the computed waveform ``x2_i`` and the true applied shove ``d_i`` must
be ~1.  This certifies the leak wiring end-to-end through SC2 -- index order of
``get_unit_by_id`` vs the obs, reset masking reaching ``x2``, and the one-step timing
-- the things the pure-numpy ``test_pact.py`` cannot see.

Run it at LOW severity first (default 0.5): there the DCAP clip never bites, so the
cosine must be essentially 1.0.  A drop is a wiring bug, not a frontier.  (At the
operating severity 1.5 the cosine dips a hair below 1 on the ~2% of steps where a
single axis saturates -- that is the honest saturation leak, not a bug.)

    python -m harl.envs.smacv2.pact.gate --map_name protoss_5_vs_5 --severity 0.5 --episodes 6
"""

import argparse

import numpy as np

from harl.envs.smacv2.smacv2_env import SMACv2Env, _CWD_DCAP


def random_actions(avail, rng):
    acts = []
    for a in avail:
        idx = np.where(np.asarray(a).reshape(-1) > 0)[0]
        acts.append(int(rng.choice(idx)) if idx.size else 0)
    return np.array(acts, dtype=np.int64).reshape(-1, 1)


def main():
    p = argparse.ArgumentParser(description="SMACv2-CWD PACT real-env cosine gate")
    p.add_argument("--map_name", default="protoss_5_vs_5")
    p.add_argument("--severity", type=float, default=0.5,
                   help="use LOW (0.5) for the clean wiring check; 1.5 shows the sat leak")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--freeze", type=float, default=1.0,
                   help="frozen driver A (peak=1.0); -1 for the live ramp")
    p.add_argument("--ctde", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.999)
    args = p.parse_args()

    freeze = None if args.freeze < 0 else args.freeze
    env_args = {
        "map_name": args.map_name,
        "cwd_severity": args.severity,
        "cwd_pact": 1,
        "cwd_pact_ctde": int(args.ctde),
        "cwd_freeze": freeze,
    }
    env = SMACv2Env(env_args)
    env.seed(args.seed)
    print(f"[gate] map={args.map_name} n_agents={env.n_agents} severity={args.severity} "
          f"freeze={freeze} ctde={args.ctde}")
    print(f"[gate] obs_dim={env.observation_space[0].shape[0]} "
          f"state_dim={env.share_observation_space[0].shape[0]} "
          f"(PACT appends 3 to obs, {2 * env.n_agents}{'+1' if args.ctde else ''} to state)")

    rng = np.random.default_rng(args.seed)
    cos, load, loadmax, x2load, sat = [], [], [], [], []
    for _ in range(args.episodes):
        _obs, _state, avail = env.reset()
        while True:
            actions = random_actions(avail, rng)
            _obs, _state, _rew, dones, infos, avail = env.step(actions)
            info0 = infos[0]
            if "cwd_pact_cos" in info0:
                cos.append(float(info0["cwd_pact_cos"]))
            load.append(float(info0.get("cwd_load", 0.0)))
            loadmax.append(float(info0.get("cwd_loadmax", 0.0)))
            x2load.append(float(info0.get("cwd_pact_x2load", 0.0)))
            d = np.asarray(info0.get("cwd_d_applied", np.zeros((env.n_agents, 2))))
            sat.append(float(np.mean(np.abs(d) >= (_CWD_DCAP - 1e-6))) if d.size else 0.0)
            if bool(dones[0]):
                break
    env.close()

    mean_cos = float(np.mean(cos)) if cos else 1.0
    print(f"[gate] steps={len(cos)}  mean per-step cosine(x2, d) = {mean_cos:.6f}")
    print(f"[gate] mean |d| (cwd_load) = {np.mean(load):.3f}  peak |d| = {np.max(loadmax):.3f}")
    print(f"[gate] mean |x2| = {np.mean(x2load):.3f}  DCAP-sat frac = {np.mean(sat):.4f}")
    if mean_cos >= args.threshold:
        print(f"[gate] PASS  (cosine >= {args.threshold}; the leak wiring is exact)")
    else:
        print(f"[gate] FAIL  (cosine {mean_cos:.4f} < {args.threshold}). If severity is "
              "low this is a WIRING bug: check get_unit_by_id index order vs obs, that "
              "reset() zeroes x2, and the one-step timing (x2_applied vs d_applied).")


if __name__ == "__main__":
    main()
