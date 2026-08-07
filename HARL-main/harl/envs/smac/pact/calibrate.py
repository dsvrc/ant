"""CWO severity calibration for SMAC -- pick SEVERITY before training (runs on server).

Rolls the REAL env with an ENGAGE-and-fire script (units move toward the enemy and
attack when in range, so the measured firing load x2 is realistic -- a wandering
script under-fires and makes this recommend a wildly-too-high SEVERITY).  Reports the
firing load x2 and the drop probability ell, and suggests a SEVERITY that lands the
peak drop prob near `--target_ell`.

*** That suggestion is a STARTING POINT, not the answer. ***  Landing ell at some
target says nothing about whether holding fire is worth more than the damage it costs
-- and that, not ell, is what decides whether the experiment measures anything.  This
criterion on its own is what produced the SEVERITY=2.0 / _LMAX=0.6 configuration in
which greedy firing beat the stagger optimum by 3.2x, so a 20M-step 3s5z run recorded a
firing fraction of 0.335 at the driver peak vs 0.345 at the trough (no modulation) and
a win rate that fell 0.95 -> 0.07 and never came back.  Always follow this with the
env banner's `coordination headroom` line and then Phase 1 (`pact.phase1`).

    python -m harl.envs.smac.pact.calibrate --map_name 3s5z --severity 1.0
"""

import argparse

import numpy as np

from harl.envs.smac.StarCraft2_Env import StarCraft2Env


def engage_fire_actions(env, rng):
    """Attack if an attack is available; else MOVE toward the enemy centroid so the
    units actually close and fire (a realistic firing load)."""
    n_no = env.n_actions_no_attack
    ex = ey = 0.0
    ne = 0
    for e in getattr(env, "enemies", {}).values():
        if getattr(e, "health", 0) > 0:
            ex += e.pos.x
            ey += e.pos.y
            ne += 1
    if ne:
        ex /= ne
        ey /= ne
    acts = []
    for i, a in enumerate(env.get_avail_actions()):
        a = np.asarray(a).reshape(-1)
        attack_ids = np.where(a[n_no:] > 0)[0] + n_no
        if attack_ids.size:
            acts.append(int(rng.choice(attack_ids)))          # in range -> fire
            continue
        u = env.get_unit_by_id(i)
        if ne and u is not None and getattr(u, "health", 0) > 0:  # move toward enemies
            dx, dy = ex - u.pos.x, ey - u.pos.y   # 2=N(+y) 3=S(-y) 4=E(+x) 5=W(-x)
            cand = (4 if dx > 0 else 5) if abs(dx) > abs(dy) else (2 if dy > 0 else 3)
            if cand < a.shape[0] and a[cand] > 0:
                acts.append(cand)
                continue
        move_ids = np.where(a > 0)[0]
        acts.append(int(rng.choice(move_ids)) if move_ids.size else 0)
    return np.array(acts, dtype=np.int64)


def main():
    p = argparse.ArgumentParser(description="CWO severity calibration")
    p.add_argument("--map_name", default="3s5z")
    p.add_argument("--state_type", default="EP")
    p.add_argument("--severity", type=float, default=1.0)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--target_ell", type=float, default=0.4, help="desired peak drop prob")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--freeze_a", type=float, default=1.0,
                   help="hold the engagement-tempo driver A(t) here (1.0 = its PEAK, "
                        "the condition SEVERITY is supposed to be sized against). "
                        "Pass -1 to let it run live.")
    args = p.parse_args()

    env = StarCraft2Env({
        "map_name": args.map_name, "state_type": args.state_type,
        "snd_severity": args.severity,
        # bypass the warmup curriculum: _curr_severity() ramps from 0 over the first
        # _WARMUP (=500k) steps of an env's life and this script runs a few thousand,
        # so without snd_eval the measured ell/drop would be 0 regardless of --severity.
        "snd_eval": 1,
        # *** FREEZE THE DRIVER, or the ell reading is an artifact of the clock. ***
        # A(t) is a raised cosine of period 5000 steps starting at its TROUGH, and
        # this script runs a few hundred: without a freeze it samples A ~= 0.01, so
        # ell = A*sigma*x2 comes out ~100x too small and the run reads as "the NS
        # does nothing" at any severity. (Measured: 302 steps gave A(mean)=0.012 and
        # ell=0.008 against a true peak ell of sigma*x2.) The load x2 is unaffected --
        # it does not depend on A -- so only the ell/drop columns were ever wrong.
        **({} if args.freeze_a < 0 else {"snd_freeze": float(args.freeze_a)}),
    })
    env.seed(args.seed)
    print(f"[calib] map={args.map_name} severity={args.severity} (engage-and-fire script)")

    rng = np.random.RandomState(args.seed)
    x2, ell, drop, firef = [], [], [], []
    for _ in range(args.episodes):
        env.reset()
        while True:
            _o, _s, _r, dones, infos, _av = env.step(engage_fire_actions(env, rng))
            d = infos[0]
            x2.append(d.get("cwo_x2_mean", 0.0)); ell.append(d.get("cwo_ell_mean", 0.0))
            drop.append(d.get("cwo_drop_frac", 0.0)); firef.append(d.get("cwo_fire_frac", 0.0))
            if bool(np.asarray(dones).reshape(-1)[0]):
                break
    env.close()

    x2, ell, drop, firef = map(np.array, (x2, ell, drop, firef))
    load = float(x2.mean())          # the firing load -- what SEVERITY multiplies
    print(f"[calib] steps={len(x2)}  fire_frac(mean)={firef.mean():.2f}  "
          f"firing load x2(mean)={load:.3f}")
    _drv = ("live (NOT frozen -- the ell column below is a snapshot of wherever the "
            "clock happened to be, not the peak)" if args.freeze_a < 0
            else f"FROZEN at A={args.freeze_a} ({'peak' if args.freeze_a >= 0.99 else 'partial'})")
    print(f"[calib] driver: {_drv}")
    print(f"[calib] at SEVERITY={args.severity}:  deflection ell(mean)={ell.mean():.3f}  "
          f"shots deflected(mean)={drop.mean():.3f}")

    if firef.mean() < 0.05 or load < 1e-3:
        print("[calib] WARNING: almost no firing was measured (units did not engage) -- "
              "the load is unreliable. Prefer reading x2_mean from a short BLIND run's "
              "pact_debug.csv.")
        return
    rec = float(np.clip(args.target_ell / max(load, 1e-3), 0.3, 4.0))
    print(f"[calib] ell-targeting suggestion: SEVERITY ~= {rec:.1f}   "
          f"(= target_ell {args.target_ell} / firing load {load:.3f}, capped to [0.3, 4]).")
    print("[calib] *** THIS IS A STARTING POINT, NOT THE ANSWER. *** Targeting an ell "
          "value says nothing about whether the NS has a coordination solution -- that "
          "is decided by (SEVERITY, _KNEE, _LMAX) jointly, and this criterion is exactly "
          "what produced the SEVERITY=2.0/_LMAX=0.6 configuration in which greedy firing "
          "beat the stagger optimum by 3.2x and a 20M-step run measured nothing.")
    print("[calib] NEXT, in order:")
    print("[calib]   1. set SEVERITY and read the env banner's `coordination headroom` "
          "line -- the gain must be comfortably above 1.0x (the same check runs offline "
          "in `python -m harl.envs.smac.pact.test_pact`);")
    print("[calib]   2. certify it: `python -m harl.envs.smac.pact.phase1` sweeps a "
          "scripted privileged hold-fire controller over (severity x gain) and reports "
          "sigma*.  Train at sigma* - 0.05.  Do not skip this.")


if __name__ == "__main__":
    main()
