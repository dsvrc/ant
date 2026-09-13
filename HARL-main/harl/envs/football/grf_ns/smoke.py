"""In-simulator identities.  Needs gfootball; ~3-5 minutes.

Everything ``selftest.py`` proves about the channel arithmetic is re-checked
here THROUGH the real engine, plus the things only the engine can answer: which
players the controller slots drive (and when the engine switches them), the
direction table's handedness (gate 4, physically), where the scenario actually
spawns, and whether two engine instances reproduce at all.

GRF does NOT reproduce across instances under ``env.seed()`` (measured: two
envs with the same seed and the same actions diverge at step 0), so the
identities are checked at the ACTION level -- the executed action equals the
commanded one on every step for sigma = 0, the placebo and the oracle, and a
shadow blind channel fed the same positions enacts exactly what ``pactoff``
enacts.  That is exact and engine-independent.  Trajectory-level bit-identity
is attempted only if a reproducibility probe succeeds, and reported as SKIP
otherwise.

    python -m harl.envs.football.grf_ns.smoke
    python -m harl.envs.football.grf_ns.smoke --steps 600      # longer runs

Ends with ``ALL SMOKE CHECKS PASSED`` or lists the failures and exits 1.
"""

import argparse
import sys

import numpy as np

from harl.utils.configs_tools import get_defaults_yaml_args

from . import make_football_ns_env
from .actions import DIR_FIRST, IDLE, N_DIRS, RELEASE_DIRECTION, SHORT_PASS, SPRINT
from .ceiling import SPAWN
from .channel import PactConfig, SwerveChannel
from .driver import DialParams

FAILS = []
P = DialParams()
PEAK = int(P.period * P.wet_fraction / 2)
DRY = int(P.period * P.wet_fraction)
GK = 0                                            # GRF's goalkeeper is always player 0


def check(name, cond, detail=""):
    ok = bool(cond)
    print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def skip(name, why):
    print("  [SKIP] %-58s %s" % (name, why), flush=True)


def make(env_name="academy_3_vs_1_with_keeper", n=3, **kw):
    _, ea = get_defaults_yaml_args("mappo", "football")
    ea = dict(ea)
    ea["env_name"] = env_name
    ea["number_of_left_players_agent_controls"] = n
    if env_name != "academy_3_vs_1_with_keeper":
        ea["ns_load_norm"] = None                  # the committed value is 3v1's
    ea.update(kw)
    return make_football_ns_env(ea, 0, 1)


def random_actions(rng, n):
    a = np.zeros((n, 1), dtype=np.int64)
    for i in range(n):
        u = rng.rand()
        if u < 0.55:
            a[i, 0] = IDLE
        elif u < 0.88:
            a[i, 0] = DIR_FIRST + rng.randint(0, N_DIRS)
        elif u < 0.92:
            a[i, 0] = RELEASE_DIRECTION
        elif u < 0.96:
            a[i, 0] = SPRINT
        else:
            a[i, 0] = SHORT_PASS
    return a


def raw(env):
    return env.env.unwrapped.observation()


def actives(rr):
    return tuple(int(o.get("active", -1)) for o in rr)


def designated(rr):
    return int(rr[0].get("designated", -1))


def rollout(env, seed, steps, policy_seed=0):
    """Fixed seed, fixed open-loop random policy.  Returns the trajectory and,
    per step, the commanded/executed actions from the layer's info."""
    env.seed(seed)
    obs, _, _ = env.reset()
    rng = np.random.RandomState(policy_seed)
    n = len(obs)
    traj, cmd, exe, corr = [], [], [], []
    for _ in range(steps):
        a = random_actions(rng, n)
        obs, _, rew, done, info, _ = env.step(a)
        traj.append((np.concatenate([np.asarray(o, dtype=np.float64).ravel() for o in obs]),
                     float(np.asarray(rew).ravel()[0]), bool(np.all(done))))
        if isinstance(info[0], dict) and "ns_executed" in info[0]:
            cmd.append([info[i]["ns_commanded"] for i in range(n)])
            exe.append([info[i]["ns_executed"] for i in range(n)])
            corr.append([info[i]["ns_c"] for i in range(n)])
        if np.all(done):
            obs, _, _ = env.reset()
    return traj, np.array(cmd), np.array(exe), np.array(corr)


def identical(t1, t2):
    if len(t1) != len(t2):
        return False, "lengths %d vs %d" % (len(t1), len(t2))
    for k, ((o1, r1, d1), (o2, r2, d2)) in enumerate(zip(t1, t2)):
        if o1.shape != o2.shape or not np.array_equal(o1, o2) or r1 != r2 or d1 != d2:
            return False, "first divergence at step %d" % k
    return True, "%d steps identical, obs and reward bit for bit" % len(t1)


def action_identity(cmd, exe):
    bad = int(np.sum(np.any(cmd != exe, axis=1))) if len(cmd) else -1
    return bad == 0, "%d of %d steps had an executed action != commanded" % (bad, len(cmd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    S = args.steps
    zeros3 = np.zeros((3, 1), dtype=np.int64)

    # ------------------------------------------------------------------ slots
    print("controller slots -- who the agents drive, and when the engine switches them",
          flush=True)
    e = make(ns_severity=0.0)
    e.seed(1)
    e.reset()
    rr = raw(e)
    seq = [(0, actives(rr), designated(rr))]
    for t in range(1, 41):
        _, _, _, done, _, _ = e.step(zeros3)                # all idle: the engine's own dynamics
        rr = raw(e)
        seq.append((t, actives(rr), designated(rr)))
        if np.all(done):
            break
    print("  all-idle: " + "  ".join("t%d:%s d=%d" % s for s in seq[:5])
          + ("  ...  t%d:%s d=%d" % seq[-1] if len(seq) > 5 else ""), flush=True)
    steady = seq[1][1] if len(seq) > 1 else seq[0][1]
    check("slots_settle_after_the_first_tick",
          len(seq) > 1 and all(s[1] == steady for s in seq[1:]),
          "kick-off frame %s -> %s over %d idle ticks (distinct after tick 1: %s)"
          % (seq[0][1], steady, len(seq) - 1, sorted({s[1] for s in seq[1:]})))
    check("steady_state_slots_are_the_outfield_players_not_the_keeper",
          GK not in steady and len(set(steady)) == e.n_agents and -1 not in steady,
          "steady slots %s (player %d is the GK); designated player after tick 1: %d"
          % (steady, GK, seq[1][2] if len(seq) > 1 else -1))
    e.seed(2)
    e.reset()
    rng = np.random.RandomState(5)
    prev = actives(raw(e))
    switches, steps_seen, gk_steps = 0, 0, 0
    for _ in range(150):
        _, _, _, done, _, _ = e.step(random_actions(rng, 3))
        cur = actives(raw(e))
        steps_seen += 1
        switches += int(cur != prev)
        gk_steps += int(GK in cur)
        prev = cur
        if np.all(done):
            e.reset()
            prev = actives(raw(e))
    print("  random policy, 150 steps: %d slot switches, keeper in a slot on %d steps "
          "(kick-off frames included); layer counted %d" % (switches, gk_steps, e.chan.n_switch),
          flush=True)
    e.close()

    # ------------------------------------------------------------------ gate 4
    print("gate 4 -- the engine's handedness and sticky order", flush=True)
    e1 = make("academy_empty_goal_close", 1, ns_severity=0.0)
    e1.seed(1)
    e1.reset()
    for _ in range(8):
        e1.step(np.array([[3]]))                    # top
    o = raw(e1)[0]
    v_top = np.asarray(o["left_team_direction"][o["active"]], dtype=np.float64)
    st_top = np.asarray(o["sticky_actions"])[:N_DIRS]
    for _ in range(8):
        e1.step(np.array([[5]]))                    # right
    o = raw(e1)[0]
    v_right = np.asarray(o["left_team_direction"][o["active"]], dtype=np.float64)
    st_right = np.asarray(o["sticky_actions"])[:N_DIRS]
    check("action_top_moves_toward_negative_y", v_top[1] < 0 and abs(v_top[1]) > abs(v_top[0]),
          "velocity after 8x top = %s" % np.round(v_top, 4).tolist())
    check("action_right_moves_toward_positive_x", v_right[0] > 0
          and abs(v_right[0]) > abs(v_right[1]),
          "velocity after 8x right = %s" % np.round(v_right, 4).tolist())
    check("sticky_index_order_matches_the_action_order",
          int(np.argmax(st_top)) == 2 and int(np.argmax(st_right)) == 4,
          "sticky after top=%s after right=%s" % (st_top.tolist(), st_right.tolist()))
    e1.close()

    # ------------------------------------------------------------------ spawn
    print("spawn geometry after the first tick vs ceiling.py's declared table", flush=True)
    e3 = make(ns_severity=0.0)
    e3.seed(1)
    e3.reset()
    e3.step(zeros3)                                 # let the slots settle
    rr = raw(e3)
    act = actives(rr)
    pos = np.array([rr[i]["left_team"][act[i]] for i in range(3)], dtype=np.float64)
    roles = [int(rr[i]["left_team_roles"][act[i]]) for i in range(3)]
    decl = SPAWN["academy_3_vs_1_with_keeper"]
    key = lambda P_: P_[np.lexsort((P_[:, 1], P_[:, 0]))]        # noqa: E731
    check("controlled_players_spawn_where_the_table_says",
          np.allclose(key(pos), key(decl), atol=0.02),
          "engine=%s declared=%s roles=%s slots=%s"
          % (np.round(pos, 3).tolist(), decl.tolist(), roles, act))
    check("observation_space_is_the_host's", e3.observation_space[0].shape[0] == 115,
          "obs dim %d" % e3.observation_space[0].shape[0])
    e3.close()

    # ------------------------------------------------------------------ reproducibility
    print("reproducibility probe -- do two engine instances agree at all?", flush=True)
    sa, sb = make(ns_on=0), make(ns_on=0)
    ta, _, _, _ = rollout(sa, 7, 60)
    tb, _, _, _ = rollout(sb, 7, 60)
    ok_seed, msg_seed = identical(ta, tb)
    sa.close()
    sb.close()
    print("  two stock envs, env.seed(7), same actions: %s (%s)" % (ok_seed, msg_seed), flush=True)
    ok_cfg, msg_cfg = False, ""
    extra = {}
    try:
        opt = {"other_config_options": {"game_engine_random_seed": 7}}
        sa, sb = make(ns_on=0, **opt), make(ns_on=0, **opt)
        ta, _, _, _ = rollout(sa, 7, 60)
        tb, _, _, _ = rollout(sb, 7, 60)
        ok_cfg, msg_cfg = identical(ta, tb)
        sa.close()
        sb.close()
        if ok_cfg and not ok_seed:
            extra = opt
    except Exception as ex:  # noqa: BLE001
        msg_cfg = "raised %r" % ex
    print("  two stock envs, game_engine_random_seed=7 in other_config_options: %s (%s)"
          % (ok_cfg, msg_cfg), flush=True)
    reproducible = ok_seed or ok_cfg
    if not reproducible:
        print("  -> the engine does not reproduce across instances; trajectory identities are "
              "SKIPPED and the action-level identities below carry the claim", flush=True)

    # ------------------------------------------------------------------ NS-2.1
    print("NS-2.1 -- sigma = 0 executes exactly the commanded actions", flush=True)
    layer0 = make(ns_severity=0.0, **extra)
    t0, c0, x0, _ = rollout(layer0, 7, S)
    ok, msg = action_identity(c0, x0)
    check("sigma_0_executes_exactly_the_commanded_actions", ok, msg)
    if reproducible:
        stock = make(ns_on=0, **extra)
        ts, _, _, _ = rollout(stock, 7, S)
        ok, msg = identical(ts, t0)
        check("sigma_0_trajectory_is_bit_identical_to_the_stock_host", ok, msg)
        stock.close()
    else:
        skip("sigma_0_trajectory_is_bit_identical_to_the_stock_host", "engine not reproducible")

    # ------------------------------------------------------------------ NS-2.5
    print("NS-2.5 -- the placebo half is inert at sigma = 3", flush=True)
    plc = make(ns_severity=3.0, ns_phase0=DRY, **extra)
    tp, cp, xp, _ = rollout(plc, 7, S)
    ok, msg = action_identity(cp, xp)
    check("placebo_executes_exactly_the_commanded_actions", ok, msg)
    check("placebo_counters_say_so", plc.chan.n_dial_live == 0 and plc.chan.n_harmed == 0
          and plc.chan.n_rot_sent == 0,
          "dial_live=%d harmed=%d rotated=%d" % (plc.chan.n_dial_live, plc.chan.n_harmed,
                                                 plc.chan.n_rot_sent))
    if reproducible:
        ok, msg = identical(t0, tp)
        check("placebo_trajectory_is_bit_identical_to_sigma_0", ok, msg)
    else:
        skip("placebo_trajectory_is_bit_identical_to_sigma_0", "engine not reproducible")
    plc.close()

    # ------------------------------------------------------------------ NS-3.3
    print("NS-3.3 -- the layer fires at the driver peak", flush=True)
    blind = make(ns_severity=2.0, ns_phase0=PEAK)
    _, cb, xb, _ = rollout(blind, 7, S)
    c = blind.chan
    check("dial_live_harmed_and_rotated_are_all_non_zero",
          c.n_dial_live > 0 and c.n_harmed > 0 and c.n_rot_sent > 0,
          "dial_live=%d harmed=%d rotated=%d switches=%d controlled=%d of %d agent-steps"
          % (c.n_dial_live, c.n_harmed, c.n_rot_sent, c.n_switch, c.n_controlled, 3 * S))
    ok, msg = action_identity(cb, xb)
    check("sigma_2_actually_changes_the_actions", not ok, msg)
    _, _, _, _, info, _ = blind.step(random_actions(np.random.RandomState(1), 3))
    check("info_carries_the_panel", all(k in info[0] for k in (
        "ns_d", "ns_k", "ns_A", "ns_fit_gain", "ns_harmed", "ns_active", "score_reward")),
        "%d ns_* keys" % len([k for k in info[0] if str(k).startswith("ns_")]))
    check("layer_close_report_accepts_a_live_dial", c.close_report(tag="  [smoke]"))
    blind.close()

    # ------------------------------------------------------------------ P-7.1
    print("P-7.1 -- pactoff enacts exactly what a blind channel would (shadow check)",
          flush=True)
    off = make(ns_severity=2.0, ns_phase0=PEAK, ns_pact=1, ns_trust="off")
    off.seed(7)
    obs, _, _ = off.reset()
    shadow = SwerveChannel(off.n_agents, off.ns, off.driver, off.coupling, PactConfig(),
                           load_norm=off.chan.load_norm, ref=off.chan.ref, scale=off.chan.scale,
                           clock0=off.chan.clock)
    rng = np.random.RandomState(0)
    prev = off._active_prev.copy()
    mismatch, corr_max, n_eps = 0, 0.0, 0
    for _ in range(S):
        a = random_actions(rng, off.n_agents)
        pos, owner, eng, act = off._squad(off._raw)         # the inputs the layer is about to use
        for i in range(off.n_agents):
            if act[i] >= 0 and prev[i] >= 0 and act[i] != prev[i]:
                shadow.reset_slot(i, eng[i])
        prev = act
        e_shadow = shadow.step(a.reshape(-1), pos, owner)
        _, _, _, done, info, _ = off.step(a)
        e_env = np.array([info[i]["ns_executed"] for i in range(off.n_agents)])
        mismatch += int(np.any(e_shadow != e_env))
        corr_max = max(corr_max, max(abs(float(info[i]["ns_c"])) for i in range(off.n_agents)))
        if np.all(done):
            off.reset()
            shadow.reset_episode()
            prev = off._active_prev.copy()
            n_eps += 1
    check("pactoff_enacts_exactly_the_blind_channel's_actions", mismatch == 0,
          "%d of %d steps differed from the shadow blind channel (%d episode resets)"
          % (mismatch, S, n_eps))
    check("pactoff_estimator_ran_but_corrected_nothing",
          off.chan.rows.sum() > 0 and corr_max == 0.0,
          "rows=%d max|c|=%g" % (int(off.chan.rows.sum()), corr_max))
    off.close()

    # ------------------------------------------------------------------ II.6
    print("II.6 -- the oracle cancels the swerve exactly", flush=True)
    orc = make(ns_severity=2.0, ns_phase0=PEAK, ns_pact=1, ns_trust="fixed", ns_g_fixed=1.0,
               ns_oracle=1, ns_corr_clip=0.0, **extra)
    to, co, xo, _ = rollout(orc, 7, S)
    ok, msg = action_identity(co, xo)
    check("oracle_executes_exactly_the_commanded_actions_at_sigma_2", ok, msg)
    check("oracle_saw_a_live_dial_and_enacted_nothing",
          orc.chan.n_dial_live > 0 and orc.chan.n_harmed == 0 and orc.chan.n_rot_sent == 0,
          "dial_live=%d harmed=%d rotated=%d" % (orc.chan.n_dial_live, orc.chan.n_harmed,
                                                 orc.chan.n_rot_sent))
    if reproducible:
        ok, msg = identical(t0, to)
        check("oracle_trajectory_is_bit_identical_to_sigma_0", ok, msg)
    else:
        skip("oracle_trajectory_is_bit_identical_to_sigma_0", "engine not reproducible")
    orc.close()
    layer0.close()

    # ------------------------------------------------------------------ I.2
    print("I.2 -- a lone player reads exactly zero; the (B) control does not", flush=True)
    solo = make("academy_empty_goal_close", 1, ns_severity=3.0, ns_phase0=PEAK)
    _, cs, xs, _ = rollout(solo, 7, 200)
    ok, msg = action_identity(cs, xs)
    check("lone_player_is_untouched_at_sigma_3",
          ok and solo.chan.n_dial_live == 0 and float(np.abs(solo.chan.d).max()) == 0.0,
          "dial_live=%d harmed=%d; %s" % (solo.chan.n_dial_live, solo.chan.n_harmed, msg))
    solo.close()
    soloB = make("academy_empty_goal_close", 1, ns_severity=3.0, ns_phase0=PEAK, ns_direct=1)
    rollout(soloB, 7, 200)
    check("direct_B_control_reaches_the_lone_player",
          soloB.chan.n_dial_live > 0 and soloB.chan.n_harmed > 0,
          "dial_live=%d harmed=%d" % (soloB.chan.n_dial_live, soloB.chan.n_harmed))
    soloB.close()

    # ------------------------------------------------------------------ PACT
    print("PACT -- the estimator identifies through the engine", flush=True)
    pact = make(ns_severity=2.0, ns_phase0=PEAK, ns_pact=1, ns_trust="fixed")
    rollout(pact, 7, max(S, 600), policy_seed=3)
    pc = pact.chan
    check("estimator_absorbed_rows_and_engaged_trust",
          pc.rows.sum() > 3 * pc.pact.warmup and float(pc.trust.max()) > 0.5,
          "rows=%d trust=%s conf=%s" % (int(pc.rows.sum()), np.round(pc.trust, 3).tolist(),
                                        np.round(pc.conf, 3).tolist()))
    check("fit_gain_and_beta_cos_are_positive_through_the_engine",
          float(np.mean(pc.fit_gain)) > 0.0 and float(np.mean(pc.beta_cos)) > 0.5,
          "fit_gain=%.3f beta_cos=%.3f beta_relerr=%.3f cond_psi=%.1f diverged=%d "
          "corr_clipped=%d harmed=%d of %d"
          % (float(np.mean(pc.fit_gain)), float(np.mean(pc.beta_cos)),
             float(np.mean(pc.beta_err)), pc.cond_psi, pc.n_diverged, pc.n_corr_clipped,
             pc.n_harmed, pc.n_steps))
    pact.close()

    # ------------------------------------------------------------------ obs tail
    aug = make(ns_severity=1.0, ns_observe_residual=1)
    aug.seed(1)
    ob, _, _ = aug.reset()
    check("observe_residual_adds_exactly_one_obs_dim",
          aug.observation_space[0].shape[0] == 116 and np.asarray(ob[0]).shape[0] == 116)
    aug.close()

    if FAILS:
        print("\nFAILED %d smoke check(s): %s" % (len(FAILS), FAILS))
        sys.exit(1)
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
