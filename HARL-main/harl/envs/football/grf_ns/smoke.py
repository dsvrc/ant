"""In-simulator identities.  Needs gfootball; ~2-4 minutes.

Everything ``selftest.py`` proves about the channel arithmetic is re-checked
here THROUGH the real engine, plus the two things only the engine can answer:
does the direction table have the engine's handedness (gate 4, physically), and
does the scenario spawn where ``ceiling.py`` says it does.

    python -m harl.envs.football.grf_ns.smoke
    python -m harl.envs.football.grf_ns.smoke --steps 600      # longer pairs

Ends with ``ALL SMOKE CHECKS PASSED`` or lists the failures and exits 1.
"""

import argparse
import sys

import numpy as np

from harl.utils.configs_tools import get_defaults_yaml_args

from . import make_football_ns_env
from .actions import DIR_FIRST, IDLE, N_DIRS, RELEASE_DIRECTION, SHORT_PASS, SPRINT
from .ceiling import SPAWN
from .driver import DialParams

FAILS = []
P = DialParams()
PEAK = int(P.period * P.wet_fraction / 2)
DRY = int(P.period * P.wet_fraction)


def check(name, cond, detail=""):
    ok = bool(cond)
    print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def make(env_name="academy_3_vs_1_with_keeper", n=3, **ns):
    _, ea = get_defaults_yaml_args("mappo", "football")
    ea = dict(ea)
    ea["env_name"] = env_name
    ea["number_of_left_players_agent_controls"] = n
    if env_name != "academy_3_vs_1_with_keeper":
        ea["ns_load_norm"] = None                  # the committed value is 3v1's
    ea.update(ns)
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


def rollout(env, seed, steps, policy_seed=0):
    """Fixed seed, fixed open-loop random policy: two envs given the same pair
    must produce the same trajectory unless the layer changed something."""
    env.seed(seed)
    obs, _, _ = env.reset()
    rng = np.random.RandomState(policy_seed)
    n = len(obs)
    traj = []
    for _ in range(steps):
        a = random_actions(rng, n)
        obs, _, rew, done, info, _ = env.step(a)
        traj.append((np.concatenate([np.asarray(o, dtype=np.float64).ravel() for o in obs]),
                     float(np.asarray(rew).ravel()[0]), bool(np.all(done))))
        if np.all(done):
            obs, _, _ = env.reset()
    return traj


def identical(t1, t2):
    if len(t1) != len(t2):
        return False, "lengths %d vs %d" % (len(t1), len(t2))
    for k, ((o1, r1, d1), (o2, r2, d2)) in enumerate(zip(t1, t2)):
        if o1.shape != o2.shape or not np.array_equal(o1, o2) or r1 != r2 or d1 != d2:
            return False, "first divergence at step %d" % k
    return True, "%d steps identical, obs and reward bit for bit" % len(t1)


def raw(env):
    return env.env.unwrapped.observation()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    S = args.steps

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
    print("spawn geometry vs ceiling.py's declared table", flush=True)
    e3 = make(ns_severity=0.0)
    e3.seed(1)
    e3.reset()
    rr = raw(e3)
    pos = np.array([rr[i]["left_team"][rr[i]["active"]] for i in range(3)], dtype=np.float64)
    roles = [int(rr[i]["left_team_roles"][rr[i]["active"]]) for i in range(3)]
    decl = SPAWN["academy_3_vs_1_with_keeper"]
    key = lambda P_: P_[np.lexsort((P_[:, 1], P_[:, 0]))]        # noqa: E731
    check("controlled_players_spawn_where_the_table_says",
          np.allclose(key(pos), key(decl), atol=2e-3),
          "engine=%s declared=%s roles=%s active=%s"
          % (np.round(pos, 3).tolist(), decl.tolist(), roles, [rr[i]["active"] for i in range(3)]))
    check("observation_space_is_the_host's", e3.observation_space[0].shape[0] == 115,
          "obs dim %d" % e3.observation_space[0].shape[0])
    e3.close()

    # ------------------------------------------------------------------ NS-2.1
    print("NS-2.1 -- sigma = 0 is stock GRF byte for byte", flush=True)
    stock = make(ns_on=0)
    layer0 = make(ns_severity=0.0)
    t_stock = rollout(stock, 7, S)
    t_layer0 = rollout(layer0, 7, S)
    ok, msg = identical(t_stock, t_layer0)
    check("sigma_0_is_bit_identical_to_the_stock_host", ok, msg
          + ("" if ok else "  <-- if this fails, first re-run once: two stock envs with the "
                            "same seed must agree, or the ENGINE is non-deterministic and "
                            "every bit-identity below has to be read against that"))
    stock.close()

    # ------------------------------------------------------------------ NS-2.5
    print("NS-2.5 -- the placebo half is inert at sigma = 3", flush=True)
    plc = make(ns_severity=3.0, ns_phase0=DRY)
    t_plc = rollout(plc, 7, S)
    ok, msg = identical(t_layer0, t_plc)
    check("placebo_regime_is_bit_identical_to_sigma_0", ok, msg)
    check("placebo_counters_say_so", plc.chan.n_dial_live == 0 and plc.chan.n_harmed == 0,
          "dial_live=%d harmed=%d" % (plc.chan.n_dial_live, plc.chan.n_harmed))
    plc.close()

    # ------------------------------------------------------------------ NS-3.3
    print("NS-3.3 -- the layer fires at the driver peak", flush=True)
    blind = make(ns_severity=2.0, ns_phase0=PEAK)
    t_blind = rollout(blind, 7, S)
    c = blind.chan
    check("dial_live_harmed_and_rotated_are_all_non_zero",
          c.n_dial_live > 0 and c.n_harmed > 0 and c.n_rot_sent > 0,
          "dial_live=%d harmed=%d rotated=%d controlled=%d of %d agent-steps"
          % (c.n_dial_live, c.n_harmed, c.n_rot_sent, c.n_controlled, 3 * S))
    ok, msg = identical(t_layer0, t_blind)
    check("sigma_2_actually_changes_the_trajectory", not ok, msg)
    _, _, _, _, info, _ = blind.step(random_actions(np.random.RandomState(1), 3))
    check("info_carries_the_panel", all(k in info[0] for k in (
        "ns_d", "ns_k", "ns_A", "ns_fit_gain", "ns_harmed", "score_reward")),
        "%d ns_* keys" % len([k for k in info[0] if str(k).startswith("ns_")]))
    check("layer_close_report_accepts_a_live_dial", c.close_report(tag="  [smoke]"))

    # ------------------------------------------------------------------ P-7.1
    print("P-7.1 -- pactoff is the blind arm bit for bit", flush=True)
    off = make(ns_severity=2.0, ns_phase0=PEAK, ns_pact=1, ns_trust="off")
    t_off = rollout(off, 7, S)
    ok, msg = identical(t_blind, t_off)
    check("pactoff_is_bit_identical_to_blind", ok, msg)
    check("pactoff_estimator_ran_but_corrected_nothing",
          off.chan.rows.sum() > 0 and float(np.abs(off.chan.c).max()) == 0.0,
          "rows=%d" % int(off.chan.rows.sum()))
    off.close()
    blind.close()

    # ------------------------------------------------------------------ II.6
    print("II.6 -- the oracle cancels the swerve exactly", flush=True)
    orc = make(ns_severity=2.0, ns_phase0=PEAK, ns_pact=1, ns_trust="fixed", ns_g_fixed=1.0,
               ns_oracle=1, ns_corr_clip=0.0)
    t_orc = rollout(orc, 7, S)
    ok, msg = identical(t_layer0, t_orc)
    check("oracle_at_sigma_2_is_bit_identical_to_sigma_0", ok, msg)
    check("oracle_saw_a_live_dial_and_enacted_nothing",
          orc.chan.n_dial_live > 0 and orc.chan.n_harmed == 0,
          "dial_live=%d harmed=%d" % (orc.chan.n_dial_live, orc.chan.n_harmed))
    orc.close()
    layer0.close()

    # ------------------------------------------------------------------ I.2
    print("I.2 -- a lone player reads exactly zero; the (B) control does not", flush=True)
    solo = make("academy_empty_goal_close", 1, ns_severity=3.0, ns_phase0=PEAK)
    rollout(solo, 7, 200)
    check("lone_player_is_untouched_at_sigma_3",
          solo.chan.n_dial_live == 0 and solo.chan.n_harmed == 0 and float(np.abs(solo.chan.d).max()) == 0.0,
          "dial_live=%d harmed=%d" % (solo.chan.n_dial_live, solo.chan.n_harmed))
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
          "fit_gain=%.3f beta_cos=%.3f beta_relerr=%.3f cond_psi=%.1f diverged=%d"
          % (float(np.mean(pc.fit_gain)), float(np.mean(pc.beta_cos)),
             float(np.mean(pc.beta_err)), pc.cond_psi, pc.n_diverged))
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
