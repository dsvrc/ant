"""The gates.  Run these BEFORE any method code, and COMMIT the output.

`NS_FORM_SPEC` Part D: *"Every one of these exists because skipping it cost a
run."*  And E.1 step 4: *"Compute the ceiling decomposition (Part C) BEFORE
writing any method code.  If the coordination gap is small, stop and pick another
environment."*

Two halves:

  OFFLINE (no StarCraft II, seconds)   -- the Part-C ceiling, the dial's B.1/B.4
                                         certificates, G1 and G2.
  LIVE    (needs StarCraft II)         -- G0, G0b, G3, G4a, G4b, G5, G6, G7 and
                                         the sigma=1 ANCHOR measurement.

Usage::

    python -m harl.envs.smac.fc.certificates --offline
    python -m harl.envs.smac.fc.certificates --map 3s5z --episodes 12

The live half runs a strong REFERENCE controller and a PRIVILEGED one (D.1).  A
strawman reference makes every gate pass for the wrong reason, and POWER's first
privileged controller was purely reactive and LOST to do-nothing (242 vs 316),
which made G4 unpassable for reasons having nothing to do with the dial.  So the
reference here focus-fires and closes, and the privileged controller additionally
knows the exact deficit AND the driver model, so it can pre-empt the trough
instead of only reacting to it.
"""

import argparse
import json
import sys

import numpy as np

from . import operator as opmod
from .driver import Driver, assert_dial, dial, driver, is_placebo
from .severity_env import FormationCongestionEnv, N_ACTIONS_NO_ATTACK


# --------------------------------------------------------------------------- #
# PART C -- the ceiling decomposition                (NS_FORM_SPEC C.1, C.2)
# --------------------------------------------------------------------------- #
def ceiling_from_state(W, W_env, ally_names, enemy_names, pos_a, pos_e,
                       alive_a, alive_e, phi, k_scale, sigma=1.0, A=1.0,
                       knee=0.35, depth=0.60, floor=0.25, prox_len=opmod.PROX_LEN):
    """The four numbers of C.1, from the declared operator and one observed state.

    No training and no method::

        Delta_i(t) = u_i(t) * (1 - g(A(t)))          the excess over the sigma=0 run

    and every unit of it traces to a contributor that partitions by WHO CAN MOVE
    IT.  On this environment W is zero-diagonal AND the agent's own body does not
    obstruct its own corridor, so the ``Delta_own`` class -- 76.9% of POWER's
    excess -- is EXACTLY ZERO here and is reported as such.  What is left is a
    two-way split:

        irreducible        L^fixed: enemy bodies and terrain, nobody controls them
        COORDINATION GAP   peers through W, amplified by 1/g   <- what PACT claims

    C.2: attribute in proportion to signed contribution, counting only what could
    actually help -- a contributor already at zero exertion offers no headroom.
    """
    r_a = opmod.radii(ally_names)
    r_e = opmod.radii(enemy_names)
    prox_aa, cone_aa = opmod.kernels(pos_a, pos_a, r_a, r_a, prox_len)
    L_peer = opmod.loading(W, prox_aa, cone_aa, phi, alive_a)
    if len(enemy_names):
        prox_ae, cone_ae = opmod.kernels(pos_a, pos_e, r_a, r_e, prox_len)
        L_fix = opmod.loading(W_env, prox_ae, cone_ae,
                              np.full(len(enemy_names), opmod.PHI_ENEMY), alive_e)
    else:
        L_fix = np.zeros_like(L_peer)
    phi_max = float(np.max(phi)) if np.size(phi) else 1.0
    K0 = k_scale * (np.asarray(W).sum(1) + np.asarray(W_env).sum(1)) * phi_max
    g = float(dial(sigma, A, knee, depth, floor))
    tot_dir = (L_peer + L_fix) / (K0 * g)[:, None]
    bind = np.argmax(tot_dir, axis=1)
    idx = np.arange(len(ally_names))
    Rp = L_peer[idx, bind]
    Rf = L_fix[idx, bind]
    denom = Rp + Rf
    live = (np.asarray(alive_a) > 0) & (denom > 0)
    if not live.any():
        return dict(irreducible=float("nan"), own=0.0,
                    coordination_gap=float("nan"), decentralized_ceiling=float("nan"),
                    delta_total=0.0, g=g, sigma=float(sigma), A=float(A))
    peer = float(np.sum(Rp[live]) / np.sum(denom[live]))
    fixed = float(np.sum(Rf[live]) / np.sum(denom[live]))
    delta_tot = float(np.mean((tot_dir[idx, bind] * (1.0 - g))[live]))
    return dict(
        irreducible=fixed,
        own=0.0,                       # structurally zero here -- see the docstring
        coordination_gap=peer,
        decentralized_ceiling=1.0 - fixed,
        non_coordinating_ceiling=1.0 - fixed - peer,
        delta_total=delta_tot,
        g=g, sigma=float(sigma), A=float(A),
    )


def ceiling_sweep(map_name="3s5z", n_agents=8, n_enemies=8, step_mul=8,
                  k_scale=0.35, sigmas=(0.5, 1.0, 1.5, 2.0), samples=400, seed=0,
                  spread=4.0):
    """C.3's table: the decomposition across severities, with NO training.

    Geometry is sampled from a plausible engagement (both squads clustered around
    a contact line), so the numbers describe the environment rather than any one
    policy.  ``--live`` in the gate suite recomputes them on real rollouts.
    """
    ally = opmod.composition(map_name, n_agents)
    enemy = opmod.enemy_composition(map_name, n_enemies)
    W = opmod.build_W(ally, ally, step_mul)
    W_env = opmod.build_W(ally, enemy, step_mul, zero_diagonal=False)
    rng = np.random.RandomState(seed)
    rows = []
    for s in sigmas:
        acc = []
        for _ in range(samples):
            pa = rng.randn(n_agents, 2) * spread + np.array([16.0, 16.0])
            pe = rng.randn(n_enemies, 2) * spread + np.array([16.0, 16.0 + spread])
            alive_a = (rng.rand(n_agents) < 0.8).astype(float)
            alive_e = (rng.rand(n_enemies) < 0.8).astype(float)
            if alive_a.sum() < 2:
                alive_a[:2] = 1.0
            phi = alive_a * (1.0 + 0.5 * (rng.rand(n_agents) < 0.55))
            A = float(rng.rand())
            acc.append(ceiling_from_state(W, W_env, ally, enemy, pa, pe,
                                          alive_a, alive_e, phi, k_scale,
                                          sigma=s, A=A))
        def m(k):
            v = np.array([x[k] for x in acc], dtype=float)
            return float(np.nanmean(v))
        rows.append(dict(sigma=float(s), irreducible=m("irreducible"),
                         own=0.0, coordination_gap=m("coordination_gap"),
                         decentralized_ceiling=m("decentralized_ceiling"),
                         delta_total=m("delta_total")))
    return rows


def temporal_split(deltas, g_series, episode_ids, warmup=200):
    """How much of the excess a purely LOCAL controller can already remove.

    C.1 splits the excess by WHO controls the source.  On this environment that
    split flatters the method, because W is zero-diagonal and almost everything is
    a peer.  The honest question for a coordination claim is the TEMPORAL one:
    the sensor reports the PAST (A.4), so how much of Delta(t+1) does the best
    local predictor -- last step's reading, rescaled by the KNOWN driver model --
    already explain?  Whatever is left is the only thing peer anticipation can
    buy, and it is what ``fit_gain`` measures online.

    *** MEASURE IT PER AGENT, NOT ON THE SQUAD MEAN. ***  Averaging Delta over the
    live squad removes exactly the cross-agent variation the peer channels exist
    to predict, so a squad-mean series is far smoother than anything PACT ever
    sees and reports a local R^2 that is much too high.  ``deltas`` is therefore
    (T, n) and every agent contributes its own lagged pair; pairs that straddle an
    episode boundary or a death are dropped rather than lagged across them.

    Returns the local predictor's R^2 and the residual share.  Guarded with NaN.
    """
    d = np.asarray(deltas, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    g = np.asarray(g_series, dtype=float)
    ep = np.asarray(episode_ids)
    if d.shape[0] <= warmup + 2:
        return dict(local_r2=float("nan"), residual_share=float("nan"), n=0)
    y = d[warmup + 1:]
    prev = d[warmup:-1]
    same_ep = (ep[warmup + 1:] == ep[warmup:-1])[:, None]
    gn, gp = 1.0 - g[warmup + 1:], 1.0 - g[warmup:-1]
    # Guard the ratio with NaN, never an epsilon: inside the placebo regime the
    # disturbance is genuinely 0 and the ratio is meaningless.
    ratio = np.where(gp > 1e-6, gn / np.maximum(1e-12, gp), np.nan)[:, None]
    pred = np.where(np.isfinite(ratio), prev * ratio, prev)
    ok = same_ep & np.isfinite(y) & np.isfinite(pred)
    yv, pv = y[ok], pred[ok]
    if yv.size < 100:
        return dict(local_r2=float("nan"), residual_share=float("nan"), n=int(yv.size))
    sst = float(np.var(yv))
    if not np.isfinite(sst) or sst <= 0.0:
        return dict(local_r2=float("nan"), residual_share=float("nan"), n=int(yv.size))
    r2 = 1.0 - float(np.mean((yv - pv) ** 2)) / sst
    return dict(local_r2=float(r2), residual_share=float(max(0.0, 1.0 - r2)),
                n=int(yv.size))


# --------------------------------------------------------------------------- #
# Controllers for the live gates                     (NS_FORM_SPEC D.1)
# --------------------------------------------------------------------------- #
def _enemy_centroid(env):
    pts = [(u.pos.x, u.pos.y) for u in (getattr(env, "enemies", {}) or {}).values()
           if u.health > 0]
    return np.mean(pts, axis=0) if pts else None


def reference_action(env, i, avail):
    """The REFERENCE controller: focus fire when anything is in range, otherwise
    close on the enemy line.  Strong on purpose (D.1) -- it is what SMAC's own
    built-in heuristic does plus focus fire, so a gate it passes is not passing
    because the control was a strawman."""
    atk = np.where(np.asarray(avail[N_ACTIONS_NO_ATTACK:]) > 0)[0]
    if atk.size:
        hp = [(env.enemies[e].health + env.enemies[e].shield, e) for e in atk
              if e in env.enemies]
        if hp:
            return N_ACTIONS_NO_ATTACK + int(min(hp)[1])
    c = _enemy_centroid(env)
    u = env.agents.get(i, None)
    if c is None or u is None:
        return 1 if avail[1] else 0
    dx, dy = c[0] - u.pos.x, c[1] - u.pos.y
    order = ([2, 3] if abs(dy) >= abs(dx) else [4, 5])
    order = ([2 if dy > 0 else 3] + [4 if dx > 0 else 5]) if abs(dy) >= abs(dx) \
        else ([4 if dx > 0 else 5] + [2 if dy > 0 else 3])
    for a in order:
        if avail[a]:
            return a
    return 1 if avail[1] else 0


def _rollout(fc, controller, privileged=False, max_steps=100000, episodes=8,
             warm_window=40):
    """Run `episodes` episodes and return the measured facts the gates need."""
    ep_ret, ep_len, wins = [], [], 0
    early_u, early_delta, deltas, gs, eps_id, moves, acts = [], [], [], [], [], 0, 0
    u_hat = np.zeros(fc.n_agents)
    for _ep in range(int(episodes)):
        fc.reset()
        done, ret, t = False, 0.0, 0
        while not done and t < max_steps:
            avail = fc.env.get_avail_actions()
            acts_t = [controller(fc.env, i, avail[i]) for i in range(fc.n_agents)]
            if privileged:
                # PRIVILEGED IN STRATEGY, not only in information (D.1): the exact
                # channel inverse on the TRUE deficit, plus anticipation -- it knows
                # the driver model, so it pre-empts the trough instead of chasing it.
                g_next = float(fc.driver.g()[0])
                pre = np.clip(u_hat * (1.0 - g_next), 0.0, fc.harm_max)
                fc.set_command(np.clip(1.0 / np.maximum(0.25, 1.0 - pre), 0.25, 4.0))
            else:
                fc.set_command(np.ones(fc.n_agents))
            out = fc.step(np.array(acts_t).reshape(-1, 1))
            rew, dn, infos = out[2], out[3], out[4]
            ret += float(np.asarray(rew).reshape(-1)[0])
            gnow = float(fc.g)
            if gnow < 1.0:
                u_hat = 0.7 * u_hat + 0.3 * (fc.delta / max(1e-9, 1.0 - gnow))
            # PER AGENT (see temporal_split): a squad mean is far smoother than
            # anything the compensator ever sees.  Dead units contribute NaN, which
            # the split drops rather than lagging across a death.
            deltas.append(np.where(fc._alive > 0, fc.delta, np.nan))
            gs.append(gnow)
            eps_id.append(_ep)
            if t < warm_window:
                # G0b: a FIXED EARLY WINDOW.  Whole-episode means have survivorship
                # bias -- a run that dies early only averages its calm opening,
                # which makes a HIGHER severity look CALMER.
                live = fc._alive > 0
                if live.any():
                    early_u.append(float(np.mean(fc.u[live])))
                    early_delta.append(float(np.mean(fc.delta[live])))
            moves += int(np.sum([2 <= a < N_ACTIONS_NO_ATTACK for a in acts_t]))
            acts += int(np.sum(fc._alive > 0))
            done = bool(np.all(np.asarray(dn)))
            if done and isinstance(infos[0], dict):
                wins += int(bool(infos[0].get("won", False)))
            t += 1
        ep_ret.append(ret)
        ep_len.append(t)
    ts = temporal_split(np.asarray(deltas), gs, eps_id)
    return dict(
        ret=float(np.mean(ep_ret)), ret_sd=float(np.std(ep_ret)),
        ep_len=float(np.mean(ep_len)), win=float(wins) / max(1, episodes),
        early_u=float(np.nanmean(early_u)) if early_u else float("nan"),
        early_delta=float(np.nanmean(early_delta)) if early_delta else float("nan"),
        delta_mean=float(np.nanmean(deltas)) if deltas else float("nan"),
        move_frac=(float(moves) / acts if acts else float("nan")),
        odom_err=float(fc.diagnostics()["fc_odom_err"]),
        phi_var=float(fc.exertion.variation()),
        local_r2=ts["local_r2"], residual_share=ts["residual_share"],
        n_pairs=int(ts["n"]), n_ep=int(episodes),
        # the return's own spread, so a gate read off 8 episodes cannot be mistaken
        # for a measurement: SMAC episode returns are heavy-tailed and 8 of them
        # will separate almost any two arms by chance.
        ret_se=float(np.std(ep_ret) / max(1.0, np.sqrt(len(ep_ret)))),
    )


def _make(map_name, sigma, seed, extra=None):
    from ..StarCraft2_Env import StarCraft2Env
    args = {"map_name": map_name, "state_type": "FP"}
    env = StarCraft2Env(args)
    env.seed(seed)
    a = {"ns_severity": float(sigma), "ns_eval": 1}
    a.update(extra or {})
    return FormationCongestionEnv(env, a)


def live_gates(map_name="3s5z", sigmas=(0.0, 0.5, 1.0, 1.5), episodes=8, seed=0):
    """G0, G0b, G3, G4a, G4b, G5, G6, G7 plus the sigma=1 anchor measurement."""
    rows = {}
    for s in sigmas:
        fc = _make(map_name, s, seed)
        ref = _rollout(fc, reference_action, False, episodes=episodes)
        fc.close()
        fc = _make(map_name, s, seed)
        priv = _rollout(fc, reference_action, True, episodes=episodes)
        fc.close()
        rows[float(s)] = dict(ref=ref, priv=priv)
        print("[sigma=%.2f] ref ret=%.1f+-%.1f win=%.2f len=%.0f | priv ret=%.1f+-%.1f "
              "win=%.2f | u_early=%.3f delta_early=%.3f move_frac=%.2f local_r2=%.3f"
              % (s, ref["ret"], ref["ret_se"], ref["win"], ref["ep_len"],
                 priv["ret"], priv["ret_se"], priv["win"],
                 ref["early_u"], ref["early_delta"], ref["move_frac"],
                 ref["local_r2"]))

    out, ok = {}, True
    base = rows[min(rows)]
    top = rows[max(rows)]

    def gate(name, cond, detail):
        nonlocal ok
        print("  [%s] %-42s %s" % ("PASS" if cond else "FAIL", name, detail))
        out[name] = dict(pass_=bool(cond), detail=detail)
        ok = ok and bool(cond)

    gate("G0  liveness (the physics changed)",
         top["ref"]["early_delta"] > 1e-6,
         "delta_early = %.4f at sigma=%.2f" % (top["ref"]["early_delta"], max(rows)))
    mono = all(rows[a]["ref"]["early_u"] <= rows[b]["ref"]["early_u"] + 1e-6
               for a, b in zip(sorted(rows), sorted(rows)[1:]))
    gate("G0b monotone on a FIXED early window", mono,
         " -> ".join("%.3f" % rows[s]["ref"]["early_u"] for s in sorted(rows)))
    drop = base["ref"]["ret"] - top["ref"]["ret"]
    se = float(np.hypot(base["ref"]["ret_se"], top["ref"]["ret_se"]))
    gate("G3  it hurts (reference falls with sigma)", drop > 2.0 * se,
         "ret %.1f -> %.1f  (drop %.1f, 2se %.1f)%s"
         % (base["ref"]["ret"], top["ref"]["ret"], drop, 2 * se,
            "" if drop > 2 * se else "  <- INSIDE THE NOISE: raise --episodes"))
    g4a = top["priv"]["ret"] / max(1e-9, base["priv"]["ret"])
    gate("G4a capacity  priv(s)/priv(0) >= 0.95", g4a >= 0.95,
         "%.3f  (D.2: a capacity-reducing dial FAILS this -- report against the "
         "ceiling measured AT that sigma)" % g4a)
    ref1 = rows.get(1.0, top)
    g4b = ref1["priv"]["ret"] / max(1e-9, ref1["ref"]["ret"])
    g4b_se = float(np.hypot(ref1["priv"]["ret_se"], ref1["ref"]["ret_se"])
                   / max(1e-9, ref1["ref"]["ret"]))
    gate("G4b coordination priv/ref >= 1.30", g4b >= 1.30,
         "%.3f +- %.3f  <- THE GATE THAT DECIDES WHETHER TO TRAIN AT ALL: a "
         "PERFECT oracle on the true deficit gains this much, so no method can "
         "gain more" % (g4b, g4b_se))
    gate("G5  coupling rises toward the limit",
         top["ref"]["early_u"] > base["ref"]["early_u"],
         "u_early %.3f -> %.3f" % (base["ref"]["early_u"], top["ref"]["early_u"]))
    # D.3, read literally.  The harm channel only reaches a unit that is MOVING,
    # so move_frac is this environment's version of "are the agents actually
    # acting?" -- and a controller that plants and shoots feels no dial at all.
    # The bar is a majority; anything much below it means the NS is being measured
    # on a small minority of decisions and G4b will fail for that reason alone.
    gate("G7  the channel reaches a majority of decisions",
         top["ref"]["move_frac"] >= 0.5,
         "move_frac = %.2f -- the throttle only touches move orders, so at this "
         "rate %.0f%% of decisions never feel the dial"
         % (top["ref"]["move_frac"], 100 * (1 - top["ref"]["move_frac"])))
    gate("anchor  sigma=1 vs the game's own congestion",
         np.isfinite(top["ref"]["odom_err"]),
         "odometric reconstruction error = %.4f (the sensor is physical)"
         % top["ref"]["odom_err"])
    gate("temporal split: peer anticipation has room",
         np.isfinite(ref1["ref"]["residual_share"])
         and ref1["ref"]["residual_share"] > 0.02,
         "local predictor R2 = %.3f, residual = %.1f%%"
         % (ref1["ref"]["local_r2"], 100 * ref1["ref"]["residual_share"]))
    return ok, out, rows


def placebo_gate(map_name="3s5z", sigmas=(0.0, 0.5, 1.0, 2.0), steps=60, seed=0):
    """G6.  Freeze the driver INSIDE its inert regime and confirm every severity
    row is byte-identical.  A reviewer alleging a rigged knob then has to explain
    why the rig switches itself off during the consolidate phase."""
    sig = []
    for s in sigmas:
        fc = _make(map_name, s, seed, {"ns_freeze": 0.0})   # A = 0 <= knee: inert
        fc.reset()
        trace = []
        for _ in range(steps):
            avail = fc.env.get_avail_actions()
            a = [reference_action(fc.env, i, avail[i]) for i in range(fc.n_agents)]
            out = fc.step(np.array(a).reshape(-1, 1))
            trace.append((float(np.asarray(out[2]).reshape(-1)[0]),
                          tuple(np.round(fc.stride, 12))))
            if bool(np.all(np.asarray(out[3]))):
                break
        fc.close()
        sig.append(trace)
    same = all(sig[k] == sig[0] for k in range(len(sig)))
    print("  [%s] G6  placebo regime byte-identical across sigma  %s"
          % ("PASS" if same else "FAIL", [len(x) for x in sig]))
    return same


#: Candidate maps for --scan, chosen on ONE criterion: how much of the outcome
#: movement decides.  The harm channel is a stride throttle, so a map where the
#: fight is a stand-and-shoot slugfest cannot transmit it to reward however large
#: the dial is, and G4b will fail for that reason alone.
#:   kiting maps      -- movement IS the task; a slowed unit dies
#:   large formations -- many bodies over one medium, which is C.4's knob for
#:                       widening the coordination gap (more agents, less local
#:                       authority per agent)
SCAN_MAPS = ["3s5z", "3s_vs_5z", "3s_vs_4z", "2c_vs_64zg", "27m_vs_30m",
             "10m_vs_11m", "MMM2", "1c3s5z"]


def scan(maps, episodes=20, seed=0, sigma=1.0):
    """Pick the environment on the gates, not on the story (NS_FORM_SPEC C.3).

    *"Use it to choose the environment, not to excuse the outcome.  If the PEER
    column is small, that environment is a poor showcase for a coordination method
    however good the method is.  Compute this FIRST, across candidate
    environments, and lead with whichever has the largest coordination gap."*

    G4b is the decisive column: it is what a PERFECT oracle on the true deficit
    buys over the blind reference, so no method can beat it.  ``move_frac`` is the
    explanation column -- the throttle only touches move orders.
    """
    print("map scan at sigma=%.2f, %d episodes/arm -- G4b is the number that "
          "decides whether training is worth it" % (sigma, episodes))
    print("  %-13s %-10s %-11s %-13s %-13s %-9s %s"
          % ("map", "move_frac", "delta_mean", "ref ret", "priv ret", "G4b",
             "local_r2"))
    rows = []
    for m in maps:
        try:
            fc0 = _make(m, 0.0, seed)
            r0 = _rollout(fc0, reference_action, False, episodes=episodes)
            fc0.close()
            fc = _make(m, sigma, seed)
            ref = _rollout(fc, reference_action, False, episodes=episodes)
            fc.close()
            fc = _make(m, sigma, seed)
            priv = _rollout(fc, reference_action, True, episodes=episodes)
            fc.close()
        except Exception as exc:                       # a map that will not build
            print("  %-13s SKIPPED (%s)" % (m, exc))
            continue
        g4b = priv["ret"] / max(1e-9, ref["ret"])
        rows.append(dict(map=m, move_frac=ref["move_frac"],
                         delta_mean=ref["delta_mean"], ret0=r0["ret"],
                         ref=ref["ret"], ref_se=ref["ret_se"], priv=priv["ret"],
                         priv_se=priv["ret_se"], g4b=float(g4b),
                         local_r2=ref["local_r2"],
                         residual=ref["residual_share"],
                         hurt=(r0["ret"] - ref["ret"]),
                         hurt_2se=2 * float(np.hypot(r0["ret_se"], ref["ret_se"]))))
        print("  %-13s %-10.2f %-11.4f %-13s %-13s %-9.3f %.3f"
              % (m, ref["move_frac"], ref["delta_mean"],
                 "%.1f+-%.1f" % (ref["ret"], ref["ret_se"]),
                 "%.1f+-%.1f" % (priv["ret"], priv["ret_se"]), g4b,
                 ref["local_r2"]))
    if rows:
        best = max(rows, key=lambda r: r["g4b"])
        print()
        print("  best G4b: %s at %.3f%s" % (best["map"], best["g4b"],
              "  -- CLEARS the 1.30 bar" if best["g4b"] >= 1.30
              else "  -- still below the 1.30 bar"))
        print("  G3 on that map: sigma=0 ret %.1f -> sigma=%.2f ret %.1f "
              "(drop %.1f, 2se %.1f)"
              % (best["ret0"], sigma, best["ref"], best["hurt"], best["hurt_2se"]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="3s5z")
    ap.add_argument("--scan", nargs="?", const=",".join(SCAN_MAPS), default=None,
                    help="comma-separated maps to rank by G4b; bare --scan uses "
                         "the built-in candidate list")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_scale", type=float, default=0.35)
    ap.add_argument("--offline", action="store_true",
                    help="skip everything that needs StarCraft II")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    if args.scan:
        maps = [m.strip() for m in args.scan.split(",") if m.strip()]
        rows = scan(maps, episodes=args.episodes, seed=args.seed, sigma=args.sigma)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(dict(scan=rows, sigma=args.sigma,
                               episodes=args.episodes), f, indent=2, default=float)
            print("wrote %s" % args.out)
        return 0

    print("=" * 74)
    print("OFFLINE certificates -- no StarCraft II needed")
    print("=" * 74)
    ally = opmod.composition(args.map, len(opmod.MAP_COMPOSITION.get(args.map, [])) or 8)
    W = opmod.build_W(ally, ally, 8)
    print(opmod.banner(opmod.report(W, ally), "W(ally,ally) %s" % args.map))
    print("  G1  W[i,i] == 0 : %s" % np.all(np.diag(W) == 0.0))
    d = assert_dial()
    print("  G2  g(0, A) == 1 over the whole domain : True")
    print("  B.4 placebo fraction of the cycle      : %.0f%%" % (100 * d["placebo_frac"]))
    print()
    print("PART C -- the ceiling, computed BEFORE any method (C.3's table)")
    print("  %-7s %-13s %-11s %-16s %s" % ("sigma", "irreducible", "own(free)",
                                           "PEER(coord)", "decentr ceiling"))
    rows = ceiling_sweep(args.map, len(ally), len(opmod.enemy_composition(args.map, 8)),
                         k_scale=args.k_scale, seed=args.seed)
    for r in rows:
        print("  %-7.2f %-13s %-11s %-16s %s"
              % (r["sigma"], "%.1f%%" % (100 * r["irreducible"]), "0.0%",
                 "%.1f%%" % (100 * r["coordination_gap"]),
                 "%.1f%%" % (100 * r["decentralized_ceiling"])))
    print("  NOTE: `own` is structurally 0 -- W is zero-diagonal AND a unit's own")
    print("  body does not obstruct its own corridor, so unlike POWER (76.9% own)")
    print("  every controllable unit of the excess is a PEER unit.  That flatters")
    print("  the source split, so read the TEMPORAL split from the live gates too:")
    print("  it is the stale local sensor, not the source, that bounds a local fix.")

    result = dict(map=args.map, ceiling=rows, dial=d)
    if not args.offline:
        print()
        print("=" * 74)
        print("LIVE gates -- StarCraft II required")
        print("=" * 74)
        ok, gates, raw = live_gates(args.map, episodes=args.episodes, seed=args.seed)
        pl = placebo_gate(args.map, seed=args.seed)
        result["gates"] = gates
        result["placebo"] = bool(pl)
        result["all_pass"] = bool(ok and pl)
        print()
        print("ALL GATES PASS" if (ok and pl) else "SOME GATES FAILED -- fix before "
              "running any method (D: every gate exists because skipping it cost a run)")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=float)
        print("wrote %s" % args.out)
    return 0 if result.get("all_pass", True) else 1


if __name__ == "__main__":
    sys.exit(main())
