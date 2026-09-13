"""The story test for SURFACE AREA UNDER DRIFT: does it bite, and can steering fix it?

NOT StarCraft II: a Lanchester battle between the map's rosters -- published hit
points and damage, no space.  What IS real: ``SurfaceMixin.step`` (the slot
lottery before the tick, sensing and prediction after it), the estimator, and the
policy's shift (``smac_ns.channel.steer``, which ``check_actor.py`` holds the torch
actor to).

    arms    blind   the host alone (= pactoff / mappo)
            pact    host + shift on the ESTIMATE  (trust at its 0.90 prior)
            oracle  host + shift on the TRUE excess
            det     cap's blind twin: the same deterministic allocation, every melee
                    unit on the host's top target -- isolates what knowing the
                    capacity is worth from what deterministic allocation is worth
            cap     THE CEILING: a centralized allocator with the true blocking
                    probabilities.  Melee units, in random order, take the host's
                    most-preferred target on which joining would be blocked less
                    than half the time.  Decentralized steering cannot beat it; if
                    it cannot beat blind, spreading has no headroom at all.

The host is a fixed focus-the-weakest policy.  Focus 8 wins ~95% of 3s5z battles
with the dial off -- the "solved" host the dial is aimed at.

Run::

    python -m harl.envs.smac.smac_sa.story --sigmas 0 1.5 2 3 --focus 8
"""

import argparse
import contextlib
import io
import sys
from multiprocessing import Pool

import numpy as np

from ..smac_ns import toy
from ..smac_ns.channel import steer
from ..smac_ns.coupling import UNIT_STATS, step_damage
from .layer import N_ACTIONS_NO_ATTACK as A0, SurfaceMixin

TRUST = 1.0 / (1.0 + np.exp(-2.2))
P_STOP = 0.2


class _ToyHost(toy.BattleHost):
    """BattleHost with a SMAC-shaped step: tick, then update_units."""

    def step(self, actions):
        acts = [int(a) for a in actions]
        toy.tick(self, acts, self.toy_ally_dmg, self.toy_enemy_tgt, self.toy_enemy_dmg)
        toy.update_units(self)
        return (None, None, None, None, [{} for _ in range(self.n_agents)], None)


class _Env(SurfaceMixin, _ToyHost):
    pass


def _host_logits(hp, focus, scale):
    lg = np.full(A0 + len(hp), -np.inf)
    alive = hp > 0
    att = -focus * hp[alive] / scale
    lg[A0:][alive] = att
    mm = att.max()
    lg[1] = mm + np.log(np.exp(att - mm).sum()) + np.log(P_STOP / (1.0 - P_STOP))
    return lg


def run(job):
    arm, sigma, seed, cycles, focus, channel, map_name, floor = job[:8]
    kappa = job[8] if len(job) > 8 else 1.0
    slot_order = job[9] if len(job) > 9 else "nearest"
    kw = dict(map_name=map_name, ns_severity=float(sigma), ns_augment=1,
              ns_oracle=int(arm == "oracle"), ns_seed=0, ns_steer_floor=floor,
              ns_slot_order=slot_order)
    with contextlib.redirect_stdout(io.StringIO()):
        e = _Env(kw)
    rng = np.random.RandomState(1000 + seed)
    period = e.driver.period
    horizon, warm = int(cycles * period), period
    e.toy_ally_dmg = step_damage(e.coupling.ally_names, e._step_mul)
    e.toy_enemy_dmg = step_damage(e.coupling.enemy_names, e._step_mul)
    K = A0 + e.n_enemies
    idx = np.arange(K)
    scale = float(max(UNIT_STATS[x]["life"] + UNIT_STATS[x]["shield"]
                      for x in e.coupling.enemy_names))
    acc = {p: dict(wins=0, n=0, num=0.0, den=0.0, gated=0, gate_n=0, mnum=0, mden=0)
           for p in ("peak", "dry", "all")}

    def new_battle():
        e.agents = toy.fresh(e.coupling.ally_names, 0)
        e.enemies = toy.fresh(e.coupling.enemy_names, 100)
        # two loose lines facing each other; units hold their positions (no space
        # in this toy) but distances differ, so slot order is well defined
        for u in e.agents.values():
            u.pos = toy.Pos(rng.uniform(0.0, 12.0), rng.uniform(0.0, 3.0))
        for u in e.enemies.values():
            u.pos = toy.Pos(rng.uniform(0.0, 12.0), rng.uniform(6.0, 9.0))
        e._ns_reset_state()
        return rng.randint(0, e.n_agents, e.n_enemies)

    enemy_tgt = new_battle()
    t_ep, A_sum, wet = 0, 0.0, 0
    while e.ns_clock < horizon:
        hp_e = np.array([(u.health + u.shield) if u.health > 0 else 0.0
                         for u in (e.enemies[k] for k in range(e.n_enemies))])
        al_alive = np.array([e.agents[i].health > 0 for i in range(e.n_agents)])
        acts = []
        if arm in ("cap", "det"):
            lg0 = _host_logits(hp_e, focus, scale)
            pref = [int(k) for k in np.argsort(-lg0[A0:]) if np.isfinite(lg0[A0 + k])]
            g_now = e.coupling.g_eff_all(e.ns_g())
            tgt = np.full(e.n_agents, -1, dtype=np.int64)
            order = rng.permutation(e.n_agents)
            for i in order:
                if not al_alive[i] or rng.rand() < P_STOP:
                    continue
                if not e.coupling.melee[i]:
                    p = np.exp(lg0 - lg0[np.isfinite(lg0)].max())
                    p[:A0] = 0.0
                    tgt[i] = int(rng.choice(len(p), p=p / p.sum())) - A0
                    continue
                choice = pref[0]
                for k in (pref if arm == "cap" else []):
                    trial = tgt.copy()
                    trial[i] = k
                    uk = e.coupling.loading(trial, al_alive.astype(float),
                                            (trial >= 0).astype(float),
                                            e._ns_dist())[i]
                    pk = float(e.coupling.p_block(e.coupling.excess(uk, g_now[i, k])))
                    if pk < 0.5:
                        choice = k
                        break
                tgt[i] = choice
            acts = [0 if not al_alive[i] else (A0 + int(tgt[i]) if tgt[i] >= 0 else 1)
                    for i in range(e.n_agents)]
        for i in range(e.n_agents if arm not in ("cap", "det") else 0):
            if not al_alive[i]:
                acts.append(0)
                continue
            lg = _host_logits(hp_e, focus, scale)
            avail = np.isfinite(lg)
            if arm != "blind":
                valid = avail & (idx >= A0)
                lg = np.where(avail, steer(np.where(avail, lg, 0.0), e.ns_cost[i],
                                           TRUST, kappa, valid, floor, channel), -np.inf)
            p = np.exp(lg - lg[avail].max())
            acts.append(int(rng.choice(K, p=p / p.sum())))
        for k in range(e.n_enemies):
            if enemy_tgt[k] < 0 or e.agents[int(enemy_tgt[k])].health <= 0:
                ai = np.where(al_alive)[0]
                enemy_tgt[k] = int(rng.choice(ai)) if ai.size else -1
        shoots = (hp_e > 0) & (rng.rand(e.n_enemies) >= P_STOP)
        e.toy_enemy_tgt = np.where(shoots, enemy_tgt, -1)
        e.step(acts)

        t_h = e.ns_t_now
        A = float(e.driver.A(t_h))
        ph = "dry" if e.driver.is_placebo(t_h) else ("peak" if A >= 0.5 else None)
        t_ep += 1
        A_sum += A
        wet += int(not e.driver.is_placebo(t_h))
        if t_h >= warm:
            for key in ([ph] if ph else []) + ["all"]:
                a = acc[key]
                a["num"] += e.ns_harm_num
                a["den"] += e.ns_harm_den
                a["gated"] += e.ns_step_gated
                a["gate_n"] += e.ns_step_gate_n
                a["mnum"] += int(np.sum(e.ns_blocked_mask & e.coupling.melee))
                a["mden"] += int(np.sum((e.ns_fired > 0) & e.coupling.melee))
        en_alive = any(u.health > 0 for u in e.enemies.values())
        al_any = any(u.health > 0 for u in e.agents.values())
        if (not en_alive) or (not al_any) or t_ep >= e.episode_limit:
            won = int(not en_alive and al_any)
            eph = "dry" if wet == 0 else ("peak" if A_sum / t_ep >= 0.5 else None)
            if t_h >= warm:
                for key in ([eph] if eph else []) + ["all"]:
                    acc[key]["wins"] += won
                    acc[key]["n"] += 1
            enemy_tgt = new_battle()
            t_ep, A_sum, wet = 0, 0.0, 0
    return (arm, sigma, seed), acc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="3s5z")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 1.5, 2.0, 3.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--focus", type=float, default=8.0)
    ap.add_argument("--channel", default="logratio", choices=["logratio", "zscore"])
    ap.add_argument("--floor", type=float, default=0.01)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--slot-order", default="nearest", choices=["nearest", "shared"])
    ap.add_argument("--arms", nargs="+", default=["blind", "pact", "oracle"])
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args(argv)

    jobs = [(arm, s, seed, a.cycles, a.focus, a.channel, a.map, a.floor, a.kappa,
             a.slot_order)
            for s in a.sigmas for seed in a.seeds for arm in a.arms
            if not (s == 0.0 and arm != "blind")]       # identical by construction
    out = []
    with Pool(min(a.workers, len(jobs))) as pool:
        for key, acc in pool.imap_unordered(run, jobs):
            out.append((key, acc))
            pk, dr, al = acc["peak"], acc["dry"], acc["all"]
            print("[job %2d/%d] %-6s sigma=%.1f seed=%d  win_all=%.3f (n=%d) "
                  "win_peak=%s win_dry=%s melee_blocked_peak=%s"
                  % (len(out), len(jobs), key[0], key[1], key[2],
                     al["wins"] / max(1, al["n"]), al["n"],
                     "%.3f" % (pk["wins"] / pk["n"]) if pk["n"] else "nan",
                     "%.3f" % (dr["wins"] / dr["n"]) if dr["n"] else "nan",
                     "%.3f" % (pk["mnum"] / pk["mden"]) if pk["mden"] else "nan"),
                  flush=True)

    pooled = {}
    for (arm, sigma, seed), acc in out:
        k = (sigma, arm)
        if k not in pooled:
            pooled[k] = {p: dict(v) for p, v in acc.items()}
        else:
            for p in acc:
                for f in acc[p]:
                    pooled[k][p][f] += acc[p][f]

    def r(x, f, n):
        return x[f] / x[n] if x[n] else float("nan")

    print("=" * 104)
    print("SURFACE AREA story -- %s, focus %.1f, channel %s, kappa %.1f, floor %.3f, "
          "slots %s, %d seeds x %d cycles (first dropped)"
          % (a.map, a.focus, a.channel, a.kappa, a.floor, a.slot_order,
             len(a.seeds), a.cycles))
    print("=" * 104)
    print("%-5s %-7s | %-7s %-8s %-8s %-6s | %-10s %-10s | %-9s %-9s | %s"
          % ("sigma", "arm", "win_all", "win_peak", "win_dry", "gap", "blk_peak",
             "melee_pk", "gate_pk", "gate_dry", "episodes pk/dry/all"))
    order = ("blind", "pact", "oracle", "det", "cap")
    for k in sorted(pooled, key=lambda k: (k[0], order.index(k[1]))):
        pk, dr, al = pooled[k]["peak"], pooled[k]["dry"], pooled[k]["all"]
        wp, wd = r(pk, "wins", "n"), r(dr, "wins", "n")
        print("%-5.1f %-7s | %-7.3f %-8.3f %-8.3f %-6.3f | %-10.4f %-10.4f | %-9s "
              "%-9s | %d/%d/%d"
              % (k[0], k[1], r(al, "wins", "n"), wp, wd, wd - wp, r(pk, "num", "den"),
                 r(pk, "mnum", "mden"),
                 "-" if k[1] == "blind" else "%.2f" % r(pk, "gated", "gate_n"),
                 "-" if k[1] == "blind" else "%.2f" % r(dr, "gated", "gate_n"),
                 pk["n"], dr["n"], al["n"]))
    print("-" * 104)
    print("blk = damage-weighted share of attack orders that found no slot; melee_pk = "
          "the same for melee only, at the peak.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
