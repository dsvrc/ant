"""The story test: does the dial hurt a focus-fire host, and does steering win it back?

NOT StarCraft II.  A Lanchester battle between the 3s5z rosters -- published hit
points and per-step damage (coupling.UNIT_STATS), no space, no movement, no
cooldowns -- stepped in the engine's order by ``toy.py``.  What IS real: the
severity layer (``SeverityMixin``, unmodified), its estimator and gate, and the
steering shift (``channel.steer`` -- ``pact1_core.steer_logits`` over the
attackable options, plus the floor -- which ``check_actor.py`` holds the torch
actor to).

The host is a FIXED focus-the-weakest policy, not a learner.  So this bounds what a
focus-fire policy loses to the dial and what the steering channel gives back; it
says nothing about whether MAPPO learns around the dial.  Use it to choose the
severity before spending a cluster run -- never as a result.

    arms     blind   the host alone                (= pactoff / mappo)
             pact    host + shift on the ESTIMATE  (trust at its 0.90 prior)
             oracle  host + shift on the TRUE excess
    configs  fixed   this layer as shipped
             first   the first 3.2M-step run, reproduced: capacity = max hp,
                     period 150, no floor, the shift z-scoring STOP against the
                     attacks, and the restore that wrote pre-tick values

Run::

    python -m harl.envs.smac.smac_ns.story
    python -m harl.envs.smac.smac_ns.story --sigmas 3 --cycles 6 --seeds 0 1 2
"""

import argparse
import contextlib
import io
import sys
from multiprocessing import Pool

import numpy as np

from . import toy
from .coupling import UNIT_STATS, step_damage
from .layer import SeverityMixin
from .channel import steer

A0 = 6
TRUST = 1.0 / (1.0 + np.exp(-2.2))       # P-5.1's prior, 0.9025
FOCUS = 4.0                               # host preference for the weakest target
P_STOP = 0.2                              # host's non-attack probability


class _Env(SeverityMixin, toy.BattleHost):
    pass


def _first_run_restore(self, tgt, fired, exc, post, landed):
    """The first run's restore, reproduced for the comparison: aimed waste, and
    absolute values built from the PRE-tick protos in ``self.enemies``."""
    waste = np.zeros(self.n_enemies)
    for i in range(self.n_agents):
        if fired[i] > 0 and tgt[i] >= 0:
            waste[tgt[i]] += self.coupling.dmg[i] * exc[i] / (1.0 + exc[i])
    restored = np.zeros(self.n_enemies)
    for e in range(self.n_enemies):
        if waste[e] <= 1e-9 or e not in post:
            continue
        pre = self.enemies.get(e, None)
        if pre is None or pre.health <= 0:
            continue
        nm = self.coupling.enemy_names[e]
        max_life, max_sh = float(UNIT_STATS[nm]["life"]), float(UNIT_STATS[nm]["shield"])
        now = post[e]
        before = now.health + now.shield
        give = float(waste[e])
        if max_sh > 0 and pre.shield < max_sh:
            add = min(give, max_sh - pre.shield)
            now.shield = pre.shield + add          # SET from the PRE-tick value
            give -= add
        if give > 1e-9 and pre.health < max_life:
            add = min(give, max_life - pre.health)
            now.health = pre.health + add
        restored[e] = (now.health + now.shield) - before
    self.ns_n_restored += float(np.sum(waste))     # what it LOGGED: the aimed waste
    return restored


def _host_logits(hp, focus=FOCUS):
    lg = np.full(A0 + len(hp), -np.inf)
    alive = hp > 0
    att = -focus * hp[alive] / 160.0
    lg[A0:][alive] = att
    m = att.max()
    lse = m + np.log(np.exp(att - m).sum())
    lg[1] = lse + np.log(P_STOP / (1.0 - P_STOP))
    return lg


def run(job):
    config, arm, sigma, seed, cycles, focus = job
    kw = dict(map_name="3s5z", ns_severity=float(sigma), ns_augment=1,
              ns_oracle=int(arm == "oracle"), ns_seed=0)
    if config == "first":
        kw.update(ns_cap_mode="max", ns_period=150, ns_steer_floor=0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        e = _Env(kw)
    if config == "first":
        e._ns_restore = _first_run_restore.__get__(e, _Env)
    rng = np.random.RandomState(1000 + seed)
    period = e.driver.period
    # identical clock horizon for both configs, so warmup is comparable
    horizon = int(cycles * 15000)
    warm = 15000
    a_dmg = e.coupling.dmg
    e_dmg = step_damage(e.coupling.enemy_names, e._step_mul)
    K = A0 + e.n_enemies
    idx = np.arange(K)

    acc = {p: dict(wins=0, n=0, lens=0.0, landed=0.0, restored=0.0, design=0.0,
                   gated=0, gate_n=0) for p in ("peak", "dry", "all")}

    def new_battle():
        e.agents = toy.fresh(e.coupling.ally_names, 0)
        e.enemies = toy.fresh(e.coupling.enemy_names, 100)
        e._ns_reset_state()
        return rng.randint(0, e.n_agents, e.n_enemies)

    enemy_tgt = new_battle()
    t_ep, A_sum, wet = 0, 0.0, 0
    ep_landed = ep_rest = ep_design = 0.0
    while e.ns_clock < horizon:
        hp_e = e._ns_live_hp()
        al_alive = np.array([e.agents[i].health > 0 for i in range(e.n_agents)])
        acts = []
        for i in range(e.n_agents):
            if not al_alive[i]:
                acts.append(0)
                continue
            lg = _host_logits(hp_e, focus)
            avail = np.isfinite(lg)
            if arm != "blind":
                if config == "first":
                    valid = avail.copy()             # z-scored STOP with the attacks
                else:
                    valid = avail & (idx >= A0)      # attackable options only
                # the policy's channel, floor included (0 for the first run)
                lg = np.where(avail, steer(np.where(avail, lg, 0.0), e.ns_cost[i],
                                           TRUST, 1.0, valid, e.ns_steer_floor),
                              -np.inf)
            p = np.exp(lg - lg[avail].max())
            p = p / p.sum()
            acts.append(int(rng.choice(K, p=p)))
        # enemies keep their target until it dies, and idle at the host's rate --
        # the fight is symmetric except for WHO the host chooses to shoot
        for k in range(e.n_enemies):
            if enemy_tgt[k] < 0 or e.agents[int(enemy_tgt[k])].health <= 0:
                alive_idx = np.where(al_alive)[0]
                enemy_tgt[k] = int(rng.choice(alive_idx)) if alive_idx.size else -1
        shoots = (hp_e > 0) & (rng.rand(e.n_enemies) >= P_STOP)
        toy.tick(e, acts, a_dmg, np.where(shoots, enemy_tgt, -1), e_dmg)
        e._ns_hook(acts)
        toy.update_units(e)

        A = float(e.driver.A(e.ns_t_now))
        ph = "dry" if e.driver.is_placebo(e.ns_t_now) else ("peak" if A >= 0.5 else None)
        t_ep += 1
        A_sum += A
        wet += int(not e.driver.is_placebo(e.ns_t_now))
        if e.ns_t_now >= warm:
            for key in ([ph] if ph else []) + ["all"]:
                a = acc[key]
                if np.isfinite(e.ns_dmg_dealt):
                    a["landed"] += e.ns_dmg_dealt
                a["restored"] += e.ns_dmg_wasted
                a["design"] += e.ns_dmg_harm_design
                a["gated"] += e.ns_step_gated
                a["gate_n"] += e.ns_step_gate_n

        en_alive = any(u.health > 0 for u in e.enemies.values())
        al_any = any(u.health > 0 for u in e.agents.values())
        if (not en_alive) or (not al_any) or t_ep >= e.episode_limit:
            won = int(not en_alive and al_any)
            eph = "dry" if wet == 0 else ("peak" if A_sum / t_ep >= 0.5 else None)
            if e.ns_t_now >= warm:
                for key in ([eph] if eph else []) + ["all"]:
                    acc[key]["wins"] += won
                    acc[key]["n"] += 1
                    acc[key]["lens"] += t_ep
            enemy_tgt = new_battle()
            t_ep, A_sum, wet = 0, 0.0, 0
    return (config, arm, sigma, seed, focus), acc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 1.0, 3.0, 5.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--no-first", action="store_true")
    ap.add_argument("--arms", nargs="+", default=["blind", "pact", "oracle"])
    ap.add_argument("--focus", type=float, default=FOCUS,
                    help="host preference for the weakest target (logit per 160 hp)")
    a = ap.parse_args(argv)

    jobs = []
    for s in a.sigmas:
        for seed in a.seeds:
            for arm in a.arms:
                jobs.append(("fixed", arm, s, seed, a.cycles, a.focus))
            if not a.no_first and s in (0.0, 1.0):
                for arm in [x for x in a.arms if x != "oracle"]:
                    jobs.append(("first", arm, s, seed, a.cycles, a.focus))
    with Pool(min(a.workers, len(jobs))) as pool:
        out = pool.map(run, jobs)

    pooled = {}
    for (config, arm, sigma, seed, _f), acc in out:
        key = (config, sigma, arm)
        if key not in pooled:
            pooled[key] = {p: dict(v) for p, v in acc.items()}
        else:
            for p in acc:
                for f in acc[p]:
                    pooled[key][p][f] += acc[p][f]

    def rate(x, f, n):
        return x[f] / x[n] if x[n] else float("nan")

    print("=" * 108)
    print("STORY TEST -- Lanchester 3s5z, fixed focus-fire host, the REAL layer "
          "(%d seeds x %d cycles, first cycle dropped, focus %.1f)"
          % (len(a.seeds), a.cycles, a.focus))
    print("=" * 108)
    print("%-6s %-5s %-7s | %-8s %-8s %-6s | %-9s %-9s %-9s | %-9s %-9s | %s"
          % ("config", "sigma", "arm", "win_peak", "win_dry", "gap", "harm_peak",
             "harm_all", "delivery", "gate_peak", "gate_dry", "episodes pk/dry"))
    for key in sorted(pooled, key=lambda k: (k[0] != "first", k[1],
                                            ("blind", "pact", "oracle").index(k[2]))):
        config, sigma, arm = key
        pk, dr, al = pooled[key]["peak"], pooled[key]["dry"], pooled[key]["all"]
        wp, wd = rate(pk, "wins", "n"), rate(dr, "wins", "n")
        print("%-6s %-5.1f %-7s | %-8.3f %-8.3f %-6.3f | %-9.4f %-9.4f %-9.3f | "
              "%-9s %-9s | %d/%d"
              % (config, sigma, arm, wp, wd, wd - wp,
                 rate(pk, "restored", "landed"), rate(al, "restored", "landed"),
                 rate(al, "restored", "design"),
                 "-" if arm == "blind" else "%.2f" % rate(pk, "gated", "gate_n"),
                 "-" if arm == "blind" else "%.2f" % rate(dr, "gated", "gate_n"),
                 pk["n"], dr["n"]))
    print("-" * 108)
    print("gap = win_dry - win_peak: what the guard phase costs this host.  "
          "harm = restored / landed.\n"
          "delivery = restored / designed (killing blows cannot be restored).  "
          "gate = fraction of steering decisions below the floor.\n"
          "For 'first', harm is what the pre-tick restore actually took, not what "
          "its log reported.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
