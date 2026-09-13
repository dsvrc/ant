"""A stand-in for the StarCraft II engine, for OFFLINE tests only.

It reproduces the ORDER of operations in ``StarCraft2_Env.step`` -- the one thing
the severity layer depends on -- and nothing else of the game (no space, no
movement, no cooldowns, no armour):

    actions -> engine tick       damage lands, stock overkill, the dead drop out
            -> observe()         self._obs = the post-tick units
            -> _ns_hook(actions) harm computed from the PRE-tick self.enemies;
                                 the restore is written into the post-tick snapshot
            -> update_units()    self.enemies / self.agents <- the snapshot

The first restore got this order wrong -- it wrote pre-tick values into the engine
-- and undid whole steps of shield damage.  Every offline test that exercises the
harm goes through here so that cannot pass silently again.
"""

import numpy as np

from .coupling import UNIT_STATS


class Unit(object):
    """The three fields of ``raw_pb.Unit`` the layer reads.  Mutable, as protos are."""

    def __init__(self, tag, health, shield):
        self.tag, self.health, self.shield = int(tag), float(health), float(shield)


class Obs(object):
    """``self._obs.observation.raw_data.units``, and nothing more."""

    def __init__(self, units):
        raw = type("RawData", (), {})()
        raw.units = list(units)
        ob = type("Observation", (), {})()
        ob.raw_data = raw
        self.observation = ob


class BattleHost(object):
    """Just enough host for ``SeverityMixin``: the 3s5z shape and unit tables.

    Like StarCraft2Env it calls its own virtuals during construction, so the
    mixin's construction-order guard is exercised here too."""

    def __init__(self, args, **kwargs):
        self.map_name = args["map_name"]
        self.n_agents, self.n_enemies = 8, 8
        self.episode_limit, self._step_mul = 150, 8
        self.n_actions_no_attack = 6
        self.n_actions = self.n_actions_no_attack + self.n_enemies
        self._controller = None
        self._obs = None
        self.agents = {i: None for i in range(self.n_agents)}
        self.enemies = {i: None for i in range(self.n_enemies)}
        self.observation_space = [self.get_obs_size() for _ in range(self.n_agents)]

    def get_obs_size(self):
        return [1, [0, 0], [0, 0], [0, 0], [0, 0]]

    def get_obs_agent(self, agent_id):
        return np.zeros(1, dtype=np.float32)

    def reset(self):
        return None


def fresh(names, tag0):
    """A full-health roster, keyed by SMAC index."""
    return {k: Unit(tag0 + k, UNIT_STATS[nm]["life"], UNIT_STATS[nm]["shield"])
            for k, nm in enumerate(names)}


def _apply(units, incoming):
    for k, dmg in incoming.items():
        if dmg <= 0.0 or k not in units:
            continue
        u = units[k]
        take = min(u.shield, dmg)
        u.shield -= take
        u.health -= dmg - take                  # overkill below zero is simply lost


def tick(env, ally_acts, ally_dmg, enemy_tgt=None, enemy_dmg=None):
    """One engine step, then ``observe()``: sets ``env._obs`` to the post-tick units.

    ``ally_acts`` are SMAC action ids (attack k = 6 + k).  ``enemy_tgt[k]`` is the
    ally enemy k shoots, or -1.  Damage is simultaneous; the dead drop out of the
    observation exactly as SC2 drops them from ``raw_data.units``.
    """
    post_en = {k: Unit(u.tag, u.health, u.shield)
               for k, u in env.enemies.items() if u is not None and u.health > 0}
    post_al = {i: Unit(u.tag, u.health, u.shield)
               for i, u in env.agents.items() if u is not None and u.health > 0}
    inc_en, inc_al = {}, {}
    for i, a in enumerate(ally_acts):
        if i in post_al and int(a) >= 6:
            k = int(a) - 6
            inc_en[k] = inc_en.get(k, 0.0) + float(ally_dmg[i])
    if enemy_tgt is not None:
        for k, j in enumerate(enemy_tgt):
            if k in post_en and int(j) >= 0:
                inc_al[int(j)] = inc_al.get(int(j), 0.0) + float(enemy_dmg[k])
    _apply(post_en, inc_en)
    _apply(post_al, inc_al)
    units = [u for u in list(post_en.values()) + list(post_al.values()) if u.health > 0]
    env._obs = Obs(units)
    return units


def update_units(env):
    """``StarCraft2Env.update_units``: take the snapshot; the dead keep their last
    proto with health set to 0 (and whatever shields it last showed)."""
    by_tag = {u.tag: u for u in env._obs.observation.raw_data.units}
    for d in (env.enemies, env.agents):
        for k, u in list(d.items()):
            if u is None:
                continue
            now = by_tag.get(u.tag, None)
            d[k] = now if now is not None else Unit(u.tag, 0.0, u.shield)


def step(env, ally_acts, ally_dmg, enemy_tgt=None, enemy_dmg=None):
    """tick -> hook -> update_units, in the engine's order."""
    tick(env, ally_acts, ally_dmg, enemy_tgt, enemy_dmg)
    env._ns_hook(ally_acts)
    update_units(env)
