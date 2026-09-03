"""A test double for ``StarCraft2Env``: same interface, no StarCraft II.

It exists for exactly one reason.  PACT_PIPELINE_SPEC 11.5 says to *"wire the env
wrapper; verify the banner and ``delta_nonzero_frac`` on an 18k-frame probe BEFORE
any long run"*, and on a machine with no StarCraft II installed the only way to do
that is to substitute the simulator.  The double implements the parts of the SMAC
interface the wrappers touch -- units with positions and health, the
``move_stride`` hook, the 6-tuple ``step`` contract -- and gives its units a crude
but honest kinematics: an ordered move actually displaces the unit by the ordered
distance, capped by its one-step reach.

IT IS NOT A SIMULATOR AND IT PROVES NOTHING ABOUT THE METHOD.  It proves the
plumbing: that the wrapper stack builds, that the shapes and the info dict come
out right, that the sensor reconstructs the deficit from odometry, and that with
the gates shut the orders issued are BYTE-IDENTICAL to the blind arm's.
"""

import numpy as np
from gym.spaces import Discrete

from . import operator as opmod


class _Pos(object):
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


class _Unit(object):
    __slots__ = ("pos", "health", "shield", "unit_type")

    def __init__(self, x, y, health=100.0, unit_type=0):
        self.pos = _Pos(x, y)
        self.health = float(health)
        self.shield = 0.0
        self.unit_type = int(unit_type)


class MockSmacEnv(object):
    """Enough of StarCraft2Env for the Formation Congestion wrappers to run."""

    def __init__(self, map_name="3s5z", episode_limit=150, step_mul=8,
                 move_amount=2.0, seed=0):
        self.map_name = map_name
        self.map_type = "stalkers_and_zealots"
        self.ally_names = opmod.MAP_COMPOSITION.get(map_name, ["Stalker"] * 8)
        self.enemy_names = opmod.MAP_ENEMY_COMPOSITION.get(map_name, ["Stalker"] * 8)
        self.n_agents = len(self.ally_names)
        self.n_enemies = len(self.enemy_names)
        self.episode_limit = int(episode_limit)
        self._step_mul = int(step_mul)
        self._move_amount = float(move_amount)
        self.n_actions_no_attack = 6
        self.n_actions = self.n_actions_no_attack + self.n_enemies
        self.move_stride = np.ones(self.n_agents, dtype=np.float64)
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.reach = opmod.reach(self.ally_names, step_mul)
        # The same declared shapes stock SMAC produces, so the HARL buffers and
        # actor size themselves exactly as they would on the real environment.
        self.obs_dim = 32
        self.state_dim = 64
        self.observation_space = [[self.obs_dim] for _ in range(self.n_agents)]
        self.share_observation_space = [[self.state_dim] for _ in range(self.n_agents)]
        self.action_space = [Discrete(self.n_actions) for _ in range(self.n_agents)]
        self._rng = np.random.RandomState(seed)
        self._t = 0
        self.battles_won = 0
        self.battles_game = 0
        self.reset()

    # ------------------------------------------------------------------ #
    def _obs(self):
        # Not a real observation -- a deterministic function of the state, enough
        # for the buffers and the actor to run.  This double proves the PLUMBING.
        o = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for i in range(self.n_agents):
            u = self.agents[i]
            o[i, 0] = u.health / 100.0
            o[i, 1] = u.pos.x / 32.0
            o[i, 2] = u.pos.y / 32.0
            o[i, 3] = self._t / float(self.episode_limit)
        s = np.tile(o[:, :4], (1, self.state_dim // 4)).astype(np.float32)
        return o, s

    def get_avail_actions(self):
        av = np.ones((self.n_agents, self.n_actions), dtype=int)
        av[:, 0] = 0
        for i in range(self.n_agents):
            if self.agents[i].health <= 0:
                av[i] = 0
                av[i, 0] = 1
        return av

    def reset(self):
        self._t = 0
        self.agents = {}
        self.enemies = {}
        for i in range(self.n_agents):
            self.agents[i] = _Unit(14.0 + self._rng.randn() * 1.5,
                                   12.0 + self._rng.randn() * 1.5, 100.0)
        for e in range(self.n_enemies):
            self.enemies[e] = _Unit(14.0 + self._rng.randn() * 1.5,
                                    20.0 + self._rng.randn() * 1.5, 100.0)
        self.death_tracker_ally[:] = 0.0
        self.move_stride[:] = 1.0
        o, s = self._obs()
        return o, s, self.get_avail_actions()

    def step(self, actions):
        acts = [int(np.asarray(a).flatten()[0]) for a in actions]
        for i, a in enumerate(acts):
            u = self.agents[i]
            if u.health <= 0:
                continue
            if 2 <= a < self.n_actions_no_attack:
                # the ORDERED distance, capped by what the unit can actually cover
                d = min(self._move_amount * float(self.move_stride[i]),
                        float(self.reach[i]))
                dx, dy = {2: (0, 1), 3: (0, -1), 4: (1, 0), 5: (-1, 0)}[a]
                u.pos.x += dx * d
                u.pos.y += dy * d
            elif a >= self.n_actions_no_attack:
                e = self.enemies.get(a - self.n_actions_no_attack)
                if e is not None and e.health > 0:
                    e.health = max(0.0, e.health - 4.0)
        for i in range(self.n_agents):
            if self.agents[i].health > 0 and self._rng.rand() < 0.004:
                self.agents[i].health = 0.0
                self.death_tracker_ally[i] = 1.0
        self._t += 1
        done = (self._t >= self.episode_limit
                or all(u.health <= 0 for u in self.agents.values())
                or all(u.health <= 0 for u in self.enemies.values()))
        o, s = self._obs()
        infos = [{"battles_won": 0, "battles_game": 0, "bad_transition": False,
                  "won": False} for _ in range(self.n_agents)]
        dones = np.full(self.n_agents, bool(done))
        rewards = [[1.0]] * self.n_agents
        return o, s, rewards, dones, infos, self.get_avail_actions()

    def seed(self, s):
        self._rng = np.random.RandomState(s)

    def close(self):
        pass

    def save_replay(self):
        pass
