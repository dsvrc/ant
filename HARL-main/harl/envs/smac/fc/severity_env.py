"""Formation Congestion -- the non-stationarity, as a wrapper around stock SMAC.

    StarCraft2Env                  untouched, byte for byte
      +-- FormationCongestionEnv   THIS FILE.  MAPPO / HAPPO / every baseline
            +-- PactEnv            adds the compensator only

NS_FORM_SPEC B.5: severity is TASK physics and must reach every arm, so the dial
sits BELOW the method in the class hierarchy and reads its value from the ENV
config, never from an algorithm's block.  A dial only the method's arm
experienced would be worthless as evidence.

THE FOUR OBJECTS (NS_FORM_SPEC A.2)
-----------------------------------
1. THE MEDIUM AND ITS LOADING RATIO.  Ground units are rigid bodies that block
   each other -- the oldest fact in StarCraft micro, and the reason players talk
   about surface area and concaves.  Unit *i*'s medium is the ground it has to
   move through; its loading is

       u_i(t) = L_i(t) / K_i(t) ,        L_i^m = sum_j W[i,j] prox_ij cone_ij^m Phi_j

   read on the BINDING direction (4.1: use the max, not the average -- on POWER
   switching the sensor from mean to max raised applied compensation 6x).

2. THE COUPLING OPERATOR W.  Declared from StarCraft II's own unit data; see
   fc/operator.py.  Zero-diagonal, asymmetric, 25x spread.  Never fitted.

3. THE EXOGENOUS DRIVER.  The enemy line's push/consolidate cycle shrinks the
   squad's usable frontage: K_i(t) = K_i^0 * g(sigma, A(t)).  It reaches the
   agents ONLY through the capacity, so at N=1 the peer term is exactly zero
   however small g becomes.  See fc/driver.py.

4. THE HARM CHANNEL AND ITS INVERSE.  Congestion throttles the step a unit
   actually takes:

       delivered_i = base_i * cmd_i * (1 - Delta_i) ,   Delta_i = u_i * (1 - g)

   which is A.2's multiplicative row -- inverse ``cmd = 1/(1 - Delta)``, exact
   until the rail.  Reward is untouched byte for byte; a unit simply gets less
   done because the ground delivers less.

   At sigma = 0, g == 1 and Delta == 0 identically, so every move order is the
   one stock SMAC would have issued.  The same holds inside the driver's placebo
   regime at EVERY severity (B.4).

WHY THE CHANNEL IS CONTINUOUS AND NOT A PERMUTATION.  A.7 measured the
alternative on SMAC directly: on a target-deflection (permutation) channel
partial compensation lands the shot on a DIFFERENT wrong target -- beta=0.5
scored 12.5 against 13.0 for no compensation at all.  A throttled stride is a
continuous multiplicative channel, so trust SCALES the correction instead of
having to threshold it, and a partly-right estimate buys a partly-right outcome.

THE REACH CLIP.  Stock SMAC sends every move order to a point ``move_amount``
away, but a unit only covers ``speed * step_mul / 22.4`` in one step -- 1.47 for a
Stalker, 1.12 for a Zealot -- so at the default ``move_amount = 2`` the order
never binds and scaling it down would do NOTHING until the scaled distance fell
below the reach: a dead zone exactly like the one A.7/PACT 6.3 warn about.  The
wrapper therefore clips the base order to the unit's own one-step reach, which
changes no displacement at all (the unit was never going to get further) and
removes the dead zone entirely.  ``ns_reach_clip: 0`` restores the raw form as an
ablation, and fc/certificates.py MEASURES the displacement difference rather than
asserting it.
"""

import numpy as np

from . import operator as opmod
from .driver import Driver, assert_dial, is_placebo

N_ACTIONS_NO_ATTACK = 6      # no-op, stop, N, S, E, W -- stock SMAC


class FormationCongestionEnv:
    """The severity layer.  Delegates everything it does not own to the wrapped
    StarCraft2Env, so it is a drop-in replacement everywhere in HARL."""

    def __init__(self, env, args=None):
        self.env = env
        a = dict(args or {})

        # ---- the dial, read from the TASK config (B.5) ----------------------
        self.severity = float(a.get("ns_severity", 1.0))
        period = a.get("ns_period", None)
        if period is None:
            # One push/consolidate cycle per engagement.  Slow relative to the
            # control step (A.2 object 3) yet fully exercised inside every episode,
            # so the placebo regime and the driver peak are both experienced by
            # every rollout rather than being an across-episode lottery.
            period = max(20, int(getattr(env, "episode_limit", 150)))
        self.driver = Driver(
            severity=self.severity,
            period=int(period),
            phase0=float(a.get("ns_phase", 0.0)),
            knee=float(a.get("ns_knee", 0.35)),
            depth=float(a.get("ns_depth", 0.60)),
            floor=float(a.get("ns_floor", 0.25)),
            warmup=int(a.get("ns_warmup", 0)),
            ramp=int(a.get("ns_ramp", 0)),
            freeze=(None if a.get("ns_freeze", None) is None
                    else float(a["ns_freeze"])),
            eval_mode=bool(int(a.get("ns_eval", 0))),
            mean_preserving=bool(int(a.get("ns_mean_preserving", 0))),
        )
        # B.1 is certified here, at construction, on every process.
        assert_dial(knee=self.driver.knee, depth=self.driver.depth,
                    floor=self.driver.floor)

        # ---- the declared operator ----------------------------------------
        self.map_name = str(getattr(env, "map_name", "unknown"))
        self.map_type = str(getattr(env, "map_type", "stalkers_and_zealots"))
        self.n_agents = int(env.n_agents)
        self.n_enemies = int(env.n_enemies)
        self.step_mul = int(getattr(env, "_step_mul", 8))
        self.ally_names = opmod.composition(self.map_name, self.n_agents, self.map_type)
        self.enemy_names = opmod.enemy_composition(self.map_name, self.n_enemies,
                                                   self.map_type)
        self.W = opmod.build_W(self.ally_names, self.ally_names, self.step_mul,
                               band_scale=float(a.get("ns_band_scale",
                                                      opmod.BAND_SCALE)))
        # G1 / A.2: zero diagonal is ASSERTED, never argued.
        assert np.allclose(np.diag(self.W), 0.0), "W[i,i] must be exactly 0"
        self.W_env = opmod.build_W(self.ally_names, self.enemy_names, self.step_mul,
                                   band_scale=float(a.get("ns_band_scale",
                                                          opmod.BAND_SCALE)),
                                   zero_diagonal=False)
        self.r_ally = opmod.radii(self.ally_names)
        self.r_enemy = opmod.radii(self.enemy_names)
        self.reach = opmod.reach(self.ally_names, self.step_mul)
        self.prox_len = float(a.get("ns_prox_len", opmod.PROX_LEN))

        # ---- the exertion functional (A.5) --------------------------------
        self.exertion = opmod.Exertion(
            self.n_agents,
            phi_fire=float(a.get("ns_phi_fire", 1.0)),
            rho=float(a.get("ns_phi_rho", 0.6)),
            # > 0 makes Phi read the EXECUTED stride, i.e. LOOP-COUPLED (A.6).
            phi_move=float(a.get("ns_phi_move", 0.0)),
        )
        self.phi_max = self.exertion.phi_max
        # A.5 counter-check: uncancellable AND varying.  phi_fire = 1.0 with
        # rho = 0.6 measures std(Phi)/mean(Phi) = 0.155 on a synthetic 3s5z
        # engagement (POWER ran 0.28); fc_phi_var reports the live value and the
        # G-gates fail the run if it drops under 0.05.

        # ---- the capacity (declared, per agent) ---------------------------
        # K_i^0 = k_scale * (every body that could ever obstruct i, at its declared
        # maximum exertion).  Declared, so u_i is comparable across unit types by
        # construction rather than by tuning; k_scale is the single units-of-the-
        # medium constant and is frozen before any method runs (fc/calibrate.py).
        row = self.W.sum(axis=1) + self.W_env.sum(axis=1)
        self.k_scale = float(a.get("ns_k_scale", 0.35))
        self.K0 = self.k_scale * row * self.phi_max
        assert np.all(self.K0 > 0.0), "some agent has zero declared capacity"

        # ---- the harm channel ---------------------------------------------
        self.harm_max = float(a.get("ns_harm_max", 0.75))   # never fully frozen
        self.reach_clip = bool(int(a.get("ns_reach_clip", 1)))
        ma = float(getattr(env, "_move_amount", 2.0))
        self.base_frac = (np.minimum(1.0, self.reach / max(1e-9, ma))
                          if self.reach_clip else np.ones(self.n_agents))
        # Declared sensor noise, drawn from its OWN generator so it can never
        # perturb the game stream: two arms differing only in sensor noise stay
        # dynamically comparable.
        self.sensor_noise = float(a.get("ns_sensor_noise", 0.0))
        self._sensor_rng = np.random.RandomState(9000 + int(a.get("ns_seed", 0)))

        # ---- state ---------------------------------------------------------
        self.cmd = np.ones(self.n_agents)          # driven by the layer above
        self.stride = np.ones(self.n_agents)       # what the env was told to do
        self.delta = np.zeros(self.n_agents)       # Delta_i this step
        self.delta_meas = np.zeros(self.n_agents)  # the SENSOR reading
        self.u = np.zeros(self.n_agents)
        self.u_bind = np.zeros(self.n_agents, dtype=int)
        self.contrib = np.zeros((self.n_agents, self.n_agents))   # for PACT's psi
        self.L_peer = np.zeros(self.n_agents)
        self.L_fix = np.zeros(self.n_agents)
        self.g = 1.0
        self.A = 0.0
        self.sigma_applied = 0.0
        self.moved = np.zeros(self.n_agents)
        self._odom_err = 0.0
        self._odom_n = 0
        self._dial_live = 0
        self._dial_steps = 0
        self._move_steps = 0
        self._act_steps = 0
        self._banner()

        # HARL reads these straight off the env object.
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space

    # ------------------------------------------------------------------ #
    def _banner(self):
        rep = opmod.report(self.W, self.ally_names)
        print(opmod.banner(rep, "W(ally,ally) %s" % self.map_name))
        print(opmod.banner(opmod.report(self.W_env, self.ally_names,
                                        zero_diagonal=False), "W(ally,enemy)"))
        A_cycle = np.array([0.5 - 0.5 * np.cos(2.0 * np.pi * k / self.driver.period)
                            for k in range(self.driver.period)])
        placebo_frac = float(np.mean(is_placebo(A_cycle, self.driver.knee)))
        print("[FC] sigma=%.3f period=%d knee=%.2f depth=%.2f floor=%.2f "
              "placebo_frac=%.2f warmup=%d ramp=%d eval=%d mean_preserving=%d"
              % (self.severity, self.driver.period, self.driver.knee,
                 self.driver.depth, self.driver.floor, placebo_frac,
                 self.driver.warmup, self.driver.ramp,
                 int(self.driver.eval_mode), int(self.driver.mean_preserving)))
        print("[FC] Phi = alive*(1 + %.2f*fired) + %.2f*stride  rho=%.2f  "
              "phi_max=%.2f  (loop=%s)"
              % (self.exertion.phi_fire, self.exertion.phi_move, self.exertion.rho,
                 self.phi_max, "YES" if self.exertion.phi_move > 0 else "no"))
        print("[FC] k_scale=%.3f K0=[%.2f..%.2f] harm_max=%.2f reach_clip=%d "
              "base_frac=[%.3f..%.3f] move_amount=%.2f reach=[%.2f..%.2f]"
              % (self.k_scale, self.K0.min(), self.K0.max(), self.harm_max,
                 int(self.reach_clip), self.base_frac.min(), self.base_frac.max(),
                 float(getattr(self.env, "_move_amount", 2.0)),
                 self.reach.min(), self.reach.max()))

    # ------------------------------------------------------------------ #
    # geometry / physics
    # ------------------------------------------------------------------ #
    def _positions(self):
        n, m = self.n_agents, self.n_enemies
        pa = np.zeros((n, 2))
        pe = np.zeros((m, 2))
        alive_a = np.zeros(n)
        alive_e = np.zeros(m)
        agents = getattr(self.env, "agents", {}) or {}
        enemies = getattr(self.env, "enemies", {}) or {}
        for i in range(n):
            u = agents.get(i, None)
            if u is not None:
                pa[i] = (u.pos.x, u.pos.y)
                alive_a[i] = 1.0 if u.health > 0 else 0.0
        for e in range(m):
            u = enemies.get(e, None)
            if u is not None:
                pe[e] = (u.pos.x, u.pos.y)
                alive_e[e] = 1.0 if u.health > 0 else 0.0
        return pa, pe, alive_a, alive_e

    def _physics(self, actions_int):
        """Compute Phi, the loading, the binding direction and Delta for this step."""
        pa, pe, alive_a, alive_e = self._positions()
        fired = np.array(
            [1.0 if (alive_a[i] > 0 and int(actions_int[i]) >= N_ACTIONS_NO_ATTACK)
             else 0.0 for i in range(self.n_agents)])
        phi = self.exertion.update(alive_a, fired, self.stride)

        prox_aa, cone_aa = opmod.kernels(pa, pa, self.r_ally, self.r_ally,
                                         self.prox_len)
        L_peer = opmod.loading(self.W, prox_aa, cone_aa, phi, alive_a)      # (n, 4)
        if self.n_enemies > 0:
            prox_ae, cone_ae = opmod.kernels(pa, pe, self.r_ally, self.r_enemy,
                                             self.prox_len)
            L_fix = opmod.loading(self.W_env, prox_ae, cone_ae,
                                  np.full(self.n_enemies, opmod.PHI_ENEMY), alive_e)
        else:
            L_fix = np.zeros_like(L_peer)

        g, A, sigma = self.driver.g()
        K = self.K0 * g
        u_dir = (L_peer + L_fix) / K[:, None]
        # 4.1 / A.4: the BINDING aggregate, MAX over directions -- congestion is a
        # property of the tightest sector, and averaging over four of them dilutes
        # the signal toward zero.
        bind = np.argmax(u_dir, axis=1)
        u = u_dir[np.arange(self.n_agents), bind]

        # Delta = u * (1 - g).  Exactly zero when g == 1, i.e. at sigma = 0 and
        # throughout the placebo regime, at every severity.
        delta = np.clip(u * (1.0 - g), 0.0, self.harm_max) * (alive_a > 0)

        # what PACT's basis needs: the per-pair contribution on the binding
        # direction.  The environment owns the geometry; the basis owns the
        # channels and the scaling.
        idx = np.arange(self.n_agents)
        self.contrib = self.W * prox_aa * cone_aa[idx, :, bind]
        self.L_peer = L_peer[idx, bind]
        self.L_fix = L_fix[idx, bind]
        self.u, self.u_bind, self.delta = u, bind, delta
        self.g, self.A, self.sigma_applied = g, A, sigma
        self._alive = alive_a
        return delta

    # ------------------------------------------------------------------ #
    # the interface the layer above drives
    # ------------------------------------------------------------------ #
    def set_command(self, cmd):
        """The commanded stride multiplier per agent (1.0 == the blind action).

        PactEnv writes here.  When it writes exactly 1.0 the order issued is
        byte-identical to the blind arm's -- PACT_PIPELINE_SPEC 1's floor
        property, and it is a property of this interface rather than a promise.
        """
        self.cmd = np.clip(np.asarray(cmd, dtype=np.float64).reshape(self.n_agents),
                           0.0, 8.0)

    def sensor(self):
        """Each agent's OWN deficit reading, Delta_i(t) -- what the last step cost
        it.  Odometry (``1 - realized/commanded``), not privilege; the odometric
        reconstruction is cross-checked every step and reported as
        ``fc_odom_err``.  It reports the PAST, which is the whole problem (A.4)."""
        return self.delta_meas.copy()

    # ------------------------------------------------------------------ #
    def reset(self):
        out = self.env.reset()
        self.exertion.reset()
        self.cmd[:] = 1.0
        self.stride[:] = 1.0
        self.delta[:] = 0.0
        self.delta_meas[:] = 0.0
        self.contrib[:] = 0.0
        self.env.move_stride[:] = 1.0
        return out

    def step(self, actions):
        actions_int = [int(np.asarray(a).flatten()[0]) for a in actions]
        delta = self._physics(actions_int)

        # the order the environment will issue:  base * cmd * (1 - Delta)
        self.stride = self.base_frac * self.cmd * (1.0 - delta)
        self.env.move_stride[:] = self.stride

        pre, _, _, _ = self._positions()
        out = self.env.step(actions)
        post, _, _, _ = self._positions()

        # --- the sensor, and its odometric cross-check ----------------------
        move_a = np.array([2 <= int(a) < N_ACTIONS_NO_ATTACK for a in actions_int],
                          dtype=np.float64) * self._alive
        self.moved = move_a
        meas = delta.copy()
        if self.sensor_noise > 0.0:
            meas = np.clip(meas + self._sensor_rng.randn(self.n_agents)
                           * self.sensor_noise, 0.0, 1.0)
        self.delta_meas = meas
        ma = float(getattr(self.env, "_move_amount", 2.0))
        for i in range(self.n_agents):
            if move_a[i] <= 0.0:
                continue
            commanded = ma * self.base_frac[i] * self.cmd[i]
            if commanded <= 1e-6:
                continue
            realized = float(np.linalg.norm(post[i] - pre[i]))
            self._odom_err += abs((1.0 - realized / commanded) - delta[i])
            self._odom_n += 1

        self._dial_steps += 1
        self._dial_live += int(self.g < 1.0)
        self._move_steps += int(move_a.sum())
        self._act_steps += int(self._alive.sum())
        self.driver.advance()

        out = self._inject(out)
        return out

    # ------------------------------------------------------------------ #
    def diagnostics(self):
        """The columns PACT_PIPELINE_SPEC 9 asks to be read before anything else.

        Every ratio is guarded with NaN, never an epsilon: at the driver trough
        the disturbance is genuinely ~0 and the ratio is meaningless.  A 1e-12
        floor once produced a -1011 that poisoned a column average.
        """
        live = self._alive > 0 if hasattr(self, "_alive") else np.ones(self.n_agents, bool)
        nz = int(live.sum())
        def m(x):
            return float(np.mean(np.asarray(x)[live])) if nz else float("nan")
        return {
            "fc_A": float(self.A),
            "fc_g": float(self.g),
            "fc_sigma": float(self.sigma_applied),
            "fc_placebo": float(bool(is_placebo(self.A, self.driver.knee))),
            "fc_u_mean": m(self.u),
            "fc_u_max": float(np.max(self.u[live])) if nz else float("nan"),
            "fc_delta_mean": m(self.delta),
            "fc_delta_max": float(np.max(self.delta[live])) if nz else float("nan"),
            "fc_peer_share": (float(np.sum(self.L_peer[live])
                                    / np.sum(self.L_peer[live] + self.L_fix[live]))
                              if nz and np.sum(self.L_peer[live] + self.L_fix[live]) > 0
                              else float("nan")),
            "fc_stride_mean": m(self.stride),
            "fc_cmd_mean": m(self.cmd),
            "fc_move_frac": (float(self._move_steps) / self._act_steps
                             if self._act_steps else float("nan")),
            "fc_phi_var": float(self.exertion.variation()),
            "fc_odom_err": (float(self._odom_err / self._odom_n)
                            if self._odom_n else float("nan")),
            "fc_dial_ratio": (float(self._dial_live) / self._dial_steps
                              if self._dial_steps else float("nan")),
            "fc_alive": float(nz),
            # A state for the BLIND arms too, so a blank column in fc_debug.csv
            # never has to be interpreted as "the compensator was off" when it
            # actually means "no compensator was ever attached".
            "pact_state": "INERT" if self.sigma_applied <= 0.0 else "BLIND",
            # --- neutral aliases, so harl/common/ns_probe.py attaches to EVERY
            # on-policy arm on this env -- including `--algo mappo`, `--algo corep`
            # and anything else that never writes fc_debug.csv.  A silently inert
            # dial is invisible in exactly the arm you most need to trust, and that
            # hole once hid an inert disturbance for a full run.
            "pcr_payload": float(self.A),
            "pcr_load": m(self.delta),
            "pcr_loadmax": (float(np.max(self.delta[live])) if nz else float("nan")),
            "pcr_severity": float(self.sigma_applied),
            "pcr_sat_frac": m(1.0 - self.stride / np.maximum(1e-12, self.base_frac)),
        }

    def _inject(self, out):
        if not isinstance(out, tuple) or len(out) < 5:
            return out
        out = list(out)
        infos = out[4]
        d = self.diagnostics()
        try:
            for i in range(len(infos)):
                if isinstance(infos[i], dict):
                    infos[i].update(d)
                    # per-agent quantities the runner's debug file wants
                    infos[i]["fc_delta_i"] = float(self.delta[i])
                    infos[i]["fc_u_i"] = float(self.u[i])
        except (TypeError, IndexError):
            pass
        out[4] = infos
        return tuple(out)

    # ------------------------------------------------------------------ #
    def __getattr__(self, name):
        # Only reached for attributes this wrapper does not define, so it can
        # never shadow the wrapper's own state.  Raises AttributeError (not
        # KeyError) before __init__ has run, which is what callers expect.
        try:
            env = self.__dict__["env"]
        except KeyError:
            raise AttributeError(name)
        return getattr(env, name)
