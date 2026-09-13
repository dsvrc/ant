"""The severity layer for SURFACE AREA UNDER DRIFT -- a mixin over StarCraft2Env.

    StarCraft2Env                       stock (its one no-op hook is not used here)
      +-- SmacSaEnv(SurfaceMixin, ..)   THIS FILE.  Every arm gets it, unmodified.
            +-- trust term in the actor  the method, and nothing else

THE HARM IS APPLIED TO ACTIONS, NEVER TO HIT POINTS
--------------------------------------------------------------------------------
Before the engine tick, each attack order is checked against the ring around its
target.  With probability  p_i = excess_i / (1 + excess_i)  it finds no free slot
and is executed as STOP for that step.  Then the stock engine plays the step.

That one choice removes every failure the overkill instance paid for:
  * no hit-point writes, so no pre-tick/post-tick ordering to get wrong, no
    snapshot/engine desynchronisation, and nothing for reward_battle's abs() to
    pay out;
  * killing blows are not special: a blocked zealot simply did not swing;
  * the reward, the observation, the termination test and the records are the
    engine's own, so NS-3.2 holds by construction.
The policy is trained on the action it CHOSE; the engine executed STOP instead --
a transition the agent does not control, exactly like a wall.

    u_i       = sum_{j != i, melee, same target, nearer} (r_j/r_i) / capacity_i(target)
    excess_i  = f(u_i/g)/f(u_i) - 1 = (g**-4 - 1) * s(u_i),   f = BPR
    g_e       = 1 - sigma * L * A(t) * sens_e,   L = 1/3  (see driver.py)
    g_i       = max(g_e, 1/capacity_i(e))     one slot always stays open

A lone attacker has u = 0, so p = 0 EXACTLY at any severity; sigma = 0 gives g = 1,
so p = 0 EXACTLY for everyone and the task is stock SMAC step for step.
"""

import numpy as np

from ..smac_ns.pact1_core import AgentRLS, herd_index, relative_excess, rls_confidence_pred
from .coupling import SlotCoupling
from .driver import FormationDriver, HEX_LINE_LOSS

N_ACTIONS_NO_ATTACK = 6          # no-op, stop, N, S, E, W -- stock SMAC
STOP = 1

#: every key ns_info() emits, in the order the debug file logs them.  The runner
#: imports this list, so the schema cannot drift between the two again.
NS_KEYS = [
    # ---- is the DIAL live? -------------------------------------------------
    "ns_sigma", "ns_on", "ns_A", "ns_g", "ns_g_min", "ns_placebo", "ns_dial_ratio",
    "ns_clock",
    # ---- did the harm REACH the game? ------------------------------------------
    "ns_attempts", "ns_blocked", "ns_harm_num", "ns_harm_den", "ns_block_frac_cum",
    "ns_melee_attempts", "ns_melee_blocked", "ns_p_block", "ns_p_block_max",
    # ---- is the MEDIUM loaded? --------------------------------------------------
    "ns_u", "ns_u_max", "ns_u_melee", "ns_excess", "ns_excess_max", "ns_y",
    "ns_fire_frac", "ns_alive", "ns_herd_index", "ns_melee_herd_index",
    "ns_switch_frac",
    # ---- is the METHOD working? -------------------------------------------------
    "ns_fit_gain", "ns_pred_gain", "ns_cond_psi", "ns_conf", "ns_rows", "ns_clip_frac",
    "ns_innov", "ns_trP", "ns_x_std", "ns_x_mean",
    # ---- is the STEERING acting where the harm is, and only there? -------------
    "ns_cost_std", "ns_cost_spread", "ns_rank_snr", "ns_steer_gated_frac",
    "ns_step_gated", "ns_step_gate_n", "ns_mem_frac",
    "ns_beta0", "ns_beta1", "ns_beta2", "ns_beta3",
]


class SurfaceMixin(object):
    """The dial.  Mix in ABOVE ``StarCraft2Env``."""

    def __init__(self, args, **kwargs):
        a = dict(args or {})
        # set BEFORE the host's __init__: StarCraft2Env calls get_obs_size() (our
        # override) while it builds its observation space
        self.ns_augment = bool(int(a.get("ns_augment", 0)))
        self.ns_cost = None

        super(SurfaceMixin, self).__init__(args, **kwargs)

        self.ns_sigma = float(a.get("ns_severity", 1.0))
        self.ns_on = bool(int(a.get("ns_on", 1)))
        # one URB "day" is one battle; URB's reference cycle is 100 days
        self.driver = FormationDriver(
            period=int(a.get("ns_period", 100 * self.episode_limit)),
            guard_frac=float(a.get("ns_guard_frac", 0.5)),
            loss=float(a.get("ns_loss", HEX_LINE_LOSS)),
            mean_preserving=bool(int(a.get("ns_mean_preserving", 0))),
        )
        self.coupling = SlotCoupling(self.map_name, self.n_agents, self.n_enemies,
                                     alpha=float(a.get("ns_bpr_alpha", 0.15)),
                                     beta=float(a.get("ns_bpr_beta", 4.0)))
        self.driver.certify()
        self._ns_startup_gates()

        # per-enemy sensitivity to closing ranks: declared, mean 1, clipped -- the
        # binding element can shift with severity instead of scaling uniformly
        rng = np.random.RandomState(7000 + int(a.get("ns_seed", 0)))
        self.ns_sens = np.clip(1.0 + 0.5 * rng.randn(self.n_enemies), 0.4, 2.0)
        self.ns_sens /= self.ns_sens.mean()
        # the slot lottery has its own stream: different per worker (phase0 is
        # de-phased per rank), identical across arms and severities
        self._ns_lottery = np.random.RandomState(
            (9000 + 7919 * int(a.get("ns_seed", 0)) + int(a.get("ns_phase0", 0)))
            % (2 ** 31 - 1))

        # ---- the estimator.  Pure identification; identical in every arm. ----
        self.ns_mu = float(a.get("ns_mu", 0.999))
        self.ns_p0 = float(a.get("ns_p0", 10.0))
        self.ns_clip = float(a.get("ns_clip", 10.0))
        self.ns_intercept_only = bool(int(a.get("ns_intercept_only", 0)))
        self.ns_oracle = bool(int(a.get("ns_oracle", 0)))
        # slots go to the nearest attackers ("nearest"); "shared" is the
        # symmetric ablation where every peer on the ring loads every attacker
        self.ns_slot_order = str(a.get("ns_slot_order", "shared"))
        assert self.ns_slot_order in ("nearest", "shared"), self.ns_slot_order
        self.ns_steer_floor = float(a.get("ns_steer_floor", 0.01))
        self._rls = [AgentRLS(self.coupling.r + 1, mu=self.ns_mu, p0=self.ns_p0)
                     for _ in range(self.n_agents)]
        mem = 1.0 / max(1e-12, 1.0 - self.ns_mu)
        if mem > 0.25 * self.driver.period:
            print("[SMAC-SA][NOT-TRACKABLE] estimator memory >= %.0f steps is %.0f%% "
                  "of the %d-step driver period (limit 25%%): beta_hat will average "
                  "the cycle.  Lengthen ns_period or lower ns_mu."
                  % (mem, 100.0 * mem / self.driver.period, self.driver.period),
                  flush=True)

        # ---- clock persists across episodes (NS-3.4) -------------------------
        self.ns_clock = int(a.get("ns_phase0", 0))
        self._ns_reset_state()
        self.ns_n_steps = 0
        self.ns_n_dial_live = 0
        self.ns_rows_total = 0
        self.ns_tot_attempt_dmg = 0.0
        self.ns_tot_blocked_dmg = 0.0
        self.ns_tot_blocked = 0
        self._innov_ema = np.full(self.n_agents, np.nan)
        self._gram = np.zeros((self.coupling.r + 1, self.coupling.r + 1))
        self._gram_n = 0
        self._fg_lam = 0.999
        self._fg = dict(n=0, sse_full=0.0, sse_null=0.0, sse_fc=0.0, sst=0.0, ybar=0.0)
        print(self.driver.banner())
        print(self.coupling.banner())
        print("[SMAC-SA] sigma=%.3f on=%d mu=%.4f p0=%.1f clip=%.1f oracle=%d "
              "intercept_only=%d" % (self.ns_sigma, int(self.ns_on), self.ns_mu,
                                     self.ns_p0, self.ns_clip, int(self.ns_oracle),
                                     int(self.ns_intercept_only)))

    # ------------------------------------------------------------------ gates
    def _ns_startup_gates(self):
        """Abort on a wiring error -- never warn."""
        c = self.coupling
        rng = np.random.RandomState(0)
        for _ in range(32):
            t = rng.randint(-1, self.n_enemies, self.n_agents)
            f = (rng.rand(self.n_agents) < 0.8).astype(float)
            al = (rng.rand(self.n_agents) < 0.9).astype(float)
            dd = rng.randint(0, 4, (self.n_agents, self.n_enemies)).astype(float)
            for d in (None, dd):                   # integer distances force ties
                assert np.allclose(c.loading(t, al, f, d),
                                   c.loading_bruteforce(t, al, f, d), atol=1e-12), (
                    "GATE 1 FAILED: loading != brute force")
                assert np.allclose(c.psi(t, al, f, d), c.psi_bruteforce(t, al, f, d),
                                   atol=1e-12), "GATE 1 FAILED: basis != brute force"
        assert np.all(np.diag(c.W()) == 0.0), "GATE: operator diagonal is not zero"
        # GATE 3, in the real fleet: every agent alone on its own target reads
        # exactly zero harm at the harshest dial -- melee and ranged alike
        if self.n_enemies >= self.n_agents:
            t = np.arange(self.n_agents)
            u = c.loading(t, np.ones(self.n_agents), np.ones(self.n_agents))
            assert float(np.max(np.abs(c.excess(u, 1e-3)))) == 0.0, (
                "GATE 3 FAILED: a lone attacker is blocked -- this is category B")
        assert list(c.classes) == sorted(set(c.enemy_names)), "GATE 4 FAILED"

    # ------------------------------------------------------------------ state
    def _ns_reset_state(self):
        n, K = self.n_agents, self.n_actions
        self.ns_target = np.full(n, -1, dtype=np.int64)
        self.ns_prev_target = np.full(n, -1, dtype=np.int64)
        self.ns_fired = np.zeros(n)
        self.ns_alive_pre = np.zeros(n)
        self.ns_u = np.zeros(n)
        self.ns_excess = np.zeros(n)
        self.ns_p = np.zeros(n)
        self.ns_blocked_mask = np.zeros(n, dtype=bool)
        self.ns_y = np.full(n, np.nan)
        self.ns_psi_prev = np.zeros((n, self.coupling.r + 1))
        self.ns_cost = np.zeros((n, K))
        self.ns_cost_prev = np.zeros((n, K))
        self.ns_conf = np.zeros(n)
        self.ns_clip_hits = 0
        self.ns_rows = 0
        self.ns_gated = 0
        self.ns_gate_n = 0
        self.ns_step_gated = 0
        self.ns_step_gate_n = 0
        self.ns_spread = float("nan")
        self.ns_snr = float("nan")
        self.ns_switch_frac = float("nan")
        self.ns_dist_pre = None
        self.ns_t_now = None
        self.ns_g_now = None

    def reset(self, *args, **kwargs):
        # cleared BEFORE the host resets: StarCraft2Env builds the first
        # observation (and so the predicted-cost tail) inside its own reset()
        self._ns_reset_state()
        return super(SurfaceMixin, self).reset(*args, **kwargs)

    def ns_g(self, t=None):
        """Per-element capacity multiplier at step t (default: now)."""
        if not self.ns_on:
            return np.ones(self.n_enemies)
        tt = self.ns_clock if t is None else t
        return np.asarray(self.driver.g(tt, self.ns_sigma, self.ns_sens),
                          dtype=np.float64).reshape(self.n_enemies)

    def _ns_dist(self):
        """Ally-to-enemy distances from the units' positions, (n, m); None when the
        host's units carry no position or the symmetric ablation is selected."""
        if self.ns_slot_order != "nearest":
            return None
        try:
            ap = np.array([[float(self.agents[i].pos.x), float(self.agents[i].pos.y)]
                           for i in range(self.n_agents)])
            ep = np.array([[float(self.enemies[e].pos.x), float(self.enemies[e].pos.y)]
                           for e in range(self.n_enemies)])
        except (AttributeError, TypeError):
            return None
        return np.linalg.norm(ap[:, None, :] - ep[None, :, :], axis=2)

    def _alive(self):
        return np.array([1.0 if (self.agents.get(i) is not None
                                 and self.agents[i].health > 0) else 0.0
                         for i in range(self.n_agents)])

    def _enemy_alive(self):
        return np.array([1.0 if (self.enemies.get(e) is not None
                                 and self.enemies[e].health > 0) else 0.0
                         for e in range(self.n_enemies)])

    # ------------------------------------------------------------------ step
    def step(self, actions):
        intended = [int(np.asarray(x).reshape(-1)[0]) for x in actions]
        executed = self._ns_block(intended)
        out = super(SurfaceMixin, self).step(executed)
        self._ns_after(intended)
        if isinstance(out, tuple) and len(out) >= 5:
            out = list(out)
            infos = out[4]
            d = self.ns_info()
            try:
                for i in range(len(infos)):
                    if isinstance(infos[i], dict):
                        infos[i].update(d)
                        infos[i]["ns_blocked_i"] = float(self.ns_blocked_mask[i])
                        infos[i]["ns_target_i"] = float(self.ns_target[i])
            except (TypeError, IndexError):
                pass
            out[4] = infos
            out = tuple(out)
        return out

    def _ns_block(self, intended):
        """THE HARM: attack orders that find no free slot become STOP."""
        n = self.n_agents
        alive = self._alive()
        tgt = np.full(n, -1, dtype=np.int64)
        for i, act in enumerate(intended):
            if alive[i] > 0 and act >= N_ACTIONS_NO_ATTACK:
                tgt[i] = act - N_ACTIONS_NO_ATTACK
        fired = (tgt >= 0).astype(float)
        g = self.ns_g()
        self.ns_t_now, self.ns_g_now = int(self.ns_clock), g
        dist = self._ns_dist()                    # positions going INTO the step
        self.ns_dist_pre = dist
        u = self.coupling.loading(tgt, alive, fired, dist)
        g_i = self.coupling.g_eff(g, tgt)
        exc = self.coupling.excess(u, g_i)
        p = np.where(fired > 0, self.coupling.p_block(exc), 0.0)
        draws = self._ns_lottery.rand(n)        # always n draws: a fixed stream
        blocked = (fired > 0) & (draws < p)
        self.ns_target, self.ns_fired, self.ns_alive_pre = tgt, fired, alive
        self.ns_u, self.ns_excess, self.ns_p = u, exc, p
        self.ns_blocked_mask = blocked
        return [STOP if blocked[i] else int(intended[i]) for i in range(n)]

    def _ns_after(self, intended):
        """Sense, identify, predict -- after the engine has played the step."""
        n = self.n_agents
        tgt, fired, alive = self.ns_target, self.ns_fired, self.ns_alive_pre
        exc = self.ns_excess
        act_m = fired > 0

        # ---- the sensor (P-2.1): the relative excess cost of landing an attack
        y = np.full(n, np.nan)
        if act_m.any():
            y[act_m] = relative_excess(1.0 + exc[act_m], np.ones(int(act_m.sum())),
                                       clip=self.ns_clip)
            self.ns_clip_hits += int(np.sum(exc[act_m] > self.ns_clip))
        self.ns_y = y

        # ---- identify: one RLS row per attacking agent, scored BEFORE updating
        psi = self.coupling.psi(tgt, alive, fired, self.ns_dist_pre)
        if self.ns_intercept_only:
            psi[:, 1:] = 0.0
        for i in range(n):
            if act_m[i] and np.isfinite(y[i]):
                pf = float(self._rls[i].predict(psi[i]))
                pn = float(self._rls[i].beta[0])
                a_i = int(intended[i])
                pc = (float(self.ns_cost_prev[i, a_i])
                      if a_i >= N_ACTIONS_NO_ATTACK else np.nan)
                self._fg_observe(y[i], pf, pn, pc)
                self._rls[i].update(psi[i][None, :], np.array([y[i]]))
                self.ns_rows += 1
                self.ns_rows_total += 1
                e_abs = abs(float(y[i]) - pf)
                self._innov_ema[i] = (e_abs if not np.isfinite(self._innov_ema[i])
                                      else 0.99 * self._innov_ema[i] + 0.01 * e_abs)
                self._gram = 0.999 * self._gram + 0.001 * np.outer(psi[i], psi[i])
                self._gram_n += 1
            self.ns_conf[i] = rls_confidence_pred(self._rls[i].P, self.ns_p0,
                                                  self.coupling.r + 1, psi[i])
        self.ns_psi_prev = psi

        # ---- accounting: damage-weighted share of attack attempts blocked -----
        d = self.coupling.dmg
        self.ns_harm_den = float(np.sum(d[act_m]))
        self.ns_harm_num = float(np.sum(d[self.ns_blocked_mask]))
        self.ns_tot_attempt_dmg += self.ns_harm_den
        self.ns_tot_blocked_dmg += self.ns_harm_num
        self.ns_tot_blocked += int(self.ns_blocked_mask.sum())

        # ---- predict every option's cost for the NEXT decision ----------------
        # peers are forecast to hold this step's targets; only survivors count
        alive_next = self._alive()
        fired_next = ((tgt >= 0) & (alive_next > 0)).astype(float)
        self._ns_predict_costs(tgt, alive_next, fired_next)

        both = (tgt >= 0) & (self.ns_prev_target >= 0)
        self.ns_switch_frac = (float(np.mean(tgt[both] != self.ns_prev_target[both]))
                               if both.any() else float("nan"))
        self.ns_prev_target = tgt.copy()
        self.ns_cost_prev = self.ns_cost.copy()
        self.ns_n_dial_live += int(float(np.min(self.ns_g_now)) < 1.0)
        self.ns_clock += 1
        self.ns_n_steps += 1

    def _ns_predict_costs(self, tgt, alive, fired):
        """``cost_hat[i, k]`` for every attack option -- the observation tail.

        Stop/move rows stay zero and the policy excludes them (pact_steer_from).
        The floor is applied by the policy, which alone knows the in-range mask;
        what the layer reports as gated is the same test over every living enemy.
        """
        A0 = N_ACTIONS_NO_ATTACK
        self.ns_cost[:] = 0.0
        dist = self._ns_dist()                    # positions the NEXT decision faces
        if self.ns_oracle:
            U = self.coupling.loading_all_options(tgt, alive, fired, dist)
            g_next = self.coupling.g_eff_all(self.ns_g(self.ns_clock + 1))
            pred = self.coupling.excess(U, g_next)
        else:
            Phi = self.coupling.psi_all_options(tgt, alive, fired, dist)
            if self.ns_intercept_only:
                Phi[:, :, 1:] = 0.0
            B = np.stack([r.beta for r in self._rls])
            pred = np.einsum("ikj,ij->ik", Phi, B)
        live_e = self._enemy_alive() > 0
        spreads, snrs = [], []
        self.ns_step_gated = 0
        self.ns_step_gate_n = 0
        for i in range(self.n_agents):
            if alive[i] <= 0:
                continue
            self.ns_cost[i, A0:] = pred[i]
            c = pred[i][live_e]
            if c.size < 2:
                continue
            sd = float(c.std())
            spreads.append(sd)
            if np.isfinite(self._innov_ema[i]) and self._innov_ema[i] > 0:
                snrs.append(sd / float(self._innov_ema[i]))
            self.ns_gate_n += 1
            self.ns_step_gate_n += 1
            if sd < self.ns_steer_floor:
                self.ns_gated += 1
                self.ns_step_gated += 1
        self.ns_spread = float(np.mean(spreads)) if spreads else float("nan")
        self.ns_snr = float(np.mean(snrs)) if snrs else float("nan")

    # ------------------------------------------------------------ fit gain
    def _fg_observe(self, y, pred_full, pred_null, pred_forecast):
        f = self._fg
        lam = self._fg_lam
        f["n"] += 1
        f["ybar"] = lam * f["ybar"] + (1 - lam) * float(y)
        if f["n"] <= 2000:
            return
        f["sse_full"] = lam * f["sse_full"] + (1 - lam) * (y - pred_full) ** 2
        f["sse_null"] = lam * f["sse_null"] + (1 - lam) * (y - pred_null) ** 2
        f["sst"] = lam * f["sst"] + (1 - lam) * (y - f["ybar"]) ** 2
        if np.isfinite(pred_forecast):
            f["sse_fc"] = lam * f["sse_fc"] + (1 - lam) * (y - pred_forecast) ** 2

    def _fg_value(self, key):
        f = self._fg
        if f["n"] <= 2000 or f["sst"] <= 0.0:
            return float("nan")
        return float((f["sse_null"] - f[key]) / f["sst"])

    # ------------------------------------------------------------ observation
    def get_obs_agent(self, agent_id):
        o = super(SurfaceMixin, self).get_obs_agent(agent_id)
        if not self.ns_augment:
            return o
        cost = (np.zeros(self.n_actions, dtype=np.float32)
                if self.ns_cost is None else self.ns_cost[agent_id])
        return np.concatenate([np.asarray(o, dtype=np.float32),
                               np.asarray(cost, dtype=np.float32)])

    def get_obs_size(self):
        sz = super(SurfaceMixin, self).get_obs_size()
        if not self.ns_augment:
            return sz
        sz = list(sz)
        sz[0] = int(sz[0]) + self.n_actions
        return sz

    # ------------------------------------------------------------------ report
    def ns_info(self):
        """II.10's instrument panel.  Ratios are NaN-guarded, never epsilon'd."""
        live = self.ns_fired > 0
        nz = int(live.sum())
        alive = self._alive()
        melee = self.coupling.melee

        def m(x, mask=live):
            v = np.asarray(x, dtype=np.float64)[mask] if mask.any() else np.array([])
            v = v[np.isfinite(v)]
            return float(v.mean()) if v.size else float("nan")

        def mx(x, mask=live):
            v = np.asarray(x, dtype=np.float64)[mask] if mask.any() else np.array([])
            v = v[np.isfinite(v)]
            return float(v.max()) if v.size else float("nan")

        t_h = self.ns_clock if self.ns_t_now is None else self.ns_t_now
        g = self.ns_g(t_h) if self.ns_g_now is None else self.ns_g_now
        beta = np.mean([r.beta for r in self._rls], axis=0)
        x = self.ns_psi_prev[:, 1:]
        tg = self.ns_target
        cnt = np.bincount(tg[tg >= 0], minlength=self.n_enemies).astype(np.float64)
        mt = tg[(tg >= 0) & melee]
        mcnt = np.bincount(mt, minlength=self.n_enemies).astype(np.float64)
        cond = float("nan")
        if self._gram_n > 50:
            try:
                cc = float(np.linalg.cond(self._gram))
                cond = cc if np.isfinite(cc) else float("inf")
            except np.linalg.LinAlgError:
                cond = float("inf")
        melee_live = live & melee
        info = {
            "ns_sigma": float(self.ns_sigma),
            "ns_on": float(self.ns_on),
            "ns_A": float(self.driver.A(t_h)),
            "ns_g": float(np.mean(g)),
            "ns_g_min": float(np.min(g)),
            "ns_placebo": float(bool(self.driver.is_placebo(t_h))),
            "ns_dial_ratio": (float(self.ns_n_dial_live) / self.ns_n_steps
                              if self.ns_n_steps else float("nan")),
            "ns_clock": float(t_h),
            "ns_attempts": float(nz),
            "ns_blocked": float(np.sum(self.ns_blocked_mask)),
            "ns_harm_num": float(getattr(self, "ns_harm_num", 0.0)),
            "ns_harm_den": float(getattr(self, "ns_harm_den", 0.0)),
            "ns_block_frac_cum": (self.ns_tot_blocked_dmg / self.ns_tot_attempt_dmg
                                  if self.ns_tot_attempt_dmg > 0 else float("nan")),
            "ns_melee_attempts": float(np.sum(melee_live)),
            "ns_melee_blocked": float(np.sum(self.ns_blocked_mask & melee)),
            "ns_p_block": m(self.ns_p),
            "ns_p_block_max": mx(self.ns_p),
            "ns_u": m(self.ns_u),
            "ns_u_max": mx(self.ns_u),
            "ns_u_melee": m(self.ns_u, melee_live),
            "ns_excess": m(self.ns_excess),
            "ns_excess_max": mx(self.ns_excess),
            "ns_y": m(self.ns_y),
            "ns_fire_frac": (float(nz) / float(np.sum(alive))
                             if np.sum(alive) > 0 else float("nan")),
            "ns_alive": float(np.sum(alive)),
            "ns_herd_index": float(herd_index(cnt)),
            "ns_melee_herd_index": float(herd_index(mcnt)) if mcnt.sum() > 0
            else float("nan"),
            "ns_switch_frac": float(self.ns_switch_frac),
            "ns_fit_gain": self._fg_value("sse_full"),
            "ns_pred_gain": self._fg_value("sse_fc"),
            "ns_cond_psi": cond,
            "ns_conf": float(np.mean(self.ns_conf)),
            "ns_rows": float(self.ns_rows),
            "ns_clip_frac": float(self.ns_clip_hits) / max(1, self.ns_rows),
            "ns_innov": float(np.mean([r.innov for r in self._rls])),
            "ns_trP": float(np.mean([np.trace(r.P) for r in self._rls])),
            "ns_x_std": float(np.std(x)),
            "ns_x_mean": float(np.mean(x)),
            "ns_cost_std": float(np.std(self.ns_cost[:, N_ACTIONS_NO_ATTACK:])),
            "ns_cost_spread": float(self.ns_spread),
            "ns_rank_snr": float(self.ns_snr),
            "ns_steer_gated_frac": (float(self.ns_gated) / self.ns_gate_n
                                    if self.ns_gate_n else float("nan")),
            "ns_step_gated": float(self.ns_step_gated),
            "ns_step_gate_n": float(self.ns_step_gate_n),
            "ns_mem_frac": ((1.0 / max(1e-12, 1.0 - self.ns_mu))
                            / (float(self.ns_rows_total)
                               / float(self.n_agents * self.ns_n_steps))
                            / float(self.driver.period)
                            if self.ns_rows_total > 0 and self.ns_n_steps > 0
                            else float("nan")),
        }
        for k in range(4):
            info["ns_beta%d" % k] = float(beta[k]) if k < len(beta) else float("nan")
        return info

    def ns_close_report(self):
        ok = (self.ns_sigma <= 0.0) or self.ns_tot_blocked > 0
        print("[SMAC-SA] steps=%d  attacks blocked=%d  damage-weighted block frac=%.4f"
              % (self.ns_n_steps, self.ns_tot_blocked,
                 self.ns_tot_blocked_dmg / max(1e-9, self.ns_tot_attempt_dmg)))
        if not ok:
            print("[SMAC-SA][REFUSE] severity is %.3f but no attack was ever blocked. "
                  "This run MUST NOT be reported as a severity arm (NS-3.3)."
                  % self.ns_sigma)
        return ok


def make_smac_sa_env(args):
    """Build ``SmacSaEnv`` -- the host with the surface-area layer mixed in."""
    from ..StarCraft2_Env import StarCraft2Env

    class SmacSaEnv(SurfaceMixin, StarCraft2Env):
        pass

    return SmacSaEnv(args)
