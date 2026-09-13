"""The severity layer for LANE SWERVE UNDER DRIFT -- SMAC's (C, invertible) cell.

    StarCraft2Env                          stock
      +-- SmacLaneEnv(LaneMixin, ..)       THIS FILE.  Every arm gets it.
            (the compensator lives INSIDE the layer: II.6's invertible row)

Per step, before the engine plays it:

    h_i        the compass direction unit i was ORDERED to move (none if it did
               not order a move)
    x_i, s_i   its lane load per sector and the signed side of that load (j != i)
    Q_i, S_i   their public, rho-filtered memory
    d_i        = sigma*L*A(t) * send . Q_i / load_norm      the swerve, radians
    e_i        = -sign(S_i)                                  away from the traffic
    c_i        the compensation: 0 (blind / pactoff), g * clip(beta_hat . psi_i)
               (pact), d_i (oracle)
    phi_i      = e_i * (d_i - c_i)            the rotation applied to the move order

The move order's world point (x +- 2 or y +- 2) is rotated by phi_i and the stock
engine plays the step.  Nothing else in SMAC changes: attacks, stops, the reward,
the observation, the termination test are the host's own.

THE SENSOR IS PROPRIOCEPTION AT THE ACTUATOR.  The order the engine executed,
measured against the order that was sent (already pre-rotated by -e*c), along the
public direction, is exactly d_i -- the swerve the unit experienced.  The
estimator regresses it on psi_i; the model is exactly linear.

    sigma = 0 or A(t) = 0  ->  phi == 0: the executed order IS the chosen order
    unit alone in its lane ->  phi == 0 at any severity (category C)
    oracle (c = d)         ->  phi == 0: compensation cancels the disturbance
    trust 0 (pactoff)      ->  identical to blind, action for action
"""

import numpy as np

from ..smac_ns.pact1_core import AgentRLS, rls_confidence_pred
from .coupling import MOVE_VEC, LaneCoupling, heading_of, rotate
from .driver import GroundDriver, L_SIGMA1

#: every key ns_info() emits, in debug-file order.  The runner imports it.
NS_KEYS = [
    # ---- is the DIAL live? ----------------------------------------------------
    "ns_sigma", "ns_on", "ns_A", "ns_amp", "ns_placebo", "ns_dial_ratio", "ns_clock",
    # ---- did the harm REACH the game? ------------------------------------------
    "ns_moves", "ns_lane_frac", "ns_d_deg", "ns_d_max_deg", "ns_exec_deg",
    "ns_harm_num", "ns_harm_den", "ns_corr_clipped",
    # ---- is the MEDIUM loaded? --------------------------------------------------
    "ns_x_front", "ns_x_flank", "ns_Q_front", "ns_Q_flank", "ns_spread",
    # ---- is the METHOD working? -------------------------------------------------
    "ns_fit_gain", "ns_beta_cos", "ns_beta_relerr", "ns_cond_psi", "ns_conf",
    "ns_rows", "ns_innov", "ns_trP", "ns_x_std", "ns_mem_frac", "ns_trust",
    "ns_beta0", "ns_beta1", "ns_beta2",
]


def rotated_target(x, y, action, phi, amount):
    """World point of SMAC move ``action`` (2..5) from (x, y), rotated by ``phi``."""
    v = MOVE_VEC[int(action) - 2][None, :] * float(amount)
    w = rotate(v, np.array([float(phi)]))[0]
    return float(x + w[0]), float(y + w[1])


class LaneMixin(object):
    """The dial and the compensator.  Mix in ABOVE ``StarCraft2Env``."""

    def __init__(self, args, **kwargs):
        a = dict(args or {})
        n_before = None
        self.ns_phi = None                      # get_agent_action guards on this
        super(LaneMixin, self).__init__(args, **kwargs)
        n = self.n_agents

        self.ns_sigma = float(a.get("ns_severity", 1.0))
        self.ns_on = bool(int(a.get("ns_on", 1)))
        self.driver = GroundDriver(
            period=int(a.get("ns_period", 100 * self.episode_limit)),
            guard_frac=float(a.get("ns_guard_frac", 0.5)),
            loss=float(a.get("ns_loss", L_SIGMA1)),
            mean_preserving=bool(int(a.get("ns_mean_preserving", 0))))
        self.driver.certify()
        send = a.get("ns_send", (1.4, 0.6))
        if isinstance(send, str):
            send = [float(v) for v in send.split(",")]
        self.coupling = LaneCoupling(n, kernel_lambda=float(a.get("ns_kernel_lambda", 2.0)),
                                     rho=float(a.get("ns_rho", 0.5)), send=send)
        self.ns_ref, self.ns_scale, self.ns_load_norm = self.coupling.reference(
            float(a.get("ns_spacing", 1.5)))
        self._ns_gates()

        # ---- the compensator: the PACT family only; a baseline never sets it --
        self.ns_pact = bool(int(a.get("ns_pact", 0)))
        self.ns_g = float(a.get("ns_g_fixed", 0.9))          # P-5.1's prior, fixed
        self.ns_oracle = bool(int(a.get("ns_oracle", 0)))
        self.ns_intercept_only = bool(int(a.get("ns_intercept_only", 0)))
        # past a quarter turn the "correction" is running the wrong way: relief valve
        self.ns_corr_clip = float(a.get("ns_corr_clip", np.pi / 2))

        # ---- the estimator (URB's core, verbatim) ------------------------------
        self.ns_mu = float(a.get("ns_mu", 0.99))
        self.ns_p0 = float(a.get("ns_p0", 10.0))
        self._rls = [AgentRLS(self.coupling.r + 1, mu=self.ns_mu, p0=self.ns_p0)
                     for _ in range(n)]
        mem = 1.0 / max(1e-12, 1.0 - self.ns_mu)
        if mem > 0.25 * self.driver.period:
            print("[SMAC-LANE][NOT-TRACKABLE] estimator memory >= %.0f steps is %.0f%% "
                  "of the %d-step period" % (mem, 100 * mem / self.driver.period,
                                             self.driver.period), flush=True)

        self.ns_clock = int(a.get("ns_phase0", 0))
        self.ns_Q = np.zeros((n, self.coupling.r))
        self.ns_S = np.zeros(n)
        self._ns_clear_step()
        self.ns_n_steps = 0
        self.ns_n_dial_live = 0
        self.ns_rows_total = 0
        self.ns_n_clipped = 0
        self._gram = np.zeros((self.coupling.r + 1, self.coupling.r + 1))
        self._gram_n = 0
        self._fg = dict(n=0, sse_full=0.0, sse_null=0.0, sst=0.0, ybar=0.0)
        self._bc = dict(dot=0.0, nh=0.0, nt=0.0, err=0.0, nrm=0.0)
        print(self.driver.banner())
        print("[SMAC-LANE] lane lambda=%.2f rho=%.2f send=%s load_norm=%.4f | pact=%d "
              "g=%.2f oracle=%d intercept=%d | sigma=%.2f mu=%.3f"
              % (self.coupling.lam, self.coupling.rho, list(self.coupling.send),
                 self.ns_load_norm, int(self.ns_pact), self.ns_g, int(self.ns_oracle),
                 int(self.ns_intercept_only), self.ns_sigma, self.ns_mu))

    # ------------------------------------------------------------------ gates
    def _ns_gates(self):
        c, n = self.coupling, self.n_agents
        rng = np.random.RandomState(0)
        for _ in range(32):
            pos = rng.uniform(0, 8, (n, 2))
            head = MOVE_VEC[rng.randint(0, 4, n)] * (rng.rand(n) < 0.8)[:, None]
            al = (rng.rand(n) < 0.9).astype(float)
            x, s = c.channels(pos, head, al)
            xb, sb = c.channels_bruteforce(pos, head, al)
            assert np.allclose(x, xb, atol=1e-12) and np.allclose(s, sb, atol=1e-12), (
                "GATE 1 FAILED: vectorised lane channels != brute-force definition")
        solo = LaneCoupling(1, c.lam, c.rho, c.send)
        x1, s1 = solo.channels(np.zeros((1, 2)), MOVE_VEC[:1], np.ones(1))
        assert float(np.abs(x1).max()) == 0.0 and float(np.abs(s1).max()) == 0.0, (
            "GATE 3 FAILED: a lone unit has a loaded lane -- this is category B")
        v = MOVE_VEC.copy()
        back = rotate(rotate(v, np.full(4, 0.7)), np.full(4, -0.7))
        assert np.allclose(back, v, atol=1e-12), "GATE: rotation is not invertible"

    # ------------------------------------------------------------------ state
    def _ns_clear_step(self):
        n = self.n_agents
        self.ns_phi = np.zeros(n)
        self.ns_d = np.zeros(n)
        self.ns_c = np.zeros(n)
        self.ns_e = np.zeros(n)
        self.ns_moving = np.zeros(n, dtype=bool)
        self.ns_x = np.zeros((n, self.coupling.r))
        self.ns_psi = np.zeros((n, self.coupling.r + 1))
        self.ns_beta_star = np.zeros(self.coupling.r)
        self.ns_t_now = None

    def reset(self, *args, **kwargs):
        # NS-3.4: the clock and the estimator persist; the lane memory does not
        # (the squad respawns), and it is cleared before the host's reset builds
        # anything
        n = self.n_agents
        self.ns_Q = np.zeros((n, self.coupling.r))
        self.ns_S = np.zeros(n)
        self._ns_clear_step()
        return super(LaneMixin, self).reset(*args, **kwargs)

    # ------------------------------------------------------------------ step
    def _ns_positions(self):
        n = self.n_agents
        pos = np.zeros((n, 2))
        alive = np.zeros(n)
        for i in range(n):
            u = self.agents.get(i)
            if u is not None and u.health > 0:
                pos[i] = (float(u.pos.x), float(u.pos.y))
                alive[i] = 1.0
        return pos, alive

    def step(self, actions):
        acts = [int(np.asarray(x).reshape(-1)[0]) for x in actions]
        self._ns_pre(acts)
        out = super(LaneMixin, self).step(actions)
        self._ns_post()
        if isinstance(out, tuple) and len(out) >= 5:
            out = list(out)
            d = self.ns_info()
            try:
                for i in range(len(out[4])):
                    if isinstance(out[4][i], dict):
                        out[4][i].update(d)
                        out[4][i]["ns_phi_i"] = float(self.ns_phi[i])
            except (TypeError, IndexError):
                pass
            out = tuple(out)
        return out

    def _ns_pre(self, acts):
        """Everything the move orders of THIS step need, before the engine runs."""
        c, n = self.coupling, self.n_agents
        pos, alive = self._ns_positions()
        head = heading_of(acts, n) * alive[:, None]
        x, s = c.channels(pos, head, alive)
        self.ns_Q, self.ns_S = c.filter(self.ns_Q, self.ns_S, x, s)
        t = int(self.ns_clock)
        amp = self.driver.amp(t, self.ns_sigma) if self.ns_on else 0.0
        beta_star = amp * c.send / self.ns_load_norm                  # rad per unit Q
        moving = (np.abs(head).sum(1) > 0) & (alive > 0)
        d = np.where(moving, self.ns_Q @ beta_star, 0.0)
        e = np.where(moving, -np.sign(self.ns_S), 0.0)
        psi = c.design(self.ns_Q, self.ns_ref, self.ns_scale)
        if self.ns_intercept_only:
            psi[:, 1:] = 0.0
        comp = np.zeros(n)
        if self.ns_pact:
            if self.ns_oracle:
                raw = d.copy()
                g = 1.0                     # the ceiling: full reliance on the truth
            else:
                raw = np.array([float(self._rls[i].predict(psi[i])) for i in range(n)])
                g = self.ns_g
            clipped = raw > self.ns_corr_clip
            self.ns_n_clipped += int(np.sum(clipped & moving))
            comp = g * np.clip(raw, 0.0, self.ns_corr_clip)
        comp = np.where(moving & (e != 0.0), comp, 0.0)
        self.ns_t_now = t
        self.ns_x, self.ns_psi, self.ns_beta_star = x, psi, beta_star
        self.ns_d, self.ns_e, self.ns_c = d, e, comp
        self.ns_moving = moving
        self.ns_phi = e * (d - comp)

    def get_agent_action(self, a_id, action):
        """The one change to the host's action path: a move order is issued to the
        rotated world point.  Every other action is the host's own."""
        phi = 0.0 if self.ns_phi is None else float(self.ns_phi[a_id])
        if 2 <= int(action) <= 5 and phi != 0.0:
            avail = self.get_avail_agent_actions(a_id)
            assert avail[action] == 1, "Agent %d cannot perform action %d" % (a_id, action)
            unit = self.get_unit_by_id(a_id)
            tx, ty = rotated_target(unit.pos.x, unit.pos.y, action, phi, self._move_amount)
            return self._ns_move_command(unit, tx, ty)
        return super(LaneMixin, self).get_agent_action(a_id, action)

    def _ns_move_command(self, unit, tx, ty):
        """SMAC's own move command to a world point (ability 16, not queued)."""
        from s2clientprotocol import common_pb2 as sc_common
        from s2clientprotocol import raw_pb2 as r_pb
        from s2clientprotocol import sc2api_pb2 as sc_pb
        cmd = r_pb.ActionRawUnitCommand(
            ability_id=16, target_world_space_pos=sc_common.Point2D(x=tx, y=ty),
            unit_tags=[unit.tag], queue_command=False)
        return sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))

    def _ns_post(self):
        """Sense, identify, account -- after the engine has played the step."""
        n = self.n_agents
        m = self.ns_moving & (self.ns_e != 0.0)
        # proprioception at the actuator: executed vs SENT, along the public side
        y = np.where(m, self.ns_e * self.ns_phi + self.ns_c, np.nan)
        beta_true = np.concatenate([[float(self.ns_beta_star @ self.ns_ref)],
                                    self.ns_beta_star * self.ns_scale])
        for i in range(n):
            if not m[i]:
                continue
            psi = self.ns_psi[i]
            pf = float(self._rls[i].predict(psi))
            pn = float(self._rls[i].beta[0])
            f = self._fg
            f["n"] += 1
            f["ybar"] = 0.999 * f["ybar"] + 0.001 * float(y[i])
            if f["n"] > 500:
                f["sse_full"] = 0.999 * f["sse_full"] + 0.001 * (y[i] - pf) ** 2
                f["sse_null"] = 0.999 * f["sse_null"] + 0.001 * (y[i] - pn) ** 2
                f["sst"] = 0.999 * f["sst"] + 0.001 * (y[i] - f["ybar"]) ** 2
            self._rls[i].update(psi[None, :], np.array([float(y[i])]))
            self.ns_rows_total += 1
            self._gram = 0.999 * self._gram + 0.001 * np.outer(psi, psi)
            self._gram_n += 1
        if float(self.driver.A(self.ns_t_now)) > 0.25 and m.any():
            bh = np.mean([self._rls[i].beta for i in np.where(m)[0]], axis=0)
            b = self._bc
            b["dot"] = 0.99 * b["dot"] + 0.01 * float(bh @ beta_true)
            b["nh"] = 0.99 * b["nh"] + 0.01 * float(bh @ bh)
            b["nt"] = 0.99 * b["nt"] + 0.01 * float(beta_true @ beta_true)
            b["err"] = 0.99 * b["err"] + 0.01 * float(np.sum((bh - beta_true) ** 2))
            b["nrm"] = b["nt"]
        self.ns_n_dial_live += int(self.driver.A(self.ns_t_now) > 0.0 and self.ns_on
                                   and self.ns_sigma > 0)
        self.ns_clock += 1
        self.ns_n_steps += 1

    # ------------------------------------------------------------------ report
    def ns_info(self):
        n = self.n_agents
        mv = self.ns_moving
        nm = int(mv.sum())
        pos, alive = self._ns_positions()
        t = self.ns_clock if self.ns_t_now is None else self.ns_t_now
        amp = self.driver.amp(t, self.ns_sigma) if self.ns_on else 0.0
        lane = (self.ns_x.sum(1) > 0) & mv
        beta = np.mean([r.beta for r in self._rls], axis=0)
        al = np.where(alive > 0)[0]
        spread = (float(np.mean([np.linalg.norm(pos[i] - pos[j])
                                 for i in al for j in al if i < j]))
                  if al.size >= 2 else float("nan"))
        cond = float("nan")
        if self._gram_n > 50:
            try:
                cc = float(np.linalg.cond(self._gram))
                cond = cc if np.isfinite(cc) else float("inf")
            except np.linalg.LinAlgError:
                cond = float("inf")
        f, b = self._fg, self._bc
        conf = [rls_confidence_pred(self._rls[i].P, self.ns_p0, self.coupling.r + 1,
                                    self.ns_psi[i]) for i in range(n)]

        def mdeg(v, mask):
            v = np.asarray(v, dtype=np.float64)[mask]
            return float(np.degrees(np.mean(np.abs(v)))) if v.size else float("nan")

        info = {
            "ns_sigma": float(self.ns_sigma),
            "ns_on": float(self.ns_on),
            "ns_A": float(self.driver.A(t)),
            "ns_amp": float(amp),
            "ns_placebo": float(bool(self.driver.is_placebo(t))),
            "ns_dial_ratio": (float(self.ns_n_dial_live) / self.ns_n_steps
                              if self.ns_n_steps else float("nan")),
            "ns_clock": float(t),
            "ns_moves": float(nm),
            "ns_lane_frac": float(lane.sum()) / nm if nm else float("nan"),
            "ns_d_deg": mdeg(self.ns_d, mv),
            "ns_d_max_deg": (float(np.degrees(np.max(np.abs(self.ns_d[mv]))))
                             if nm else float("nan")),
            "ns_exec_deg": mdeg(self.ns_phi, mv),
            "ns_harm_num": float(np.sum(np.abs(self.ns_phi[mv]))),
            "ns_harm_den": float(nm),
            "ns_corr_clipped": float(self.ns_n_clipped),
            "ns_x_front": float(self.ns_x[mv, 0].mean()) if nm else float("nan"),
            "ns_x_flank": float(self.ns_x[mv, 1].mean()) if nm else float("nan"),
            "ns_Q_front": float(self.ns_Q[:, 0].mean()),
            "ns_Q_flank": float(self.ns_Q[:, 1].mean()),
            "ns_spread": spread,
            "ns_fit_gain": (float((f["sse_null"] - f["sse_full"]) / f["sst"])
                            if f["n"] > 500 and f["sst"] > 0 else float("nan")),
            "ns_beta_cos": (float(b["dot"] / np.sqrt(b["nh"] * b["nt"]))
                            if b["nh"] > 0 and b["nt"] > 0 else float("nan")),
            "ns_beta_relerr": (float(np.sqrt(b["err"] / b["nrm"]))
                               if b["nrm"] > 0 else float("nan")),
            "ns_cond_psi": cond,
            "ns_conf": float(np.mean(conf)),
            "ns_rows": float(self.ns_rows_total),
            "ns_innov": float(np.mean([r.innov for r in self._rls])),
            "ns_trP": float(np.mean([np.trace(r.P) for r in self._rls])),
            "ns_x_std": float(np.std(self.ns_psi[:, 1:])),
            "ns_mem_frac": ((1.0 / max(1e-12, 1.0 - self.ns_mu))
                            / (float(self.ns_rows_total) / float(n * self.ns_n_steps))
                            / float(self.driver.period)
                            if self.ns_rows_total > 0 and self.ns_n_steps > 0
                            else float("nan")),
            "ns_trust": (1.0 if self.ns_oracle else self.ns_g) if self.ns_pact else 0.0,
        }
        for k in range(3):
            info["ns_beta%d" % k] = float(beta[k])
        return info

    def ns_close_report(self):
        ok = self.ns_sigma <= 0.0 or self.ns_n_dial_live > 0
        print("[SMAC-LANE] steps=%d dial_live=%d rows=%d corr_clipped=%d"
              % (self.ns_n_steps, self.ns_n_dial_live, self.ns_rows_total,
                 self.ns_n_clipped))
        if not ok:
            print("[SMAC-LANE][REFUSE] severity %.2f but the dial never went live."
                  % self.ns_sigma)
        return ok


def make_smac_lane_env(args):
    from ..StarCraft2_Env import StarCraft2Env

    class SmacLaneEnv(LaneMixin, StarCraft2Env):
        pass

    return SmacLaneEnv(args)
