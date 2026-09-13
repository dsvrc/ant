"""The declared operator for SURFACE AREA UNDER DRIFT.

    medium     the ring of contact positions around each enemy unit
    element    an enemy unit e
    capacity   how many melee attackers fit on e's ring at contact:
                   cap_i(e) = 2*pi*(r_e + r_i) / (2*r_i)
               from published unit radii -- never fitted
    loading    u_i = sum_{j != i, melee, same target, j NEARER e than i} (r_j/r_i) / cap_i(e)
               and u_i = 0 EXACTLY for a ranged attacker: it shoots from range and
               stands on no ring

SLOTS GO TO THE NEAREST.  Whoever reaches the ring first takes a position; the
attackers who arrive last find it full.  So an attacker is loaded only by the
melee peers on its target that are closer to that target than it is (ties to the
lower index).  The nearest attacker is never blocked.  This is the physics, and it
is also what lets independent agents coordinate: on a crowded target the far
zealot reads a full ring and the near ones do not, so the far one leaves and the
near ones stay -- the allocation a centralized planner would pick, reached without
one.  With every attacker loaded by every peer, all of them read the same cost and
leave or stay together; offline, that capped any decentralized recovery near zero
while a centralized allocator recovered all of it.
    function   BPR, the standard congestion curve:  f(u) = 1 + alpha * u**beta,
               alpha = 0.15, beta = 4

"Zealots need surface area" is everyday StarCraft: past a handful of attackers on
one unit, the rest cannot reach it and stand behind the others doing nothing.
Ranged units do not compete for contact, and a tightening enemy line denies
contact, not line of fire -- so the constraint falls on melee only.  (An earlier
draft gave ranged units a weapon-range arc; with the ring nearly closed the
g**-4 factor let even a 36-slot arc block Stalkers, which is not the physics.)

AT LEAST ONE SLOT STAYS OPEN.  However tightly the enemy stands, one attacker can
always reach a unit, so the effective capacity never drops below one slot:
g_eff = max(g, 1/cap).  A lone attacker is unaffected regardless (u = 0).

WHY THIS AND NOT OVERKILL.  The overkill instance wasted a fraction of every
shared attack, so each extra attacker still added damage and concentrating always
beat spreading -- measured offline, the harder a blind host focused the MORE it
won at the peak of the harm (3s5z, sigma=5: focus 1 -> 0.00, 2 -> 0.03, 4 -> 0.28,
8 -> 0.65).  No steering channel can recover a harm whose best response is the
host's own behaviour.  Here an attacker past capacity lands NOTHING, so moving it
to an open target costs no kill speed at all: the best response under the drift
is to spread past the cap, and it differs from the best response without it.

THE HARM RATIO IS EXACT IN SHARE UNITS
    excess = f(u/g) / f(u) - 1 = (g**-beta - 1) * s(u),   s(u) = alpha u**beta / (1 + alpha u**beta)
so ``y = beta_hat . psi`` with psi built from s(u) is not an approximation, and the
estimator's slope IS the driver factor (g**-beta - 1).
"""

import numpy as np

from ..smac_ns.coupling import UNIT_STATS, roster

#: published StarCraft II unit radii (world units)
RADIUS = {
    "Zealot": 0.5, "Stalker": 0.625, "Colossus": 1.0, "Marine": 0.375,
    "Marauder": 0.5625, "Medivac": 0.75, "Hydralisk": 0.625, "Zergling": 0.375,
    "Baneling": 0.375,
}
#: a weapon range at or below this fights in contact (Zealot 0.1, Zergling 0.1)
MELEE_RANGE = 1.0
#: the Bureau of Public Roads congestion function, the family URB's links use
BPR_ALPHA, BPR_BETA = 0.15, 4.0
#: bound on the loading for the REGRESSOR; far past capacity s(u) is ~1 anyway
U_CLIP = 5.0


class SlotCoupling(object):
    """Capacities, peer weights, classes and the share basis for one map."""

    U_CLIP = U_CLIP

    def __init__(self, map_name, n_agents, n_enemies, alpha=BPR_ALPHA, beta=BPR_BETA,
                 min_share=0.05):
        self.map_name = str(map_name)
        self.n, self.m = int(n_agents), int(n_enemies)
        self.alpha, self.beta = float(alpha), float(beta)
        self.ally_names = roster(self.map_name, self.n, "ally")
        self.enemy_names = roster(self.map_name, self.m, "enemy")
        self.r_ally = np.array([RADIUS[x] for x in self.ally_names], dtype=np.float64)
        self.r_enemy = np.array([RADIUS[x] for x in self.enemy_names], dtype=np.float64)
        rng_ally = np.array([UNIT_STATS[x]["rng"] for x in self.ally_names],
                            dtype=np.float64)
        self.melee = rng_ally <= MELEE_RANGE                          # (n,)
        self.dmg = np.array([UNIT_STATS[x]["damage"] / UNIT_STATS[x]["cd"]
                             for x in self.ally_names], dtype=np.float64)

        # capacity of enemy e for melee attacker i: ring circumference / diameter
        ring = self.r_enemy[None, :] + self.r_ally[:, None]
        self.cap0 = np.pi * ring / self.r_ally[:, None]               # (n, m)
        # one slot always stays open: the driver cannot close the ring below it
        self.g_floor = np.where(self.melee[:, None], 1.0 / self.cap0, 1.0)

        # j occupies (r_j / r_i) of i's slots iff BOTH fight in contact.  Ranged
        # rows and columns are zero, so a ranged attacker reads u = 0 exactly.
        # Diagonal zero BY CONSTRUCTION -- the j != i of P-3.1 lives here.
        both = self.melee[:, None] & self.melee[None, :]
        self.wpeer = np.where(both, self.r_ally[None, :] / self.r_ally[:, None], 0.0)
        np.fill_diagonal(self.wpeer, 0.0)

        # ---- the r classes: enemy unit TYPES, in a stable declared order -----
        self.classes = sorted(set(self.enemy_names))
        self.cls_of = np.array([self.classes.index(x) for x in self.enemy_names],
                               dtype=np.int64)
        self.r_full = len(self.classes)
        share = np.array([float((self.cls_of == c).sum()) / max(1, self.m)
                          for c in range(self.r_full)])
        self.min_share = float(min_share)
        self.keep = np.where(share >= self.min_share)[0]
        self.pruned = [self.classes[c] for c in range(self.r_full)
                       if c not in set(self.keep.tolist())]
        self.r = int(len(self.keep))
        assert self.r >= 1, "every element class was pruned -- check the roster"
        self._col = np.full(self.r_full, -1, dtype=np.int64)
        self._col[self.keep] = np.arange(self.r)

        # ---- P-3.3: centre on the GEOMETRIC reference, scale at capacity ------
        # centre: the load agent i would meet on a class-k target if every peer
        # picked a target uniformly at random -- roster and fleet size only
        self.x_ref = np.zeros((self.n, self.r))
        for kk, c in enumerate(self.keep):
            cap_c = self.cap0[:, self.cls_of == c].mean(axis=1)          # (n,)
            u_ref = self.wpeer.sum(axis=1) / max(1, self.m) / cap_c
            self.x_ref[:, kk] = self._share(u_ref)
        # scale: the share at exactly nominal capacity, s(1) = alpha/(1+alpha).
        # A declared constant of the congestion function, so the regressor is O(1)
        # wherever the medium is anywhere near binding.
        self.x_scale = float(self._share(1.0))

    # ------------------------------------------------------------------ physics
    def _share(self, u):
        a = self.alpha * np.power(np.maximum(np.asarray(u, dtype=np.float64), 0.0),
                                  self.beta)
        return a / (1.0 + a)

    def perf(self, u):
        """The congestion function f(u) = 1 + alpha u**beta."""
        return 1.0 + self.alpha * np.power(np.maximum(np.asarray(u, dtype=np.float64),
                                                      0.0), self.beta)

    def g_eff(self, g_elem, targets):
        """The multiplier each agent meets on its target: the element's g, never
        below the one-open-slot floor for that attacker and target; 1 if none."""
        t = np.asarray(targets, dtype=np.int64).reshape(self.n)
        g = np.asarray(g_elem, dtype=np.float64).reshape(self.m)
        out = np.ones(self.n)
        on = t >= 0
        idx = np.where(on)[0]
        out[on] = np.maximum(g[t[on]], self.g_floor[idx, t[on]])
        return out

    def g_eff_all(self, g_elem):
        """``g_eff`` for every agent against every option, (n, m)."""
        g = np.asarray(g_elem, dtype=np.float64).reshape(self.m)
        return np.maximum(g[None, :], self.g_floor)

    def excess(self, u_nom, g):
        """``f(u/g)/f(u) - 1``, computed in its exact share form.

        g == 1  ==>  exactly 0      (the stock task, NS-2.1)
        u == 0  ==>  exactly 0      (a lone attacker, at any g -- I.2)
        """
        gg = np.maximum(np.asarray(g, dtype=np.float64), 1e-3)
        return (np.power(gg, -self.beta) - 1.0) * self._share(u_nom)

    def p_block(self, excess):
        """Probability an attack order finds no slot: the fraction of attempts
        that fail when a landed attack costs (1 + excess) attempts on average."""
        e = np.asarray(excess, dtype=np.float64)
        return e / (1.0 + e)

    # ------------------------------------------------------------------ loading
    def _weights(self, targets, alive, fired):
        t = np.asarray(targets, dtype=np.int64).reshape(self.n)
        w = (np.asarray(alive, dtype=np.float64).reshape(self.n)
             * np.asarray(fired, dtype=np.float64).reshape(self.n)
             * (t >= 0))
        T = np.zeros((self.n, self.m))
        on = t >= 0
        T[np.where(on)[0], t[on]] = 1.0
        return t, w, T

    def nearer(self, dist):
        """``near[j, i, k]``: attacker j is closer to enemy k than attacker i is
        (ties to the lower index), (n, n, m).  None means no geometry: everyone
        counts, which is the symmetric fallback."""
        d = np.asarray(dist, dtype=np.float64).reshape(self.n, self.m)
        dj, di = d[:, None, :], d[None, :, :]
        idx = np.arange(self.n)
        tie = (idx[:, None] < idx[None, :])[:, :, None]
        return (dj < di) | ((dj == di) & tie)

    def loading_all_options(self, targets, alive, fired, dist=None):
        """``U[i, k]``: the loading agent i WOULD meet on enemy k, peers fixed.
        Agent i's own footprint never counts (the diagonal of wpeer is zero);
        with ``dist`` only peers nearer to k than i count."""
        t, w, T = self._weights(targets, alive, fired)
        if dist is None:
            occ = (self.wpeer * w[None, :]) @ T                        # (n, m)
        else:
            occ = np.einsum("ij,j,jk,jik->ik", self.wpeer, w, T, self.nearer(dist))
        return np.minimum(occ / self.cap0, self.U_CLIP)

    def loading(self, targets, alive, fired, dist=None):
        """``u_i`` on the element agent i is actually attacking (0 if none).
        NS-1.1's max over the agent's elements is over one element here."""
        t = np.asarray(targets, dtype=np.int64).reshape(self.n)
        U = self.loading_all_options(targets, alive, fired, dist)
        u = np.zeros(self.n)
        on = t >= 0
        u[on] = U[np.where(on)[0], t[on]]
        return u

    def loading_bruteforce(self, targets, alive, fired, dist=None):
        """Straight off the definition, for P-3.2."""
        t = np.asarray(targets, dtype=np.int64).reshape(self.n)
        d = None if dist is None else np.asarray(dist, dtype=np.float64)
        a = np.asarray(alive, dtype=np.float64).reshape(self.n)
        f = np.asarray(fired, dtype=np.float64).reshape(self.n)
        u = np.zeros(self.n)
        for i in range(self.n):
            if t[i] < 0:
                continue
            s = 0.0
            if not self.melee[i]:
                continue                                     # no ring, no loading
            for j in range(self.n):
                if j == i or t[j] != t[i] or not self.melee[j]:
                    continue                                 # STRICTLY j != i
                if d is not None and not (d[j, t[i]] < d[i, t[i]] or
                                          (d[j, t[i]] == d[i, t[i]] and j < i)):
                    continue                                 # farther: no slot taken
                s += a[j] * f[j] * self.r_ally[j] / self.r_ally[i]
            u[i] = min(s / self.cap0[i, t[i]], self.U_CLIP)
        return u

    # ------------------------------------------------------------------ basis
    def psi_all_options(self, targets, alive, fired, dist=None):
        """``psi[i, k] = [1, x_1..x_r]`` if agent i attacked enemy k, (n, m, r+1)."""
        s = self._share(self.loading_all_options(targets, alive, fired, dist))
        raw = np.zeros((self.n, self.m, self.r))
        for kk, c in enumerate(self.keep):
            sel = self.cls_of == c
            raw[:, sel, kk] = s[:, sel]
        x = (raw - self.x_ref[:, None, :]) / self.x_scale
        return np.concatenate([np.ones((self.n, self.m, 1)), x], axis=2)

    def psi(self, targets, alive, fired, dist=None):
        """The regressor row for each agent's ACTUAL target, (n, r+1); an agent
        with no target reads its class references (x = -ref/scale)."""
        t = np.asarray(targets, dtype=np.int64).reshape(self.n)
        P = self.psi_all_options(targets, alive, fired, dist)
        out = np.concatenate([np.ones((self.n, 1)), -self.x_ref / self.x_scale], axis=1)
        on = t >= 0
        out[on] = P[np.where(on)[0], t[on]]
        return out

    def psi_bruteforce(self, targets, alive, fired, dist=None):
        t = np.asarray(targets, dtype=np.int64).reshape(self.n)
        u = self.loading_bruteforce(targets, alive, fired, dist)
        out = np.zeros((self.n, self.r + 1))
        out[:, 0] = 1.0
        for i in range(self.n):
            x = np.zeros(self.r)
            if t[i] >= 0 and self._col[self.cls_of[t[i]]] >= 0:
                a = self.alpha * max(u[i], 0.0) ** self.beta
                x[self._col[self.cls_of[t[i]]]] = a / (1.0 + a)
            out[i, 1:] = (x - self.x_ref[i]) / self.x_scale
        return out

    # ------------------------------------------------------------------ operator
    def W(self):
        """NS-1.2: ``W[i, j] = mean over e of Op[e, j]`` with Op = footprint /
        capacity -- zero diagonal, nonzero only between melee units, asymmetric
        wherever the two units' radii differ."""
        return self.wpeer * np.mean(1.0 / self.cap0, axis=1)[:, None]

    def report(self):
        W = self.W()
        off = W[~np.eye(self.n, dtype=bool)]
        off = off[off > 0]
        d = np.abs(W) + np.abs(W.T)
        with np.errstate(invalid="ignore", divide="ignore"):
            asym = float(np.max(np.where(d > 0, np.abs(W - W.T) / d, 0.0)))
        melee_caps = self.cap0[self.melee] if self.melee.any() else np.array([np.nan])
        return dict(r=self.r, classes=list(self.classes), pruned=list(self.pruned),
                    n_melee=int(self.melee.sum()),
                    zero_diag=bool(np.all(np.diag(W) == 0.0)),
                    spread=float(off.std() / off.mean()) if off.size else float("nan"),
                    asymmetry=asym,
                    melee_capacity=[float(np.round(melee_caps.min(), 3)),
                                    float(np.round(melee_caps.max(), 3))],
                    alpha=self.alpha, beta=self.beta)

    def banner(self):
        r = self.report()
        return ("[SMAC-SA] surface area %s  melee=%d/%d  melee capacity %s  classes=%s"
                "  W: zero_diag=%s spread=%.3f asym=%.3f  BPR alpha=%.2f beta=%.0f"
                % (self.map_name, r["n_melee"], self.n, r["melee_capacity"],
                   r["classes"], r["zero_diag"], r["spread"], r["asymmetry"],
                   self.alpha, self.beta))
