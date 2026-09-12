"""The declared coupling operator and the basis.  Structure only, never fitted.

`PACT_NS_SPEC` NS-1.2 and II.3.  Everything in this module is computable from
StarCraft II's published unit table and the map's roster, before a single episode
runs.

THE MEDIUM, AND WHY IT IS THE NATURAL ONE FOR SMAC
--------------------------------------------------------------------------------
The shared resource is the ENEMY LINE'S ABSORPTION: how much allied damage can
usefully land on a given enemy before the rest is overkill.  Overkill is the
canonical StarCraft coordination failure -- every player has said "stop
overkilling that zealot" -- and it is what a 3s5z policy at a 90% win rate has
actually learned to avoid.

    element  e   an enemy unit                       (like a road link)
    cap_e        its published effective hit points   (like lanes x 600 veh/h)
    load_e       allied damage directed at e,  sum over j != i
    u_i          load_{target(i)} / (cap * g)         the loading ratio

A LONE AGENT CANNOT OVERKILL.  It fires, the target dies, it retargets; nothing is
wasted at any guard intensity.  The peer sum is empty, so ``u_i = 0`` and the harm
is exactly 1.0 -- NS-1.2's category-C signature obtained *structurally*, from the
``j != i``, not from a number happening to be small.

THE REGIME DECIDES THE PERFORMANCE FUNCTION (III.1, "regime, not convention")
--------------------------------------------------------------------------------
On 3s5z the whole squad focused on one target delivers ~39 damage in a step
against a Zealot's 150 effective hit points, so the medium runs at u ~ 0.26 --
two orders of magnitude below the regime where a quartic BPR exponent means
anything, and almost exactly URB's measured v/c = 0.219.  So the performance
function is LINEAR, as the spec instructs for this regime:

    cost_i / cost_i^nominal = 1 + alpha * u_i        alpha declared, swept

and the sensor is then ``realized/nominal - 1``, which is what
``pact1_core.relative_excess`` computes, with no rescaling.

THE r CLASSES (P-1.1, P-1.2)
--------------------------------------------------------------------------------
``r`` = the number of enemy UNIT TYPES on the map (2 on 3s5z: Stalker, Zealot).
Independent of the number of agents and of the number of elements, as P-1.1
requires.  The class of a target and its published hit points are visible to every
player -- the counterpart of a road's painted lane count.  What is NOT handed over
is ``beta*``: how much a peer's shot at a target of that class actually costs you
today, which drifts as the guard cycle moves.
"""

import numpy as np

# --------------------------------------------------------------------------- #
# StarCraft II's published unit table.  Effective hit points = life + shields;
# damage is per weapon cycle, cooldown in game seconds.  These are the game's own
# numbers, not ours, and nothing here is fitted.
# --------------------------------------------------------------------------- #
UNIT_STATS = {
    #            life  shield  armour  damage  cooldown  range
    "Stalker":   dict(life=80,  shield=80,  armour=1, damage=10.0, cd=1.87, rng=6.0),
    "Zealot":    dict(life=100, shield=50,  armour=1, damage=16.0, cd=0.86, rng=0.1),
    "Colossus":  dict(life=200, shield=150, armour=1, damage=24.0, cd=1.65, rng=7.0),
    "Marine":    dict(life=45,  shield=0,   armour=0, damage=6.0,  cd=0.61, rng=5.0),
    "Marauder":  dict(life=125, shield=0,   armour=1, damage=10.0, cd=1.07, rng=6.0),
    "Medivac":   dict(life=150, shield=0,   armour=1, damage=0.0,  cd=1.00, rng=4.0),
    "Hydralisk": dict(life=90,  shield=0,   armour=0, damage=12.0, cd=0.59, rng=5.0),
    "Zergling":  dict(life=35,  shield=0,   armour=0, damage=5.0,  cd=0.497, rng=0.1),
    "Baneling":  dict(life=30,  shield=0,   armour=0, damage=20.0, cd=1.00, rng=0.25),
}

SC2_FPS = 22.4          # SMAC advances step_mul frames; a step is step_mul/22.4 s

#: Ally and enemy rosters per map, in the index order SMAC uses.
ALLY = {
    "3s5z": ["Stalker"] * 3 + ["Zealot"] * 5,
    "1c3s5z": ["Colossus"] + ["Stalker"] * 3 + ["Zealot"] * 5,
    "2s3z": ["Stalker"] * 2 + ["Zealot"] * 3,
    "3s5z_vs_3s6z": ["Stalker"] * 3 + ["Zealot"] * 5,
    "MMM2": ["Marauder"] + ["Marine"] * 7 + ["Medivac"],
    "8m": ["Marine"] * 8,
    "10m_vs_11m": ["Marine"] * 10,
    "27m_vs_30m": ["Marine"] * 27,
    "corridor": ["Zealot"] * 6,
    "bane_vs_bane": ["Baneling"] * 4 + ["Zergling"] * 20,
}
ENEMY = {
    "3s5z": ["Stalker"] * 3 + ["Zealot"] * 5,
    "1c3s5z": ["Colossus"] + ["Stalker"] * 3 + ["Zealot"] * 5,
    "2s3z": ["Stalker"] * 2 + ["Zealot"] * 3,
    "3s5z_vs_3s6z": ["Stalker"] * 3 + ["Zealot"] * 6,
    "MMM2": ["Marauder"] * 2 + ["Marine"] * 8 + ["Medivac"] * 2,
    "8m": ["Marine"] * 8,
    "10m_vs_11m": ["Marine"] * 11,
    "27m_vs_30m": ["Marine"] * 30,
    "corridor": ["Zergling"] * 24,
    "bane_vs_bane": ["Baneling"] * 4 + ["Zergling"] * 20,
}


def roster(map_name, n, which="ally"):
    tbl = ALLY if which == "ally" else ENEMY
    names = list(tbl.get(map_name, []))
    if len(names) == int(n):
        return names
    return ["Marine"] * int(n)


def effective_hp(names):
    """cap_e -- published life + shields.  THE DENOMINATOR."""
    return np.array([UNIT_STATS[x]["life"] + UNIT_STATS[x]["shield"] for x in names],
                    dtype=np.float64)


def step_damage(names, step_mul=8):
    """Damage one unit of each type delivers in one environment step.  Published
    weapon damage divided by published cooldown, times the step duration."""
    dt = float(step_mul) / SC2_FPS
    return np.array([UNIT_STATS[x]["damage"] / UNIT_STATS[x]["cd"] * dt
                     for x in names], dtype=np.float64)


class Coupling(object):
    """The declared operator, the r classes, and the basis.

    Args:
        map_name:  SMAC map, used only to look up the published rosters.
        n_agents / n_enemies: from the map params.
        step_mul:  SMAC's own, for the per-step damage conversion.
        alpha:     the linear performance coefficient (III.1).  DECLARED and
                   swept in the ablation; calibrated once against an observable
                   by the stated one-line procedure, never turned toward a curve.
    """

    def __init__(self, map_name, n_agents, n_enemies, step_mul=8, alpha=2.28):
        self.map_name = str(map_name)
        self.n = int(n_agents)
        self.m = int(n_enemies)
        self.step_mul = int(step_mul)
        self.alpha = float(alpha)
        self.ally_names = roster(self.map_name, self.n, "ally")
        self.enemy_names = roster(self.map_name, self.m, "enemy")
        self.cap = effective_hp(self.enemy_names)                    # (m,)
        self.dmg = step_damage(self.ally_names, self.step_mul)       # (n,)

        # ---- the r classes: enemy unit TYPES, in a stable declared order -----
        self.classes = sorted(set(self.enemy_names))
        self.cls_of = np.array([self.classes.index(x) for x in self.enemy_names],
                               dtype=np.int64)                       # (m,)
        self.r_full = len(self.classes)
        # P-3.4: prune classes below a declared share of the medium, and KEEP THE
        # INDEX ALIGNMENT when you do.  `keep` maps kept-channel -> class id.
        share = np.array([float((self.cls_of == c).sum()) / max(1, self.m)
                          for c in range(self.r_full)])
        self.min_share = 0.05
        self.keep = np.where(share >= self.min_share)[0]
        self.pruned = [self.classes[c] for c in range(self.r_full)
                       if c not in set(self.keep.tolist())]
        self.r = int(len(self.keep))
        assert self.r >= 1, "every element class was pruned -- check the roster"

        # ---- P-3.3: the GEOMETRIC centring reference -------------------------
        # The load an agent would see if every peer picked a target uniformly at
        # random.  A function of the roster and the fleet size ONLY -- no run data
        # enters, which is what keeps a declared model class from becoming a fit.
        # A peer lands on my specific target with probability 1/m, and contributes
        # its own step damage over that target's capacity.
        self.dmg_ref = float(self.dmg.mean())
        self.x_ref = np.zeros(self.r)
        for k, c in enumerate(self.keep):
            cap_c = float(self.cap[self.cls_of == c].mean())
            self.x_ref[k] = (self.n - 1) * self.dmg_ref / (max(1, self.m) * cap_c)
        # scale so the centred regressor is O(1): divide by the reference itself.
        self.x_scale = np.maximum(self.x_ref, 1e-12)

    # ------------------------------------------------------------------ basis
    def channels(self, targets, alive, fired, exclude=None):
        """``x[i, k]`` -- the peer load on class k landing on agent i's own target.

        ``targets`` (n,)  index of the enemy each agent is shooting, or -1
        ``alive``   (n,)  0/1 ally liveness
        ``fired``   (n,)  0/1 did this agent actually discharge its weapon
        ``exclude`` optional (n,) override of the target used for agent i, so the
                    same routine serves the COUNTERFACTUAL query "what would I
                    meet if I shot k instead" that the steering channel needs.

        Every sum is strictly over ``j != i`` (P-3.1), so a lone agent reads
        exactly zero on every channel.  Centred on the geometric reference and
        scaled by it (P-3.3): raw channels carry a large common mean against the
        intercept column, and leaving it in drove URB's design-matrix condition
        number to 1.3e5 with beta unidentifiable while prediction looked fine.
        """
        t = np.asarray(targets, dtype=np.int64).reshape(self.n)
        w = (np.asarray(alive, dtype=np.float64)
             * np.asarray(fired, dtype=np.float64) * self.dmg)        # (n,) load
        mine = t if exclude is None else np.asarray(exclude, dtype=np.int64)

        # total damage aimed at each element, then subtract the agent's own so the
        # sum is over j != i without an O(n^2) loop.
        tot = np.zeros(self.m + 1)
        np.add.at(tot, np.where(t >= 0, t, self.m), w)
        out = np.zeros((self.n, self.r))
        for i in range(self.n):
            e = int(mine[i])
            if e < 0:
                continue
            peer = tot[e] - (w[i] if t[i] == e else 0.0)              # j != i
            c = int(self.cls_of[e])
            if c in set(self.keep.tolist()):
                k = int(np.where(self.keep == c)[0][0])
                out[i, k] = peer / max(1e-12, self.cap[e])
        return (out - self.x_ref[None, :]) / self.x_scale[None, :]

    def channels_bruteforce(self, targets, alive, fired, exclude=None):
        """The same thing written straight off the definition, for P-3.2.

        Index order and self-exclusion are exactly the kind of wiring bug that
        leaves every diagnostic looking healthy, so the vectorised routine is
        checked against this at startup and the run aborts on mismatch.
        """
        t = np.asarray(targets, dtype=np.int64).reshape(self.n)
        a = np.asarray(alive, dtype=np.float64)
        f = np.asarray(fired, dtype=np.float64)
        mine = t if exclude is None else np.asarray(exclude, dtype=np.int64)
        out = np.zeros((self.n, self.r))
        for i in range(self.n):
            e = int(mine[i])
            if e < 0:
                continue
            c = int(self.cls_of[e])
            if c not in set(self.keep.tolist()):
                continue
            k = int(np.where(self.keep == c)[0][0])
            s = 0.0
            for j in range(self.n):
                if j == i:
                    continue                                   # STRICTLY j != i
                if int(t[j]) == e:
                    s += a[j] * f[j] * self.dmg[j]
            out[i, k] = s / max(1e-12, self.cap[e])
        return (out - self.x_ref[None, :]) / self.x_scale[None, :]

    def psi(self, targets, alive, fired, exclude=None):
        """``psi_i = [1, x_1, ..., x_r]`` -- the regressor rows, (n, r+1)."""
        x = self.channels(targets, alive, fired, exclude)
        return np.concatenate([np.ones((self.n, 1)), x], axis=1)

    def psi_options(self, i, targets, alive, fired, options):
        """``psi`` for ONE agent over its K candidate targets, (K, r+1).

        This is what the steering channel ranks.  Agent i's own contribution is
        excluded from every row by construction, so switching target never makes
        the agent load itself.
        """
        rows = []
        for k in options:
            ex = np.array(targets, dtype=np.int64).copy()
            ex[i] = int(k)
            rows.append(self.psi(targets, alive, fired, exclude=ex)[i])
        return np.asarray(rows, dtype=np.float64)

    # ------------------------------------------------------------------ loading
    def loading(self, targets, alive, fired, g):
        """``u_i`` -- the loading ratio on the element agent i is using.

        NS-1.1 says the aggregation is a MAXIMUM over the agent's own elements,
        never a mean.  An attacking agent uses exactly one element, so the max is
        over a single term and the requirement is met trivially; the code keeps
        the max form so a multi-element variant stays honest.

        ``g`` is the driver's capacity multiplier, per element.  It divides, which
        is what makes the driver an AMPLIFIER on the peer term rather than a level
        added on top of it (NS-1.3 / I.2).
        """
        t = np.asarray(targets, dtype=np.int64).reshape(self.n)
        w = (np.asarray(alive, dtype=np.float64)
             * np.asarray(fired, dtype=np.float64) * self.dmg)
        gg = np.broadcast_to(np.asarray(g, dtype=np.float64).reshape(-1), (self.m,)) \
            if np.size(g) > 1 else np.full(self.m, float(g))
        tot = np.zeros(self.m + 1)
        np.add.at(tot, np.where(t >= 0, t, self.m), w)
        u = np.zeros(self.n)
        for i in range(self.n):
            e = int(t[i])
            if e < 0:
                continue
            peer = tot[e] - w[i]                                    # j != i
            u[i] = np.max([peer / max(1e-12, self.cap[e] * gg[e])])
        return u

    def excess(self, u_nom, g):
        """The HARM, as I.2 defines it -- a RATIO of the performance function, not
        the performance function itself:

            harm_i   = f(u_i under derated capacity) / f(u_i at nominal capacity)
            excess_i = harm_i - 1 = [f(u/g) - f(u)] / f(u),   f(u) = 1 + alpha*u

        Getting this wrong is the difference between category C and a broken dial,
        so it is worth spelling out.  Overkill already exists in stock StarCraft --
        ``f(u) > 1`` at sigma = 0 -- and that base cost is NOT the disturbance; it
        is the game.  The disturbance is only the AMPLIFICATION of it by a shrunken
        absorption capacity.  Taking ``alpha*u`` as the harm would have charged the
        team for stock overkill at every severity including zero.

        Two identities fall out exactly, not approximately:

            g == 1  ==>  excess == 0     NS-2.1, so sigma=0 is the stock task
            u == 0  ==>  excess == 0     I.2, so a lone agent is untouched at ANY g

        and for small u this is the spec's linear reading, ``~ alpha*u*(1/g - 1)``:
        peer load times a weather factor, nothing an agent can absorb alone.

        ``u_nom`` is the loading at NOMINAL capacity (call ``loading`` with g = 1).
        """
        u = np.asarray(u_nom, dtype=np.float64)
        gg = np.asarray(g, dtype=np.float64)
        f_nom = 1.0 + self.alpha * u
        f_der = 1.0 + self.alpha * u / np.maximum(gg, 1e-9)
        return (f_der - f_nom) / f_nom

    # ------------------------------------------------------------------ operator
    def W(self):
        """``W[i, j] = mean over e in E(i) of 1[j uses e] / cap_e``, zero diagonal.

        Declared from the roster alone: ``E(i)`` is the set of enemies agent i can
        engage, which its published weapon range decides, and the weight is one
        over the element's published capacity.  Never fitted.
        """
        rng_i = np.array([UNIT_STATS[x]["rng"] for x in self.ally_names])
        # An element is in E(i) when agent i's weapon can reach that band.  Melee
        # units engage the near band only; ranged units engage everything.  This
        # is published weapon range, not geometry measured from a run.
        band = np.array([UNIT_STATS[x]["rng"] for x in self.enemy_names])
        W = np.zeros((self.n, self.n))
        for i in range(self.n):
            Ei = np.where(band <= max(rng_i[i], 0.5) + 1e-9)[0]
            if Ei.size == 0:
                Ei = np.arange(self.m)
            inv_cap = 1.0 / self.cap[Ei]
            for j in range(self.n):
                if j == i:
                    continue                                    # ASSERTED zero
                W[i, j] = float(np.mean(inv_cap)) * self.dmg[j]
        return W

    def report(self):
        W = self.W()
        off = W[~np.eye(self.n, dtype=bool)]
        off = off[off > 0]
        asym = 0.0
        d = np.abs(W) + np.abs(W.T)
        with np.errstate(invalid="ignore", divide="ignore"):
            asym = float(np.max(np.where(d > 0, np.abs(W - W.T) / d, 0.0)))
        return dict(
            r=self.r, r_full=self.r_full, classes=list(self.classes),
            pruned=list(self.pruned), zero_diag=bool(np.all(np.diag(W) == 0.0)),
            spread=float(off.std() / off.mean()) if off.size else float("nan"),
            ratio=float(off.max() / off.min()) if off.size else float("nan"),
            asymmetry=asym, cap=[float(x) for x in self.cap[:4]],
            alpha=self.alpha, x_ref=[float(x) for x in self.x_ref],
        )

    def banner(self):
        r = self.report()
        return ("[SMAC-NS] coupling %s  r=%d/%d classes=%s pruned=%s  "
                "W: zero_diag=%s spread=%.3f ratio=%.1fx asym=%.3f  alpha=%.2f"
                % (self.map_name, r["r"], r["r_full"], r["classes"], r["pruned"],
                   r["zero_diag"], r["spread"], r["ratio"], r["asymmetry"],
                   r["alpha"]))
