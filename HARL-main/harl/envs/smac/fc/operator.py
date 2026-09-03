"""The DECLARED coupling operator for Formation Congestion on SMAC.

`NS_FORM_SPEC` A.2 object 2 demands an operator that a domain expert would write
down from the environment's own model -- PTDF in a power grid, a mechanical
Jacobian in a robot -- and it is emphatic that a geometric PROXY is not good
enough (POWER measured ``fit_gain = -0.0045`` with one: the peer channels made
prediction WORSE than an intercept-only null).

StarCraft II's own model gives us one.  Ground units are rigid collision bodies
with published radii and speeds, and a unit executing a move order sweeps a
corridor through the formation.  The configuration-space obstacle a body of
radius ``r_j`` presents to a body of radius ``r_i`` is a disc of radius
``r_i + r_j`` (the Minkowski sum) -- textbook motion planning, not an analogy --
and the corridor unit *i* sweeps in one environment step is ``2 r_i d_i`` with
``d_i = v_i * dt``.  So

    W_raw[i, j] = pi (r_i + r_j)^2 / (2 r_i d_i)          j's obstacle / i's corridor
    W[i, j]     = W_raw[i, j] * band(i, j)                x how much they share ground
    W[i, i]     = 0                                       ASSERTED, not argued

``band`` is the second half of the physics: a melee unit fights at weapon range
~0.1 and therefore lives in the contact line, a ranged unit stands off at 6.  Two
units whose weapon ranges differ do not occupy the same ground, so one obstructs
the other far less:

    band(i, j) = exp(-|R_i - R_j| / BAND_SCALE)

On 3s5z (3 Stalkers + 5 Zealots) with the SC2 unit data below this gives the
properties `NS_FORM_SPEC` A.2 says a real operator has and a proxy does not --
see ``report()``:

    spread (max/min off-diagonal)   ~25x
    std/mean over off-diagonals     ~0.96      (POWER measured 1.35)
    ASYMMETRIC                      W[zealot, stalker] != W[stalker, zealot]

Nothing here is fitted.  Every number is either an SC2 unit statistic or one of
the declared constants at the top of the file, and the whole matrix is built once
at construction from the map name alone.
"""

import math

import numpy as np

# --------------------------------------------------------------------------- #
# StarCraft II unit data.  These are the game's own published values: collision
# radius (world units), movement speed (world units / game second) and weapon
# range (world units).  They are the environment's model, not ours.
# --------------------------------------------------------------------------- #
UNIT_STATS = {
    #                 radius  speed  weapon_range
    "Marine":    dict(radius=0.375,  speed=3.15, weapon_range=5.0),
    "Marauder":  dict(radius=0.5625, speed=3.15, weapon_range=6.0),
    "Medivac":   dict(radius=0.75,   speed=3.50, weapon_range=4.0),
    "Stalker":   dict(radius=0.625,  speed=4.13, weapon_range=6.0),
    "Zealot":    dict(radius=0.50,   speed=3.15, weapon_range=0.1),
    "Colossus":  dict(radius=1.00,   speed=3.15, weapon_range=7.0),
    "Hydralisk": dict(radius=0.625,  speed=3.15, weapon_range=5.0),
    "Zergling":  dict(radius=0.375,  speed=4.13, weapon_range=0.1),
    "Baneling":  dict(radius=0.375,  speed=4.13, weapon_range=0.25),
}

# SC2 runs at 22.4 game frames per second; SMAC advances `step_mul` frames per
# environment step, so one step lasts `step_mul / SC2_FPS` game seconds.
SC2_FPS = 22.4

# The one shape constant of the band model: how fast obstruction falls off with a
# difference in weapon range (world units).  2.0 puts a melee/ranged pair at
# exp(-5.9/2) ~ 0.05 of a same-band pair, which is the point -- zealots and
# stalkers genuinely do not stand on the same ground.
BAND_SCALE = 2.0

# Obstruction decays with the clear gap between two bodies (world units).  A peer
# that is touching you blocks fully; one 2 units clear of you barely does.
PROX_LEN = 2.0

# Enemy bodies obstruct too.  They are not controllable by any agent, so they are
# `L_i^fixed` in NS_FORM_SPEC A.2 -- the irreducible class of the Part-C ceiling.
PHI_ENEMY = 1.0

# The four SMAC move directions as unit (x, y) vectors, in the order the env's
# action ids 2..5 use: north, south, east, west.
MOVE_DIRS = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]])
MOVE_NAMES = ("N", "S", "E", "W")

# Which unit types each SMAC `map_type` fields, in the order `_init_ally_unit_types`
# assigns ids.  Lets a unit be named without a live StarCraft II process, so the
# operator can be built and certified offline.
MAP_TYPE_UNITS = {
    "marines": ["Marine"],
    "stalkers_and_zealots": ["Stalker", "Zealot"],
    "colossi_stalkers_zealots": ["Colossus", "Stalker", "Zealot"],
    "MMM": ["Marauder", "Marine", "Medivac"],
    "zealots": ["Zealot"],
    "hydralisks": ["Hydralisk"],
    "stalkers": ["Stalker"],
    "colossus": ["Colossus"],
    "bane": ["Baneling", "Zergling"],
}

# Squad composition per map, in agent-index order.  SMAC sorts allied units by
# unit_type then position, so the composition is fixed and known from the map name.
MAP_COMPOSITION = {
    "3s5z": ["Stalker"] * 3 + ["Zealot"] * 5,
    "2s3z": ["Stalker"] * 2 + ["Zealot"] * 3,
    "1c3s5z": ["Colossus"] * 1 + ["Stalker"] * 3 + ["Zealot"] * 5,
    "3s5z_vs_3s6z": ["Stalker"] * 3 + ["Zealot"] * 5,
    "3s_vs_3z": ["Stalker"] * 3,
    "3s_vs_4z": ["Stalker"] * 3,
    "3s_vs_5z": ["Stalker"] * 3,
    "2s_vs_1sc": ["Stalker"] * 2,
    "8m": ["Marine"] * 8,
    "3m": ["Marine"] * 3,
    "25m": ["Marine"] * 25,
    "27m_vs_30m": ["Marine"] * 27,
    "MMM": ["Marauder"] + ["Marine"] * 7 + ["Medivac"],
    "MMM2": ["Marauder"] + ["Marine"] * 7 + ["Medivac"],
    "2c_vs_64zg": ["Colossus"] * 2,
    "5m_vs_6m": ["Marine"] * 5,
    "10m_vs_11m": ["Marine"] * 10,
}

# Enemy composition per map (the `L^fixed` source).
MAP_ENEMY_COMPOSITION = {
    "3s5z": ["Stalker"] * 3 + ["Zealot"] * 5,
    "2s3z": ["Stalker"] * 2 + ["Zealot"] * 3,
    "1c3s5z": ["Colossus"] * 1 + ["Stalker"] * 3 + ["Zealot"] * 5,
    "3s5z_vs_3s6z": ["Stalker"] * 3 + ["Zealot"] * 6,
    "3s_vs_3z": ["Zealot"] * 3,
    "3s_vs_4z": ["Zealot"] * 4,
    "3s_vs_5z": ["Zealot"] * 5,
    "2s_vs_1sc": ["Colossus"],
    "8m": ["Marine"] * 8,
    "3m": ["Marine"] * 3,
    "25m": ["Marine"] * 25,
    "27m_vs_30m": ["Marine"] * 30,
    "MMM": ["Marauder"] + ["Marine"] * 7 + ["Medivac"],
    "MMM2": ["Marauder"] + ["Marine"] * 7 + ["Medivac"],
    "2c_vs_64zg": ["Zergling"] * 64,
    "5m_vs_6m": ["Marine"] * 6,
    "10m_vs_11m": ["Marine"] * 11,
}


def composition(map_name, n_agents, map_type="stalkers_and_zealots"):
    """The ally squad's unit names, in agent-index order.

    Falls back to repeating the map_type's first unit when the map is not in the
    table -- the operator is then type-homogeneous, which is correct for e.g. 8m
    and is reported as such by `report()`.
    """
    if map_name in MAP_COMPOSITION:
        names = list(MAP_COMPOSITION[map_name])
        if len(names) == int(n_agents):
            return names
    base = MAP_TYPE_UNITS.get(map_type, ["Marine"])[0]
    return [base] * int(n_agents)


def enemy_composition(map_name, n_enemies, map_type="stalkers_and_zealots"):
    """The enemy line's unit names, in enemy-index order (the `L^fixed` source)."""
    if map_name in MAP_ENEMY_COMPOSITION:
        names = list(MAP_ENEMY_COMPOSITION[map_name])
        if len(names) == int(n_enemies):
            return names
    base = MAP_TYPE_UNITS.get(map_type, ["Marine"])[0]
    return [base] * int(n_enemies)


def _stats(names):
    r = np.array([UNIT_STATS[n]["radius"] for n in names], dtype=np.float64)
    v = np.array([UNIT_STATS[n]["speed"] for n in names], dtype=np.float64)
    rng = np.array([UNIT_STATS[n]["weapon_range"] for n in names], dtype=np.float64)
    return r, v, rng


def radii(names):
    """Collision radii, in agent-index order."""
    return _stats(list(names))[0]


def reach(names, step_mul=8):
    """How far each unit can actually travel in one environment step (world units).

    This is what decides whether a throttled move order BINDS: SMAC sends the
    order to a point ``move_amount`` away, and the unit simply walks toward it for
    the step.  If ``move_amount`` exceeds this reach the unit never arrives and
    scaling the order down has no effect until the scaled distance drops below it
    -- a dead zone.  ``smac.yaml`` therefore ships ``move_amount`` at or below the
    smallest reach on the map, so every derating bites from the first unit of
    severity.  See fc/README.md.
    """
    _, v, _ = _stats(list(names))
    return v * (float(step_mul) / SC2_FPS)


def build_W(ally_names, peer_names, step_mul=8, band_scale=BAND_SCALE,
            zero_diagonal=True):
    """The declared obstruction operator ``W[i, j]``.

    ``i`` indexes the ALLY agents (rows -- the units that get obstructed); ``j``
    indexes the obstructing bodies: ally peers for the coupling operator, enemy
    units for ``L^fixed``.  ``zero_diagonal`` is True only when the two lists are
    the same squad, where NS_FORM_SPEC A.2 asserts ``W[i, i] == 0``.
    """
    r_i, v_i, R_i = _stats(list(ally_names))
    r_j, _, R_j = _stats(list(peer_names))
    dt = float(step_mul) / SC2_FPS
    d_i = v_i * dt                                   # ground i covers in one step
    swept = 2.0 * r_i * d_i                          # i's corridor area
    obstacle = math.pi * (r_i[:, None] + r_j[None, :]) ** 2      # Minkowski disc
    band = np.exp(-np.abs(R_i[:, None] - R_j[None, :]) / float(band_scale))
    W = (obstacle / swept[:, None]) * band
    if zero_diagonal:
        np.fill_diagonal(W, 0.0)
    return W


def report(W, names, zero_diagonal=True):
    """The structural summary PACT_PIPELINE_SPEC 2.4 asks to be printed and kept.

    ``cond`` here is the OPERATOR's conditioning.  The REGRESSOR's conditioning --
    the one that decides whether beta can be DECOMPOSED rather than merely
    predicted (spec 12.6) -- is a different number, measured online and logged as
    ``cond_psi``.
    """
    W = np.asarray(W, dtype=np.float64)
    square = W.shape[0] == W.shape[1]
    if zero_diagonal and square:
        off = W[~np.eye(W.shape[0], dtype=bool)]
    else:
        off = W.reshape(-1)
    off = off[off > 0.0]
    if off.size == 0:
        return dict(ok=False, reason="operator is entirely zero")
    asym = 0.0
    if square:
        denom = np.abs(W) + np.abs(W.T)
        with np.errstate(invalid="ignore", divide="ignore"):
            a = np.where(denom > 0, np.abs(W - W.T) / denom, 0.0)
        asym = float(np.max(a))
    # 2.4: treat non-finite as a VALUE and test it FIRST -- `if isfinite(c) and
    # c > thr` lets the most degenerate basis possible pass silently.
    try:
        cond = float(np.linalg.cond(W))
    except np.linalg.LinAlgError:
        cond = float("inf")
    if not np.isfinite(cond):
        cond = float("inf")
    return dict(
        ok=True,
        shape=tuple(W.shape),
        names=list(names),
        mean=float(off.mean()),
        std=float(off.std()),
        spread=float(off.std() / off.mean()),
        ratio=float(off.max() / off.min()),
        wmin=float(off.min()),
        wmax=float(off.max()),
        asymmetry=asym,
        cond=cond,
        zero_diag=bool((not square) or np.allclose(np.diag(W), 0.0)),
    )


def banner(rep, title="W(ally,ally)"):
    if not rep.get("ok", False):
        return "[FC] %s: INVALID -- %s" % (title, rep.get("reason", "?"))
    return (
        "[FC] %s %s min=%.4f max=%.4f spread(std/mean)=%.3f ratio=%.1fx "
        "asym=%.3f cond=%.1f zero_diag=%s"
        % (title, rep["shape"], rep["wmin"], rep["wmax"], rep["spread"],
           rep["ratio"], rep["asymmetry"], rep["cond"], rep["zero_diag"])
    )


# --------------------------------------------------------------------------- #
# The exertion functional Phi  (NS_FORM_SPEC A.5)
# --------------------------------------------------------------------------- #
class Exertion:
    """``Phi_j`` -- what unit j does that takes up the squad's ground.

        phi_raw_j(t) = alive_j * (1 + phi_fire * fired_j)
        Phi_j(t)     = rho * Phi_j(t-1) + (1 - rho) * phi_raw_j(t)

    THE ESCAPE-HATCH RULE (A.5).  The ``alive_j`` factor is a FLOOR the team
    cannot get under: the only way to reduce it is to die, which loses reward
    directly.  The ``fired_j`` term is what makes it VARY, and it is uncancellable
    in the same sense -- a squad that stops shooting cannot kill the enemy team,
    so both directions cost reward.  Compare the rejected alternatives in A.5:
    "j is engaged" is escaped by disengaging, "j pulled the trigger" ALONE is
    escaped by staggering fire, and a signed sum is escaped by anti-symmetry.
    Here staggering fire moves Phi by at most phi_fire/(1+phi_fire) = 33% and
    costs damage; it cannot switch the coupling off.

    THE LEAK ``rho`` IS WHAT MAKES THE ONE-STEP PERSISTENCE IN PACT'S REGRESSOR
    SOUND (PACT_PIPELINE_SPEC 4.2: "peer columns: PREVIOUS ... sound because the
    coupling is slow by construction").  A body that just fired is still rooted in
    its animation; obstruction has inertia.  rho = 0.7 gives a ~3-step memory --
    slower than one step, far faster than the driver cycle.
    """

    def __init__(self, n, phi_fire=0.5, rho=0.7, phi_move=0.0):
        self.n = int(n)
        self.phi_fire = float(phi_fire)
        self.rho = float(rho)
        # phi_move > 0 makes Phi read the EXECUTED stride, which turns the
        # environment LOOP-COUPLED (NS_FORM_SPEC A.6): compensating then feeds the
        # medium it compensates against and T4's interior optimum applies.
        # DEFAULT 0 -- SMAC's compensator moves the unit, which changes neither
        # liveness nor firing, so the environment is loop-FREE and the spec's
        # E-table prediction ("SMAC loop? no") is the thing being confirmed.
        self.phi_move = float(phi_move)
        self.phi = np.zeros(self.n, dtype=np.float64)
        self.raw = np.zeros(self.n, dtype=np.float64)
        self._sum = 0.0
        self._sumsq = 0.0
        self._cnt = 0

    @property
    def phi_max(self):
        """The DECLARED maximum of Phi -- used for the psi scaling (spec 2.3) and
        for the Part-C headroom bound.  Never a measured maximum."""
        return 1.0 + self.phi_fire + self.phi_move

    def reset(self):
        self.phi[:] = 0.0
        self.raw[:] = 0.0

    def update(self, alive, fired, stride=None):
        """Advance Phi one step.  ``alive``/``fired`` are 0/1 arrays of length n;
        ``stride`` is the EXECUTED stride multiplier (read only when phi_move>0)."""
        alive = np.asarray(alive, dtype=np.float64).reshape(self.n)
        fired = np.asarray(fired, dtype=np.float64).reshape(self.n)
        raw = alive * (1.0 + self.phi_fire * fired)
        if self.phi_move > 0.0 and stride is not None:
            raw = raw + alive * self.phi_move * np.asarray(
                stride, dtype=np.float64).reshape(self.n)
        self.raw = raw
        self.phi = self.rho * self.phi + (1.0 - self.rho) * raw
        live = self.phi[alive > 0.0]
        if live.size:
            self._sum += float(live.sum())
            self._sumsq += float((live ** 2).sum())
            self._cnt += int(live.size)
        return self.phi

    def variation(self):
        """``std(Phi)/mean(Phi)`` over every live reading so far.  A.5's
        counter-check demands > 0.05; POWER ran 0.28."""
        if self._cnt < 2:
            return float("nan")
        m = self._sum / self._cnt
        v = max(0.0, self._sumsq / self._cnt - m * m)
        return float(math.sqrt(v) / m) if m > 1e-12 else float("nan")


# --------------------------------------------------------------------------- #
# Geometry: who is in whose way, right now
# --------------------------------------------------------------------------- #
def kernels(pos_i, pos_j, r_i, r_j, prox_len=PROX_LEN):
    """``prox`` and ``cone`` for every (i, j) pair and every move direction.

    ``prox[i, j] = exp(-clear_gap / prox_len)`` -- a body only obstructs you if it
    is near you.  ``cone[i, j, m] = max(0, cos angle)`` between the bearing to j
    and move direction m -- it only obstructs you in the directions it lies in.

    These are STATE, not the operator, exactly as POWER's PTDF is fixed while the
    flows it maps depend on the current injections.

    Returns ``prox`` (n, m) and ``cone`` (n, m, 4).
    """
    pos_i = np.asarray(pos_i, dtype=np.float64).reshape(-1, 2)
    pos_j = np.asarray(pos_j, dtype=np.float64).reshape(-1, 2)
    d = pos_j[None, :, :] - pos_i[:, None, :]                    # (n, m, 2)
    dist = np.linalg.norm(d, axis=2)                             # (n, m)
    touch = np.asarray(r_i, dtype=np.float64)[:, None] + \
        np.asarray(r_j, dtype=np.float64)[None, :]
    gap = np.maximum(0.0, dist - touch)
    prox = np.exp(-gap / float(prox_len))
    safe = np.where(dist > 1e-9, dist, 1.0)
    unit = d / safe[:, :, None]
    cone = np.maximum(0.0, np.einsum("ijk,dk->ijd", unit, MOVE_DIRS))   # (n, m, 4)
    cone = np.where((dist < 1e-9)[:, :, None], 0.0, cone)
    return prox, cone


def loading(W, prox, cone, phi, live_mask):
    """``L_i^m = sum_j W[i, j] prox_ij cone_ij^m Phi_j`` over LIVE j.

    Returns (n, 4).  The caller sums the ally and enemy terms separately and
    divides by the capacity -- keeping them apart is exactly what Part C's
    decomposition needs.
    """
    w = np.asarray(W, dtype=np.float64)
    ph = np.asarray(phi, dtype=np.float64) * np.asarray(live_mask, dtype=np.float64)
    contrib = w * prox * ph[None, :]                             # (n, m)
    return np.einsum("ij,ijd->id", contrib, cone)
