"""The exogenous driver A(t) and the severity dial g(sigma, A).

`NS_FORM_SPEC` A.2 object 3 and Part B.  The driver reaches the agents ONLY by
shrinking the medium's capacity:

    K_i(t) = K_i^0 * g(sigma, A(t))            never  L_i += B(t)

which is what makes the non-stationarity category C rather than category B in
disguise (A.3).  Because loading is a RATIO, shrinking the denominator multiplies
EVERY term including the peer sum, and at N = 1 the peer sum is empty, so the
cross-agent contribution is exactly zero however small g becomes.  Irreducibility
is structural here, not something verified afterwards.

THE STORY.  The enemy line does not stand still: it pushes and consolidates on a
slow cycle, and the allied squad's usable frontage breathes with it.  A(t) is
that push-consolidate cycle -- exogenous (the built-in StarCraft II AI drives it,
not the learned policy), slow, and computable from observable time.

THE FOUR B.1 REQUIREMENTS, and where each is enforced:

  1. IDENTITY AT ZERO.  ``g(0, A) == 1.0`` exactly, at EVERY driver value.  The
     subtracted term carries sigma as a factor, so this is exact arithmetic and
     not an approximation.  ``assert_dial()`` checks it over the whole domain.
  2. MONOTONE IN DIFFICULTY at every driver value.  The subtracted term is
     non-negative and scales linearly in sigma, and clipping is monotone.
  3. NEVER GENEROUS.  ``g <= 1`` always -- B.2's uprating trap.  On POWER a
     two-sided physical law made sigma=2 strictly EASIER (reference survival
     350 -> 901 steps).  Only the harmful half is kept.
  4. PHYSICALLY ANCHORED.  SMAC has no textbook derating curve (the spec's E-table
     records "--" for the anchor), so ``depth`` is anchored instead to a MEASURED
     property of the stock game: fc/certificates.py's ``anchor`` gate reports the
     spread of realized-over-commanded displacement in a stock episode, i.e. how
     much StarCraft II's own body-blocking already costs a moving unit.  sigma=1
     is set so the dial's peak equals that.  Anything above sigma=1 is labelled a
     beyond-physical stress test in every table it appears in.

THE PLACEBO REGIME (B.4) -- the single strongest defence against "rigged knob".
For A(t) <= ``knee`` the dial returns exactly 1.0 at EVERY severity, so the
consolidate phase of the cycle is provably inert.  Run the identical sweep there
and every sigma row must be byte-identical.  The spec's E-table lists SMAC's
placebo regime as "--"; this design has one.
"""

import math

import numpy as np

DEFAULT_KNEE = 0.35     # driver values at or below this are provably inert
DEFAULT_DEPTH = 0.60    # capacity shrink at sigma=1, A=1  (the anchored amplitude)
DEFAULT_FLOOR = 0.25    # g never goes below this: a unit is never fully frozen


def driver(clock, period, phase0=0.0):
    """A(t) in [0, 1] -- the enemy line's push/consolidate cycle.

    A raised cosine on a clock that PERSISTS ACROSS EPISODES: the campaign keeps
    running whether or not this squad died, which is what makes it exogenous.
    ``phase0`` de-phases parallel workers so the rollout batch tiles the cycle
    instead of sampling one phase (NS_FORM_SPEC E.2 pitfall 6).
    """
    ph = (float(clock) / float(period) + float(phase0)) % 1.0
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * ph)


def dial(sigma, A, knee=DEFAULT_KNEE, depth=DEFAULT_DEPTH, floor=DEFAULT_FLOOR):
    """g(sigma, A) in (0, 1] -- the capacity multiplier.

    ``g = 1 - sigma * depth * max(0, A - knee) / (1 - knee)``, clipped to
    ``[floor, 1]``.  All four B.1 requirements hold by construction; see the
    module docstring and ``assert_dial``.
    """
    sigma = float(sigma)
    A = np.asarray(A, dtype=np.float64)
    excess = np.maximum(0.0, A - float(knee)) / (1.0 - float(knee))
    g = 1.0 - sigma * float(depth) * excess
    g = np.clip(g, float(floor), 1.0)
    return float(g) if np.ndim(A) == 0 else g


def is_placebo(A, knee=DEFAULT_KNEE):
    """True where the dial is provably inert (B.4).  ``g == 1.0`` there at every
    severity, so those steps must be byte-identical across a severity sweep."""
    return np.asarray(A, dtype=np.float64) <= float(knee)


def assert_dial(sigmas=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0), n=1001,
                knee=DEFAULT_KNEE, depth=DEFAULT_DEPTH, floor=DEFAULT_FLOOR):
    """Certify B.1's four requirements over the WHOLE driver domain.

    Returns a dict of measured facts; raises AssertionError on a violation.  This
    is gate G2 (identity) plus the monotonicity and never-generous checks, and it
    needs neither StarCraft II nor a trained policy.
    """
    A = np.linspace(0.0, 1.0, int(n))
    g0 = dial(0.0, A, knee, depth, floor)
    # 1. identity at zero, EXACTLY, at every driver value
    assert np.all(g0 == 1.0), "B.1.1 violated: g(0, A) != 1 at some A"
    prev = g0
    rows = []
    for s in sorted(float(x) for x in sigmas):
        g = dial(s, A, knee, depth, floor)
        # 3. never generous
        assert np.all(g <= 1.0 + 1e-15), "B.1.3 violated: g > 1 at sigma=%g" % s
        assert np.all(g > 0.0), "g must stay positive at sigma=%g" % s
        # 2. monotone in sigma AT EVERY DRIVER VALUE (not just at the peak)
        assert np.all(g <= prev + 1e-15), \
            "B.1.2 violated: g rose with sigma at some A (sigma=%g)" % s
        prev = g
        # 4/B.4. the placebo regime is exactly inert
        pl = is_placebo(A, knee)
        assert np.all(g[pl] == 1.0), \
            "B.4 violated: dial acts inside the placebo regime at sigma=%g" % s
        rows.append(dict(sigma=s, g_min=float(g.min()), g_mean=float(g.mean()),
                         inert_frac=float(np.mean(g == 1.0))))
    return dict(rows=rows, knee=float(knee), depth=float(depth), floor=float(floor),
                placebo_frac=float(np.mean(is_placebo(A, knee))))


class Driver:
    """The live driver: a persistent clock, the dial, and the severity curriculum.

    ``severity`` is read from the TASK config, never from a method's block
    (NS_FORM_SPEC B.5): the dial sits BELOW the compensator in the class hierarchy
    so every arm -- MAPPO, HAPPO, PACT -- experiences the identical physics.

    THE CURRICULUM IS OFF BY DEFAULT AND THAT IS DELIBERATE.  A warmup that holds
    sigma at 0 for the first N frames makes the compensator provably inert over
    that whole stretch, so any arm difference measured during it is basin luck,
    not method.  Set ``warmup``/``ramp`` only if you intend that, and never
    compare arms inside the warmup window.
    """

    def __init__(self, severity=1.0, period=75, phase0=0.0, knee=DEFAULT_KNEE,
                 depth=DEFAULT_DEPTH, floor=DEFAULT_FLOOR, warmup=0, ramp=0,
                 freeze=None, eval_mode=False, mean_preserving=False):
        self.severity = float(severity)
        self.period = max(2, int(period))
        self.phase0 = float(phase0)
        self.knee = float(knee)
        self.depth = float(depth)
        self.floor = float(floor)
        self.warmup = int(warmup)
        self.ramp = int(ramp)
        # freeze pins A(t) to a constant -- calibration and the certificates only.
        self.freeze = None if freeze is None else float(freeze)
        # eval envs skip the curriculum so evaluation measures the harmed task.
        self.eval_mode = bool(eval_mode)
        # D.2 option 2: divide g by its own cycle mean so total capacity is
        # unchanged and only the SHAPE varies.  Satisfies G4a at the cost of
        # B.1.3 (g then exceeds 1 somewhere), so it is OFF by default and is
        # reported explicitly whenever it is on.
        self.mean_preserving = bool(mean_preserving)
        self.clock = 0                 # persists across episodes -- exogenous
        self.age = 0                   # steps since construction; curriculum only
        self._mp_norm = self._cycle_mean() if self.mean_preserving else 1.0

    def _cycle_mean(self):
        A = np.array([driver(k, self.period, 0.0) for k in range(self.period)])
        return float(np.mean(dial(self.severity, A, self.knee, self.depth, self.floor)))

    def current_severity(self):
        """The severity ACTUALLY APPLIED this step (after any curriculum)."""
        if self.eval_mode or self.warmup <= 0:
            return self.severity
        if self.age < self.warmup:
            return 0.0
        if self.ramp <= 0:
            return self.severity
        frac = min(1.0, (self.age - self.warmup) / float(self.ramp))
        return self.severity * frac

    def A(self):
        return self.freeze if self.freeze is not None else \
            driver(self.clock, self.period, self.phase0)

    def g(self):
        """The capacity multiplier for this step, and the driver value behind it."""
        a = self.A()
        s = self.current_severity()
        val = dial(s, a, self.knee, self.depth, self.floor)
        if self.mean_preserving and self._mp_norm > 1e-9:
            val = val / self._mp_norm
        return float(val), float(a), float(s)

    def advance(self):
        self.clock += 1
        self.age += 1

    def state(self):
        return dict(clock=int(self.clock), age=int(self.age))
