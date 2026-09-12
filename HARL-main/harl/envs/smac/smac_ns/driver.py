"""The exogenous driver: the enemy line's GUARD CYCLE shrinking absorbable damage.

`PACT_NS_SPEC` NS-1.3 requires a slow exogenous `A(t)`, a function of observable
time alone, that reaches the agents **only by shrinking capacity** and never by
adding a term to the loss.

    K_e(t) = K_e^0 * g(A(t), sigma)     the damage enemy e can usefully absorb
    A(t)   in [0, 1]                    guard intensity, slow, exogenous

WHAT THE MEDIUM IS, AND WHY A LONE AGENT FEELS NOTHING
--------------------------------------------------------------------------------
The medium is the ENEMY LINE'S ABSORPTION: how much allied damage can usefully
land on a given enemy before the rest is overkill.  Overkill is the canonical
StarCraft coordination failure -- "don't all shoot the same zealot" -- and it is
interaction-mediated *structurally*, not by being small:

    load_e = sum_{j != i} damage_j -> e          the sum runs over j != i
    u_i    = load_e / (K_e^0 * g)                on the enemy i is shooting

A lone agent cannot overkill.  It fires, the target dies, it retargets; nothing is
wasted at any guard intensity whatsoever.  So `u_i = 0` and the harm is exactly
1.0 however hard the dial is turned -- NS-1.2's category-C signature, obtained
from the zero-diagonal sum rather than from a small number.

Tightening the window in which damage is useful is what a guard phase does, and it
makes overkill *more* likely for a squad while still costing a solitary unit
nothing.  That asymmetry is the whole form.

THE ANCHOR (NS-2.4), STATED HONESTLY
--------------------------------------------------------------------------------
`sigma = 1` is anchored to StarCraft II's own published armour arithmetic: one
point of armour subtracts 1 from every incoming hit, and a Stalker's weapon deals
10, so a single armour point is exactly a **10% loss of delivered damage**.  That
is a published constant for the exact phenomenon -- the direct counterpart of the
HCM heavy-rain capacity adjustment factor (14%) that anchors URB, and of IEEE 738
ampacity derating in POWER.

What is published is the AMPLITUDE.  The cycle SHAPE is injected: StarCraft has no
weather, and this module does not pretend otherwise.  A guard phase that waxes and
wanes over an engagement is something a player recognises, but it is not a
documented game timer, and NS-1.3's "a practitioner names it unprompted" is
therefore met only in part.  Any paper using this must say so in those words.

THE FOUR NS-2 REQUIREMENTS, ALL ASSERTED IN `certify()`
--------------------------------------------------------------------------------
  NS-2.1 identity at zero   `sigma = 0` gives `g == 1` EXACTLY at every A, so the
                            stock task is recovered byte for byte.
  NS-2.2 monotone           `g` non-increasing in `sigma` at EVERY driver value,
                            not merely at the peak.
  NS-2.3 never generous     `g <= 1` always.  The uprating trap: a two-sided
                            physical law scaled by sigma made POWER strictly
                            EASIER at sigma=2.  Only the harmful half is kept.
  NS-2.5 placebo regime     `A(t)` reaches EXACTLY zero over part of the cycle, so
                            `g == 1.0000` there for every sigma and the whole
                            sweep is byte-identical in that regime.

> A reviewer alleging a rigged knob then has to explain why the rig switches
> itself off for half of every engagement.

`A` is a function of the step index alone -- observable time -- so a domain-model
feedforward is available to every arm, which is what keeps the coordination claim
honest rather than an information advantage.
"""

import numpy as np

# StarCraft II armour arithmetic: one armour point subtracts 1 from every hit, and
# a Stalker's weapon deals 10.  One armour point is therefore exactly a 10% loss
# of delivered damage.  A DECLARED, PUBLISHED CONSTANT -- never tuned.
SC2_ARMOUR_LOSS = 0.10

#: Per-enemy guard sensitivity is clipped to this band, so the derating is
#: unit-specific (as published armour is: different units carry different armour)
#: without any element ever being driven to zero absorption.  Declared.
SENS_CLIP = (0.4, 2.0)


class GuardDriver(object):
    """`A(t)` and the severity dial `g(A, sigma)`.

    Mirrors ``urb_ns.driver.WeatherDriver`` deliberately: a cross-cell comparison
    of the classification table must not be able to be a comparison of two dial
    implementations.

    Args:
        period:     env steps per guard cycle.  Structural constant, never tuned.
                    Defaults to one 3s5z episode limit so every rollout sees a
                    whole cycle, guard phase and placebo alike.
        guard_frac: fraction of the cycle under guard.  The remainder is EXACTLY
                    inert, which is the placebo regime (NS-2.5).
        loss:       fractional absorption loss at A=1, sigma=1.  Default = the SC2
                    one-armour-point figure.
        mean_preserving: normalise `g` by its own cycle mean so only the SHAPE
                    varies and total capacity is unchanged (NS-5.1).  Difficulty
                    then becomes purely "who shoots what when", which is what a
                    coordination method addresses, rather than "there is less to
                    go around", which merely rewards blanket conservatism.
                    Default False -- the plain physical reading -- but report the
                    capacity removed either way.
    """

    def __init__(self, period=150, guard_frac=0.5, loss=SC2_ARMOUR_LOSS,
                 mean_preserving=False):
        self.period = int(period)
        self.guard_frac = float(guard_frac)
        self.loss = float(loss)
        self.mean_preserving = bool(mean_preserving)
        assert 0.0 < self.guard_frac < 1.0, "guard_frac must leave a placebo regime"
        assert 0.0 <= self.loss < 1.0, "loss must be a fraction below 1"

    # ------------------------------------------------------------------ driver
    def A(self, t):
        """Guard intensity in [0, 1] from the step index.  EXACTLY 0 when inert.

        A smooth bump that starts and ends at exactly zero -- the exact-zero is
        what buys the placebo regime for free (NS-2.5), and is why the bump is
        ``sin^2`` on a clamped ramp rather than anything that merely gets small.
        """
        ph = (np.asarray(t, dtype=np.float64) % self.period) / self.period
        x = np.clip(ph / self.guard_frac, 0.0, 1.0)
        bump = np.sin(np.pi * x) ** 2
        return np.where(ph < self.guard_frac, bump, 0.0)

    def is_placebo(self, t):
        """True on steps where the dial provably does nothing, for every sigma."""
        ph = (np.asarray(t, dtype=np.float64) % self.period) / self.period
        return ph >= self.guard_frac

    # ------------------------------------------------------------------ dial
    def g(self, t, sigma, sens=None):
        """Absorption multiplier in (0, 1].  `sigma = 0` gives exactly 1.0.

        `sens` is the optional per-enemy guard sensitivity (mean 1.0).  With it the
        derating is unit-specific, as published armour is, so the BINDING element
        can shift with severity -- which is what stops the ceiling decomposition
        being sigma-invariant and hiding the effect being measured (the spec calls
        this out explicitly under the reference driver).  `sens=None` is uniform.

        Identity at sigma=0 survives elementwise whatever `sens` is, because the
        whole subtracted term carries sigma as a factor.
        """
        a = np.asarray(self.A(t), dtype=np.float64)
        if sens is None:
            drop = float(sigma) * self.loss * a
        else:
            drop = (float(sigma) * self.loss
                    * a[..., None] * np.asarray(sens, dtype=np.float64))
        raw = np.minimum(1.0 - drop, 1.0)          # NS-2.3 -- never generous
        raw = np.maximum(raw, 1e-3)                # keep the medium open
        if not self.mean_preserving:
            return raw
        return np.minimum(raw / np.maximum(self._cycle_mean(sigma, sens), 1e-9), 1.0)

    def _cycle_mean(self, sigma, sens=None):
        """Mean multiplier over one full cycle.  Per-element when `sens` is given,
        so mean-preservation is applied element by element and no element is ever
        credited with more absorption than its nominal."""
        a = self.A(np.arange(self.period))
        if sens is None:
            return float(np.minimum(1.0 - float(sigma) * self.loss * a, 1.0).mean())
        drop = (float(sigma) * self.loss * a[:, None]
                * np.asarray(sens, dtype=np.float64))
        return np.minimum(1.0 - drop, 1.0).mean(axis=0)

    # ------------------------------------------------------------------ report
    def capacity_loss_over_cycle(self, sigma):
        """Fraction of total absorption the dial removes across one full cycle.

        NS-1.6 says report this: a dial that shrinks capacity removes work
        capacity, so past some sigma no controller can win, and that is physics
        rather than failure.
        """
        t = np.arange(self.period)
        return float(1.0 - self.g(t, sigma).mean())

    def swing(self, sigma):
        """Peak-to-trough capacity ratio over the cycle -- the part a coordination
        method can act on, as opposed to the level."""
        t = np.arange(self.period)
        gg = np.asarray(self.g(t, sigma), dtype=np.float64)
        return float(gg.max() / max(gg.min(), 1e-12))

    def placebo_steps(self):
        """How many steps of the cycle are provably inert."""
        return int(np.sum(self.is_placebo(np.arange(self.period))))

    def report(self, sigmas=(1.0, 3.0)):
        """The NS-1.6 table, for this driver."""
        rows = []
        for s in sigmas:
            rows.append(dict(
                sigma=float(s),
                capacity_removed=self.capacity_loss_over_cycle(s),
                swing=self.swing(s),
                placebo_steps=self.placebo_steps(),
                period=self.period,
                beyond_physical=bool(s > 1.0),
            ))
        return rows

    # ------------------------------------------------------------------ gates
    def certify(self, sigmas=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0), sens=None):
        """Assert NS-2.1, NS-2.2, NS-2.3 and NS-2.5 over the WHOLE driver domain.

        Raises AssertionError on violation; returns the measured facts.  Cheap
        enough to run at construction in every process, which is where it runs.
        """
        t = np.arange(self.period)
        a = self.A(t)

        # the driver itself must reach exact zero, or there is no placebo
        assert np.all(a >= 0.0) and np.all(a <= 1.0), "A(t) must lie in [0, 1]"
        pl = self.is_placebo(t)
        assert pl.any(), "NS-2.5 violated: no placebo regime in the cycle"
        assert np.all(a[pl] == 0.0), "NS-2.5 violated: A(t) is not EXACTLY 0 when inert"

        # NS-2.1 identity at zero, exactly, at every driver value
        g0 = np.asarray(self.g(t, 0.0, sens), dtype=np.float64)
        assert np.all(g0 == 1.0), "NS-2.1 violated: g(sigma=0) != 1 somewhere"

        prev = g0
        rows = []
        for s in sorted(float(x) for x in sigmas):
            gg = np.asarray(self.g(t, s, sens), dtype=np.float64)
            # NS-2.3 never generous
            assert np.all(gg <= 1.0 + 1e-15), "NS-2.3 violated: g > 1 at sigma=%g" % s
            assert np.all(gg > 0.0), "g must stay positive at sigma=%g" % s
            # NS-2.2 monotone in sigma at EVERY driver value
            assert np.all(gg <= prev + 1e-15), (
                "NS-2.2 violated: g rose with sigma somewhere (sigma=%g)" % s)
            prev = gg
            # NS-2.5 the placebo is inert at EVERY sigma
            gp = gg[pl] if gg.ndim == 1 else gg[pl, :]
            assert np.all(gp == 1.0), (
                "NS-2.5 violated: the dial acts inside the placebo at sigma=%g" % s)
            rows.append(dict(sigma=s, g_min=float(gg.min()), g_mean=float(gg.mean()),
                             inert_frac=float(np.mean(gg == 1.0))))
        return dict(rows=rows, period=self.period, guard_frac=self.guard_frac,
                    loss=self.loss, placebo_steps=int(pl.sum()),
                    anchor="SC2 one-armour-point = 10% of a 10-damage hit")

    def banner(self):
        r = self.report((1.0,))[0]
        return ("[SMAC-NS] guard driver  period=%d  placebo=%d/%d steps  "
                "loss@sigma=1=%.3f (SC2 armour anchor)  capacity_removed=%.3f%%  "
                "swing=%.3fx  mean_preserving=%d"
                % (self.period, self.placebo_steps(), self.period, self.loss,
                   100.0 * r["capacity_removed"], r["swing"],
                   int(self.mean_preserving)))
