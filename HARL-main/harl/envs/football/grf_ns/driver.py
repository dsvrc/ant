"""The exogenous driver -- PITCH CONDITION over a match -- and the severity dial.

`PACT_NS_SPEC` NS-1.3 asks for a slow exogenous ``A(t)``, a function of
observable time alone, that no player's action can influence and that reaches
the agents ONLY by scaling the cross-agent term -- never by adding a term to the
loss.

WHICH CELL THIS IS
--------------------------------------------------------------------------------
`NS_design_guide.md` sorts non-stationarity three ways and II.6 cuts (C) again
by whether the harm has an inverse:

    (C, INVERTIBLE)   <- THIS INSTANCE.  simple_ns (VMAS) is the other one.
    (C, no inverse)   urb_ns / road_ns (traffic), smac_ns (overkill).
    (B) exogenous     the control, `ns_direct: 1` below.

The disturbance is a ROTATION of the agent's commanded heading, and rotations
of the eight compass directions form a group: every rotation has an exact
inverse and there is no saturation in the action space.  A correct estimate
therefore CANCELS the disturbance rather than routing around it -- II.6's first
row, where the method may claim identification AND compensation.

THE STORY, AND WHY EVERY CLAUSE IS A REQUIREMENT
--------------------------------------------------------------------------------
    Three attackers share one half of a pitch.  A player running with a
    teammate in his lane has to go round him: he swerves off the line he
    wanted.  How early and how wide he must swerve depends on the footing.  On
    a firm dry surface he can check and cut late; on a wet, greasy, cutting-up
    pitch he cannot check quickly, so he gives everyone more room, earlier.
    The pitch condition changes over a match -- rain comes and goes, the
    surface cuts up and then holds -- and nobody on the pitch controls it.
    A player ALONE in that half never swerves round anyone, however wet it is.

  * INTERACTION-MEDIATED.  The swerve is caused by TEAMMATES in the lane.  The
    sum runs over j != i, so a lone player reads exactly zero at every
    severity -- category C structurally, not because a number is small.
  * EXOGENOUS DRIVER.  Pitch condition over the match.  Asked "what makes this
    harder some days, that nobody here controls?" a footballer or a
    groundsman names the pitch and the weather before anything else.
  * NEVER A REWARD TERM.  It removes capability -- the player ends up a few
    degrees off the line he chose.  GRF's reward is untouched byte for byte.
  * INVERTIBLE.  Aiming off to allow for the swerve is what players do on a bad
    surface.  Knowing WHICH WAY he will be pushed (away from the teammate:
    public geometry) a scalar estimate of HOW MUCH (the unknown, drifting with
    the pitch) cancels it exactly.  Past one compass step per step the
    correction is no longer aiming off, it is running the wrong way -- that
    bound, ``corr_clip``, is the relief valve.
  * THE CLASSES ARE REAL.  A teammate dead AHEAD forces a bigger swerve than one
    on the FLANK; one BEHIND is not in the lane at all.  Which sector a
    teammate is in is public geometry.  What a teammate in that sector costs
    you TODAY is not -- it is the state of the surface, and it is what the
    estimator has to recover.

WHAT IS ANCHORED AND WHAT IS NOT (NS-2.4, stated honestly)
--------------------------------------------------------------------------------
`road_ns` anchors sigma = 1 on the Highway Capacity Manual's heavy-rain factor;
`smac_ns` on a published armour point.  Football has no such constant for
"how much wider you swerve on a wet pitch", so sigma = 1 here is a STATED
CALIBRATION PROCEDURE, not an anchor, and must be presented as one:

    at sigma = 1, at the driver's peak, with the squad at the scenario's own
    spawn geometry and every player under way, the swerve reaching a player is
    0.14 of one compass step (45 degrees) per step -- one 45-degree swerve
    every ~7 steps, an effective heading error of ~6 degrees.

0.14 is carried over from the HCM figure only so the ladders of the different
instances are readable against each other.

THE FOUR NS-2 REQUIREMENTS, ALL ASSERTED IN `certify()`
--------------------------------------------------------------------------------
  NS-2.1 identity at zero   sigma = 0 gives an amplitude of EXACTLY 0.0 at every
                            driver value, so the stock task is recovered byte
                            for byte.
  NS-2.2 monotone           amplitude non-decreasing in sigma at EVERY t.
  NS-2.3 never generous     amplitude >= 0 always: the surface never HELPS.
  NS-2.5 placebo regime     A(t) is EXACTLY zero over the dry half of every
                            cycle, so the whole sweep is byte-identical there.

`A` is a function of the step index alone -- observable time -- so a domain
feed-forward is available to every arm; what is withheld is nothing an agent
could see, and the coordination claim stays honest.

numpy only.  No gfootball.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["DialParams", "PitchDriver", "LOSS_AT_SIGMA1", "SECTOR_NAMES"]

#: The stated calibration target at sigma = 1.  NOT a published anchor -- see
#: the module docstring -- carried over from the HCM heavy-rain factor so the
#: ladders of the different instances share a scale.
LOSS_AT_SIGMA1 = 0.14

#: The declared element classes, ``r = 2``.  A public property of the geometry:
#: which sector of MY lane a teammate occupies.  Independent of N and of the
#: number of teammates (P-1.1).  Teammates BEHIND (> 135 degrees off the
#: heading) are not in the lane and are excluded from the operator by
#: declaration, not by pruning.
SECTOR_NAMES = ("front", "flank")


@dataclass(frozen=True)
class DialParams:
    """Every declared constant.  ``severity`` is the only experimental variable."""

    severity: float = 1.0
    """sigma.  0.0 makes the disturbance EXACTLY zero at every driver value."""

    # -- the driver (I.3 reference form) ----------------------------------------
    period: int = 12000
    """Steps per pitch-condition cycle -- about 100 typical 3v1 episodes, or
    four full 3000-step matches.  URB's driver unit is the DAY, not the step:
    the surface is effectively constant within an episode and drifts across
    them, and the clock is NOT reset between episodes (NS-3.4).  This is also
    what makes the estimator's job honest -- a 100-step RLS memory averages the
    sigma-delta quantisation noise and still tracks a 6000-step bump with a
    ~10% lag, whereas a 400-step cycle would sit on the tracking floor
    (trap 9) whatever mu is set to.  Rollout threads are de-phased across the
    cycle so a batch is a cycle average."""
    wet_fraction: float = 0.5
    """Fraction of the cycle the surface is bad.  The rest is EXACTLY dry, which
    is what makes NS-2.5's placebo provable rather than merely small."""
    loss_at_sigma1: float = LOSS_AT_SIGMA1
    mean_preserving: bool = False
    """NS-5.1: normalise A by its own cycle mean so only the SHAPE varies and
    the total swerve over a cycle is the same at every sigma.  Difficulty is then
    purely 'who is in whose lane when'."""

    # -- the medium ---------------------------------------------------------------
    rho: float = 0.6
    """Memory of the lane channel.  A swerve is committed to over a few steps
    rather than re-decided every frame.  The leak is on the PUBLIC channels, not
    on the private disturbance, so the estimator's model stays exactly linear in
    what the agent can compute (see ``coupling``)."""
    kernel_lambda: float = 0.12
    """Lane length scale in pitch units (the pitch is 2.0 long; 0.12 is ~6 m).
    ``kappa(d) = 1 / (1 + (d/lambda)^2)``: a teammate twice the lane length away
    counts a fifth as much.  Declared structure, evaluated on observed
    positions -- exactly as road_ns's loading is a declared operator on observed
    occupancy."""
    recv_ball: float = 1.5
    """Receiver susceptibility of the BALL CARRIER.  Dribbling needs room; the
    man on the ball swerves widest.  Public (ball ownership is in everyone's
    observation), and it is what makes the operator asymmetric."""

    # -- the declared classes (P-1.1 / P-1.2) -------------------------------------
    send_front: float = 1.4
    send_flank: float = 0.6
    """Per-sector SENDER gain, normalised to mean 1.  This is beta*: how much a
    teammate in that sector actually costs today.  NOT handed to the agent."""

    # -- sensor / channel bounds -----------------------------------------------------
    y_clip: float = 10.0
    """P-2.1's declared outlier bound on the sensor.  Reported as clip_frac."""
    corr_clip: float = 1.0
    """The relief valve: the correction never exceeds one compass step per step.
    Past that the agent is not aiming off, it is running the wrong way.  A
    physical bound on a physical quantity; it is also what closes trap 11 (a
    diverged estimate scoring well by accident)."""

    # -- the (B) control ------------------------------------------------------------
    direct: bool = False
    """THE (B) CONTROL.  Same driver, same amplitude, same channel -- but the
    swerve reaches every player DIRECTLY, in a fixed public direction, with no
    sum over j != i anywhere (a crosswind, if you like).  A lone player then
    feels it, which is exactly what makes that cell (B).  See ``channel``."""


class PitchDriver(object):
    """``A(t)`` and the severity dial.

    Mirrors ``urb_ns.driver.WeatherDriver`` / ``smac_ns.driver.GuardDriver``
    deliberately: a cross-cell comparison of the classification table must not
    be able to be a comparison of two dial implementations.
    """

    def __init__(self, p):
        self.p = p
        assert 0.0 < p.wet_fraction < 1.0, "wet_fraction must leave a placebo regime"
        assert p.loss_at_sigma1 >= 0.0
        assert p.period >= 2
        # the declared classes, normalised to mean 1 (P-1.2: declared, never fitted)
        raw = np.array([p.send_front, p.send_flank], dtype=np.float64)
        assert np.all(raw > 0), "sender gains must be positive"
        self.send = raw / raw.mean()
        self.r = int(self.send.size)

    # ------------------------------------------------------------------ driver
    def A(self, t):
        """Pitch badness in [0, 1] from the step index.  EXACTLY 0 when dry.

        A smooth bump that starts and ends at exactly zero -- ``sin^2`` on a
        clamped ramp rather than anything that merely gets small, because the
        exact zero is what buys the placebo regime (NS-2.5).  The same form the
        other instances use, so the placebo argument is the same argument.
        """
        ph = (np.asarray(t, dtype=np.float64) % self.p.period) / self.p.period
        x = np.clip(ph / self.p.wet_fraction, 0.0, 1.0)
        bump = np.sin(np.pi * x) ** 2
        return np.where(ph < self.p.wet_fraction, bump, 0.0)

    def is_placebo(self, t):
        """True on steps where the dial provably does nothing, for every sigma."""
        ph = (np.asarray(t, dtype=np.float64) % self.p.period) / self.p.period
        return ph >= self.p.wet_fraction

    def cycle_mean_A(self):
        return float(self.A(np.arange(self.p.period)).mean())

    # ------------------------------------------------------------------ dial
    def amplitude(self, t, sigma=None):
        """``sigma * L * A(t)`` -- compass steps of swerve per step at reference
        load.  The whole of what drifts.

        * sigma = 0 gives EXACTLY 0.0 at every t (NS-2.1): the product carries
          sigma as a factor, so no rounding can leave a residue.
        * linear in sigma with a non-negative coefficient (NS-2.2, NS-2.3).
        * exactly zero wherever A(t) is exactly zero (NS-2.5).
        """
        s = float(self.p.severity if sigma is None else sigma)
        a = np.asarray(self.A(t), dtype=np.float64)
        if self.p.mean_preserving:
            m = self.cycle_mean_A()
            if m > 0.0:
                a = a / m
        return s * self.p.loss_at_sigma1 * a

    def beta_star(self, t, sigma=None):
        """The true per-class gain right now, ``(r,)``: ``amplitude * send``."""
        return float(self.amplitude(t, sigma)) * self.send

    # ------------------------------------------------------------------ report
    def swerve_over_cycle(self, sigma):
        """Mean swerve (compass steps / step) over one full cycle at reference
        load -- NS-1.6's 'capacity removed', in this channel's units."""
        return float(self.amplitude(np.arange(self.p.period), sigma).mean())

    def swing(self, sigma):
        """Peak-to-trough of the amplitude over the cycle: the part a
        coordination method can act on, as opposed to the level."""
        amp = self.amplitude(np.arange(self.p.period), sigma)
        return float(amp.max()) - float(amp.min())

    def placebo_steps(self):
        return int(np.sum(self.is_placebo(np.arange(self.p.period))))

    def report(self, sigmas=(1.0, 3.0)):
        rows = []
        for s in sigmas:
            rows.append(dict(
                sigma=float(s),
                mean_swerve=self.swerve_over_cycle(s),
                peak_swerve=float(self.amplitude(np.arange(self.p.period), s).max()),
                swing=self.swing(s),
                placebo_steps=self.placebo_steps(),
                period=int(self.p.period),
                beyond_physical=bool(s > 1.0),
            ))
        return rows

    # ------------------------------------------------------------------ gates
    def certify(self, sigmas=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)):
        """Assert NS-2.1, NS-2.2, NS-2.3 and NS-2.5 over the WHOLE driver domain.

        Raises AssertionError on violation; returns the measured facts.  Cheap
        enough to run at construction in every process, which is where it runs.
        """
        t = np.arange(self.p.period)
        a = self.A(t)
        assert np.all(a >= 0.0) and np.all(a <= 1.0), "A(t) must lie in [0, 1]"
        pl = self.is_placebo(t)
        assert pl.any(), "NS-2.5 violated: no placebo regime in the cycle"
        assert (~pl).any(), "the driver never fires"
        assert np.all(a[pl] == 0.0), "NS-2.5 violated: A(t) is not EXACTLY 0 when dry"
        assert float(a.max()) > 0.99, "the driver never reaches its peak"

        amp0 = self.amplitude(t, 0.0)
        assert np.all(amp0 == 0.0), "NS-2.1 violated: amplitude(sigma=0) != 0 somewhere"

        prev = amp0
        rows = []
        for s in sorted(float(x) for x in sigmas):
            amp = np.asarray(self.amplitude(t, s), dtype=np.float64)
            assert np.all(amp >= 0.0), "NS-2.3 violated: negative amplitude at sigma=%g" % s
            assert np.all(amp >= prev - 1e-15), (
                "NS-2.2 violated: amplitude fell with sigma somewhere (sigma=%g)" % s)
            prev = amp
            assert np.all(amp[pl] == 0.0), (
                "NS-2.5 violated: the dial acts inside the placebo at sigma=%g" % s)
            rows.append(dict(sigma=s, amp_max=float(amp.max()), amp_mean=float(amp.mean()),
                             inert_frac=float(np.mean(amp == 0.0))))
        return dict(rows=rows, period=int(self.p.period), wet_fraction=self.p.wet_fraction,
                    loss=self.p.loss_at_sigma1, placebo_steps=int(pl.sum()),
                    anchor="STATED CALIBRATION, not a published constant: 0.14 step/step "
                           "at sigma=1, peak, spawn geometry")

    def banner(self):
        r = self.report((1.0,))[0]
        return ("[GRF-NS] pitch driver  period=%d  placebo=%d/%d steps  "
                "L@sigma=1=%.3f step/step (stated calibration, NOT an anchor)  "
                "mean_swerve=%.4f  swing=%.4f  mean_preserving=%d  send=%s"
                % (self.p.period, self.placebo_steps(), self.p.period,
                   self.p.loss_at_sigma1, r["mean_swerve"], r["swing"],
                   int(self.p.mean_preserving), np.round(self.send, 3).tolist()))
