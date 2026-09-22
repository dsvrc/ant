"""The exogenous driver -- DRIVETRAIN THERMAL STATE over a run -- and the dial.

`PACT_NS_SPEC` NS-1.3 asks for a slow exogenous ``A(t)``, a function of
observable time alone, that no agent's action can influence and that reaches the
agents ONLY by scaling the cross-agent term -- never by adding a term to the
loss.

WHICH CELL THIS IS
--------------------------------------------------------------------------------
    (C, INVERTIBLE)   <- THIS INSTANCE.  With `simple_ns` (VMAS point masses)
                         and `grf_ns` (a discrete heading group), this is the
                         third action space in which the same method cancels the
                         same form: continuous joint torque.
    (C, no inverse)   `smac_ns`, `urb_ns` / `road_ns`.
    (B) exogenous     the control, ``ns_direct: 1``.

The disturbance is an unmodelled torque at the agent's own joints, so it lives
in exactly the space its action lives in and a correct estimate SUBTRACTS it --
II.6's first row, identification AND compensation.  There is no gremlin in the
channel: it is the trunk reaction the robot already produces, with a gain that
drifts.

THE STORY, AND WHY EVERY CLAUSE IS A REQUIREMENT
--------------------------------------------------------------------------------
    Four legs are bolted to one torso.  When another leg drives hard the trunk
    reacts, and that reaction arrives at my hips as a torque I never commanded:
    I am holding against my neighbours.  How much of a neighbour's push reaches
    me depends on the state of the drivetrain -- a cold machine is stiff and
    crisply preloaded; over a run the gearbox and bearing grease warm, viscosity
    falls, preload drops and backlash opens, and the same neighbour push arrives
    through a softer, sloppier path.  Nobody on the robot decides how warm it
    is.  A leg alone on the torso -- the others slack -- feels nothing from the
    others at any temperature.

  * INTERACTION-MEDIATED.  The reaction is caused by the OTHER agents' torques;
    the sum runs over ``j != i``, so a lone agent reads exactly zero at every
    severity -- structurally, not because a number is small.
  * EXOGENOUS DRIVER.  Drivetrain temperature over a run.  Ask any roboticist
    what makes the machine behave differently from one session to the next and
    "it hadn't warmed up yet" comes back without prompting.
  * NEVER A REWARD TERM.  It adds a torque at the actuator, below the reward.
    Ant's reward -- forward velocity, control cost, contact cost, survival -- is
    byte-for-byte the host's own, and its control cost is charged on the torque
    the POLICY commanded (see ``layer.py``).  The ant earns less only because it
    physically walks worse.
  * INVERTIBLE.  Subtracting a feed-forward estimate of a known disturbance at
    the actuator is what every real joint controller does, and it dies at the
    torque limit -- which is where sigma* comes from rather than from a number
    we chose.
  * THE CLASSES ARE REAL.  Hip-to-hip, ankle-to-ankle and the cross path are
    public (which joint is which is the hardware).  What a unit of a
    neighbour's torque on each path costs you TODAY is not: that is the thermal
    state, and it is what the estimator must recover.

WHAT IS ANCHORED AND WHAT IS NOT (NS-2.4, stated honestly)
--------------------------------------------------------------------------------
`road_ns` anchors sigma = 1 on the Highway Capacity Manual's heavy-rain factor;
`smac_ns` on a published SC2 armour point.  A simulated quadruped has no
published constant for "how much more of your neighbour's push reaches you when
the gearbox is warm", so sigma = 1 here is a STATED CALIBRATION PROCEDURE, not
an anchor, and must be presented as one:

    at sigma = 1, at the driver's peak, with every peer commanding its full
    torque range, the disturbance reaching an agent is 0.14 of its own torque
    range.

0.14 is carried over from the HCM figure only so the ladders of the different
instances share a severity scale and can be read against each other.  This is
the same statement ``simple_ns`` makes, deliberately: the VMAS and Ant rows then
measure the same physical severity in two different action spaces.

THE FOUR NS-2 REQUIREMENTS, ALL ASSERTED IN `certify()`
--------------------------------------------------------------------------------
  NS-2.1 identity at zero   sigma = 0 gives an amplitude of EXACTLY 0.0 at every
                            driver value, so the stock task is recovered byte
                            for byte.
  NS-2.2 monotone           amplitude non-decreasing in sigma at EVERY t.
  NS-2.3 never generous     amplitude >= 0 always: warming never HELPS.
  NS-2.5 placebo regime     A(t) is EXACTLY zero over the cold half of every
                            cycle, so the whole sweep is byte-identical there.

numpy only.  No mujoco.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["DialParams", "ThermalDriver", "LOSS_AT_SIGMA1"]

#: The stated calibration target at sigma = 1, as a fraction of the agent's own
#: torque range.  NOT a published anchor -- see the module docstring.
LOSS_AT_SIGMA1 = 0.14


@dataclass(frozen=True)
class DialParams:
    """Every declared constant.  ``severity`` is the only experimental variable."""

    severity: float = 1.0
    """sigma.  0.0 makes the disturbance EXACTLY zero at every driver value."""

    # -- the driver (I.3 reference form) ----------------------------------------
    period: int = 20000
    """Steps per thermal cycle.  Ant runs at dt = 0.05 s, so 20000 steps is
    about 17 minutes of operation and 20 episodes of the 1000-step limit -- the
    time scale a drivetrain actually warms and cools over.  The clock persists
    across episodes (NS-3.4) and is de-phased across rollout threads, so a
    rollout batch is a cycle average rather than one phase of it.

    MEASURED, because the obvious claim is not quite true: A moves by at most
    3.1e-4 per step, 0.031 over the estimator's own 100-step memory, and up to
    0.31 over a FULL 1000-step episode.  So the surface is slow against the
    control loop and against the estimator, but it is NOT frozen within a
    long episode -- a gearbox does warm measurably inside a minute of hard work.
    Say that, rather than "constant within an episode".

    It also keeps the estimator's job honest: a 100-step RLS memory (mu = 0.99)
    tracks a 10000-step bump with a few per cent lag, where a cycle of an
    episode's own length would sit on the tracking floor whatever mu was set to
    (the porting brief's trap 9)."""
    warm_fraction: float = 0.5
    """Fraction of the cycle the machine is warm.  The rest is EXACTLY cold,
    which is what makes NS-2.5's placebo provable rather than merely small."""
    loss_at_sigma1: float = LOSS_AT_SIGMA1
    mean_preserving: bool = False
    """NS-5.1: normalise A by its own cycle mean so only the SHAPE varies and the
    total transmitted load over a cycle is the same at every sigma.  Difficulty
    is then purely "who pushes when"."""

    # -- the medium ---------------------------------------------------------------
    rho: float = 0.8
    """Memory of the trunk channel.  The structure does not respond instantly:
    compliance and backlash integrate over a few control steps.  The leak is
    applied to the PUBLIC channels, never to the private disturbance, so the
    estimator's model stays EXACTLY linear in quantities the agent can compute
    (see ``coupling``)."""
    length_scale: float = 0.25
    """Transmission length scale, in metres: ``ant.xml``'s own torso radius.  A
    published number from the model file."""
    recv_spread: float = 0.35
    """Spread of the per-actuator receiver susceptibility, normalised to mean 1.

    A DECLARED heterogeneity of the hardware -- no two drivetrains on a real
    machine are identical, and the unit that has done more work is more easily
    thrown about by the same trunk reaction.  It is what makes the operator
    ASYMMETRIC, since the transmission structure itself is symmetric either way.

    It is NOT claimed to be measurable in this simulator.  An earlier version
    declared a hip/ankle ratio of 1.86 on the argument that an ankle carries
    only the foot; the model was asked and answered 1.01, so the argument was
    wrong and was removed rather than defended.  See ``structure.recv_vector``."""

    # -- the declared classes (P-1.1, P-1.2) --------------------------------------
    send_hh: float = 1.5
    send_aa: float = 0.9
    send_cross: float = 0.6
    """Per-class SENDER gain, normalised to mean 1.  This is beta*, what the
    estimator must recover, and it is NOT handed to the agent.  Hips bolt
    straight to the trunk and load each other hardest; ankles hang two segments
    out and are the most isolated."""

    # -- sensor / channel bounds -----------------------------------------------------
    y_clip: float = 10.0
    """P-2.1's declared outlier bound on the sensor.  Reported as clip_frac."""
    corr_clip: float = 0.5
    """The relief valve, as a fraction of the torque range: the feed-forward
    never exceeds half of what the actuator can deliver.  A guard rail on a
    physical quantity, and the reason a diverged estimate can only fail to help
    (the porting brief's trap 11) rather than score well by accident through a
    large saturating kick."""

    # -- the (B) control ------------------------------------------------------------
    direct: bool = False
    """THE (B) CONTROL.  Same driver, same amplitude, same channel -- but the
    disturbance reaches every agent DIRECTLY, along its own commanded torque,
    with no sum over ``j != i`` anywhere.  A lone agent then feels it, which is
    exactly what makes that cell (B).  See ``channel.py``."""


class ThermalDriver(object):
    """``A(t)`` and the severity dial.

    Deliberately the same object as ``grf_ns.PitchDriver`` and
    ``urb_ns.WeatherDriver``: a cross-cell comparison of the classification must
    not be able to be a comparison of three dial implementations.
    """

    def __init__(self, p):
        self.p = p
        assert 0.0 < p.warm_fraction < 1.0, "warm_fraction must leave a placebo regime"
        assert p.loss_at_sigma1 >= 0.0
        assert p.period >= 2
        raw = np.array([p.send_hh, p.send_aa, p.send_cross], dtype=np.float64)
        assert np.all(raw > 0), "sender gains must be positive"
        self.send = raw / raw.mean()
        self.r = int(self.send.size)

    # ------------------------------------------------------------------ driver
    def A(self, t):
        """Thermal load in [0, 1] from the step index.  EXACTLY 0 when cold.

        A smooth bump that starts and ends at exactly zero -- ``sin^2`` on a
        clamped ramp rather than anything that merely gets small, because the
        exact zero is what buys the placebo regime (NS-2.5).
        """
        ph = (np.asarray(t, dtype=np.float64) % self.p.period) / self.p.period
        x = np.clip(ph / self.p.warm_fraction, 0.0, 1.0)
        bump = np.sin(np.pi * x) ** 2
        return np.where(ph < self.p.warm_fraction, bump, 0.0)

    def is_placebo(self, t):
        """True on steps where the dial provably does nothing, for every sigma."""
        ph = (np.asarray(t, dtype=np.float64) % self.p.period) / self.p.period
        return ph >= self.p.warm_fraction

    def cycle_mean_A(self):
        return float(self.A(np.arange(self.p.period)).mean())

    # ------------------------------------------------------------------ dial
    def amplitude(self, t, sigma=None):
        """``sigma * L * A(t)`` -- the excess transmission gain right now.

        * sigma = 0 gives EXACTLY 0.0 at every t (NS-2.1): sigma is a factor of
          the whole product, so no rounding can leave a residue.
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
        """The true per-class gain right now, ``(r,)``."""
        return float(self.amplitude(t, sigma)) * self.send

    # ------------------------------------------------------------------ report
    def load_over_cycle(self, sigma):
        return float(self.amplitude(np.arange(self.p.period), sigma).mean())

    def swing(self, sigma):
        amp = self.amplitude(np.arange(self.p.period), sigma)
        return float(amp.max()) - float(amp.min())

    def placebo_steps(self):
        return int(np.sum(self.is_placebo(np.arange(self.p.period))))

    def report(self, sigmas=(1.0, 3.0)):
        rows = []
        for s in sigmas:
            rows.append(dict(
                sigma=float(s),
                mean_amplitude=self.load_over_cycle(s),
                peak_amplitude=float(self.amplitude(np.arange(self.p.period), s).max()),
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
        assert np.all(a[pl] == 0.0), "NS-2.5 violated: A(t) is not EXACTLY 0 when cold"
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
        return dict(rows=rows, period=int(self.p.period),
                    warm_fraction=self.p.warm_fraction, loss=self.p.loss_at_sigma1,
                    placebo_steps=int(pl.sum()),
                    anchor="STATED CALIBRATION, not a published constant: 0.14 of the "
                           "torque range at sigma=1, peak, full peer torque")

    def banner(self):
        r = self.report((1.0,))[0]
        return ("[ANT-NS] thermal driver  period=%d  placebo=%d/%d steps  "
                "L@sigma=1=%.3f of the torque range (stated calibration, NOT an anchor)  "
                "mean=%.4f  swing=%.4f  mean_preserving=%d  send=%s"
                % (self.p.period, self.placebo_steps(), self.p.period,
                   self.p.loss_at_sigma1, r["mean_amplitude"], r["swing"],
                   int(self.p.mean_preserving), np.round(self.send, 3).tolist()))
