"""The driver: how much room the ground leaves.

Units moving together in open ground brush past each other; the same units in a
choke, on a ramp, between rocks, have to go round each other.  How much room the
ground leaves drifts as the fight moves across the map, and nobody controls it.
Asked why their stalkers bump more in some fights than others, a StarCraft player
names the terrain.

WHAT IS ANCHORED (NS-2.4, stated honestly).  StarCraft publishes no constant for
"how far a unit swerves round a teammate in a choke", so sigma = 1 is a STATED
CALIBRATION PROCEDURE, exactly as in the football instance:

    at sigma = 1, at the driver's peak, with the squad in the declared line
    (spacing 1.5, jittered by half a spacing) and every unit ordered along a
    uniformly random compass direction, the swerve reaching a unit is
    L = 0.25 rad (~14 degrees) per move order.

sigma > 1 is beyond-physical and labelled so.  The operating point is chosen by
``calibrate.py`` against a trained policy, never by hand.

The waveform, the placebo half and every NS-2 gate are the reference driver's.
"""

from ..smac_ns.driver import GuardDriver

#: the stated calibration at sigma = 1 (radians of swerve per move, reference squad)
L_SIGMA1 = 0.25


class GroundDriver(GuardDriver):
    """``A(t)`` = how cramped the ground is; the dial scales the swerve by it."""

    def __init__(self, period=25000, guard_frac=0.5, loss=L_SIGMA1,
                 mean_preserving=False):
        # the parent asserts loss < 1 for a capacity FRACTION; here L is an angle
        # scale and the capacity reading is 1/(1 + swerve), so keep it in range
        super(GroundDriver, self).__init__(period=period, guard_frac=guard_frac,
                                           loss=min(float(loss), 0.999),
                                           mean_preserving=mean_preserving)
        self.L = float(loss)

    def amp(self, t, sigma):
        """The swerve scale at step t, radians: exactly 0 at sigma 0 and in the
        placebo, non-decreasing in sigma, never negative (NS-2.1/2.2/2.3/2.5)."""
        return float(sigma) * self.L * float(self.A(t))

    def banner(self):
        return ("[SMAC-LANE] ground driver  period=%d  placebo=%d/%d steps  "
                "L=%.3f rad per move at sigma=1 (stated calibration)"
                % (self.period, self.placebo_steps(), self.period, self.L))
