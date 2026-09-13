"""The driver: the enemy line closing ranks.

Surface area shrinks when the enemy army tightens its formation -- neighbours
occupy the positions an attacker would have stood in.  Nobody on our side controls
how tightly the enemy stands; a StarCraft player names it unprompted when asked
why the same zealots got fewer hits in some fights than others.

THE ANCHOR.  In hexagonal packing a unit has six neighbour positions.  In a line
formation two of them -- its flanks -- are taken by the line itself, so a third of
the ring around it is not available to attackers:

    sigma = 1   ->   L = 1/3   at the peak of the cycle      (a line)
    sigma = 3   ->   the whole ring                            (a ball; clipped)

so sigma > 1 is a denser formation than a line and is labelled BEYOND-PHYSICAL in
every table, as the spec requires.

The waveform, the placebo half and every NS-2 gate are the reference driver's --
this class only renames it and fixes the anchor, so a cross-instance comparison
cannot be a comparison of two dial implementations.
"""

from ..smac_ns.driver import GuardDriver

#: two of the six hex-neighbour positions around a unit in a line formation
HEX_LINE_LOSS = 1.0 / 3.0


class FormationDriver(GuardDriver):
    """``A(t)`` = how tightly the enemy line stands; ``g`` = the ring left open."""

    def __init__(self, period=15000, guard_frac=0.5, loss=HEX_LINE_LOSS,
                 mean_preserving=False):
        super(FormationDriver, self).__init__(period=period, guard_frac=guard_frac,
                                              loss=loss,
                                              mean_preserving=mean_preserving)

    def banner(self):
        r = self.report((1.0,))[0]
        return ("[SMAC-SA] formation driver  period=%d  placebo=%d/%d steps  "
                "ring lost at sigma=1 peak=%.3f (hex line: 2 of 6)  "
                "capacity_removed=%.2f%%  mean_preserving=%d"
                % (self.period, self.placebo_steps(), self.period, self.loss,
                   100.0 * r["capacity_removed"], int(self.mean_preserving)))
