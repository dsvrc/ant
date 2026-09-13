"""SMAC-LANE -- LANE SWERVE UNDER DRIFT.  SMAC's (C, invertible) cell.

A unit ordered to move with a teammate in its lane has to go round it and swerves
off the line it was ordered along; how far depends on how much room the ground
leaves, which drifts and which nobody controls.  A unit alone never swerves.

    medium    my lane: the cone within 135 degrees of my ordered move direction,
              proximity-weighted, teammates only (j != i)
    driver    how cramped the ground is; exactly zero over half of every cycle
    harm      the move order's world point is rotated away from the traffic
    inverse   pre-rotate by the estimated swerve.  Rotations of the plane have
              exact inverses and SMAC issues moves as world points, so a correct
              estimate CANCELS the disturbance -- identification AND compensation.

Why this cell for SMAC: the (C, no inverse) instances (overkill in smac_ns,
surface area in smac_sa) showed that steering over targets cannot recover what a
coupling harm takes from focus fire -- decentralized agents all read the same
cost and move together.  A compensation needs no coordination: each unit undoes
its own swerve.

Map: 3s_vs_5z (three Stalkers kiting five Zealots), where movement decides the
battle and MAPPO is known to solve it.

MODULES
    driver.py     the ground driver (reference waveform, stated calibration)
    coupling.py   the lane operator, the filtered public channels, references
    layer.py      the mixin: rotation before the tick, sensing and RLS after
    calibrate.py  the sigma ladder against a trained, frozen policy (needs SC2)
    selftest.py   offline conformance; no StarCraft II, no torch
"""


def make_smac_lane_env(args):
    """Build the host with the lane layer mixed in (StarCraft II needed)."""
    from .layer import make_smac_lane_env as _mk
    return _mk(args)


__all__ = ["make_smac_lane_env"]
