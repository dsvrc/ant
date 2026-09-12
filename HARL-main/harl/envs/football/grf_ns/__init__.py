"""GRF-NS -- the Coupling-Under-Drift form, instantiated in Google Research Football.

CELL: (C, INVERTIBLE) -- interaction-mediated, identification AND compensation.
--------------------------------------------------------------------------------
A lone player feels NOTHING, structurally: the medium is his running lane, the
load is the TEAMMATES in it, and the harm is the swerve he makes round them --
wider on a bad surface.  The swerve is a rotation of his compass heading, and
rotations of GRF's eight headings form a group with an exact inverse and no
saturation in the action space, so a correct estimate CANCELS the disturbance
(II.6, first row).  That is the demonstration cell of the classification, the
one `simple_ns` (VMAS) also fills; `urb_ns` / `road_ns` / `smac_ns` fill the
bounded cell where the method can only steer.

THE FOUR OBJECTS
--------------------------------------------------------------------------------
  medium    my running lane.  Element = the cone ahead of me, weighted by
            proximity; capacity = how much lane traffic the surface lets me
            ignore.
  operator  W[i,j] = recv_i * kappa(d_ij) * 1[j in i's lane], zero diagonal,
            asymmetric (sectors, ball carrier), spanning an order of magnitude
            (kernel).  Declared from public geometry, never fitted.
  driver    pitch condition over the match.  A function of the clock alone that
            reaches a player only by scaling what his TEAMMATES cost him; exactly
            zero for the dry half of every cycle -- the placebo.
  harm      a heading rotation enacted through the host's own action interface.
            GRF's reward is untouched byte for byte.

MODULES
--------------------------------------------------------------------------------
    actions.py      GRF's default action set, the compass group, gate 4
    driver.py       A(t), the dial, the declared constants, certify()
    coupling.py     the declared operator W, the r = 2 sectors, the basis, the
                    geometric references
    channel.py      the per-step arithmetic: disturbance, sigma-delta, sensor,
                    estimator (URB's core, vendored), the exact inverse, arms
    layer.py        the severity MIXIN over HARL's FootballEnv
    ceiling.py      Part C -- loading distribution and the (trivial) partition
    selftest.py     offline conformance suite; numpy only, no gfootball
    smoke.py        in-simulator identities; needs gfootball
    calibrate.py    the sigma ladder against a trained B0 policy
    check_plumbing.py  yaml <-> layer <-> runner <-> registry consistency
"""

from .actions import ACTION_NAMES, DIR_VEC, rotate
from .coupling import Coupling
from .driver import DialParams, PitchDriver

__all__ = ["ACTION_NAMES", "DIR_VEC", "rotate", "Coupling", "DialParams",
           "PitchDriver", "make_football_ns_env"]


def make_football_ns_env(args, rank=0, n_threads=1):
    """Build the host with the severity layer mixed in (or the stock host when
    ``ns_on: 0``).

    Across parallel rollout threads the driver clock is DE-PHASED so a rollout
    batch is a true cycle average rather than one phase of a 12000-step cycle;
    the same offset rule is used for the eval envs.

    Imported lazily: ``layer`` needs gfootball, and the offline conformance
    suite must run without it.
    """
    a = dict(args)
    period = int(a.get("ns_period", 12000))
    if n_threads > 1:
        a["ns_phase0"] = int(a.get("ns_phase0", 0)) + int(rank * period / max(1, n_threads))
    on = a.get("ns_on", 1)
    if isinstance(on, str):
        on = on.strip().lower() in ("1", "true", "yes", "on")
    if not int(on):
        # B0: the stock host, byte for byte.  gfootball rejects unknown kwargs,
        # so every ns_* key is stripped first.
        from harl.envs.football.football_env import FootballEnv
        stock = {k: v for k, v in a.items() if not str(k).startswith("ns_")}
        return FootballEnv(stock)
    from .layer import make_football_ns_env as _mk
    return _mk(a)
