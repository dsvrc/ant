"""ANT-NS -- the Coupling-Under-Drift form, instantiated in MAMuJoCo Ant.

CELL: (C, INVERTIBLE) -- interaction-mediated, identification AND compensation.
--------------------------------------------------------------------------------
Four legs are bolted to one torso.  Drive one and the trunk reacts; every other
leg hangs off that same trunk and feels it at its own hips.  The medium is not
something this package names into existence -- it is the robot, and it is there
in stock Ant at every severity including zero.  What the dial injects is a slow
drift in HOW STRONGLY the structure transmits, which is the design guide's
`C_i[theta(t)]` with `F_i` untouched, and a lone agent's projection of the world
stays exactly stationary because every sum runs over ``j != i``.

The disturbance is an unmodelled torque at the agent's own joints, so it lives in
exactly the space its action lives in: a correct estimate SUBTRACTS it and the
executed torque, the reward and the trajectory are the sigma = 0 ones, bit for
bit.  That is II.6's first row -- the cell where the method may claim
identification AND compensation, and where the oracle is a ceiling rather than a
competitor.

THE FOUR OBJECTS
--------------------------------------------------------------------------------
  medium    the trunk every leg is bolted to.  Element = a joint; the load path
            between two joints is public geometry from ``ant.xml``.
  operator  ``W[p,q] = recv_p * kappa(p,q) * 1[agent(p) != agent(q)]``, declared
            from the published anchors and the torso radius, zero diagonal,
            asymmetric through the receiver's own susceptibility.  Never fitted;
            checked at startup against the model's OWN inverse inertia.
  driver    drivetrain thermal state over a run: a function of the clock alone
            that reaches an agent only by scaling what its NEIGHBOURS cost it,
            and that is exactly zero over the cold half of every cycle.
  harm      an unmodelled torque on the actuators, applied below the reward, so
            Ant's reward function is byte-for-byte the host's own.

WHY THIS IS THE CHEAPEST INSTANCE TO DEFEND, AND THE BEST N-SCALING
--------------------------------------------------------------------------------
The coupling is native, so NS-1.2's "written down from the environment's own
structure" is literally the robot's drawing.  And because ``r = 3`` load paths
are independent of the partition, the SAME dial and the SAME basis run at
``2x4`` (N=2), ``4x2`` (N=4) and ``8x1`` (N=8) -- so NS-4.2's prediction that the
coordination gap rises with N is a measurement in the real host here, not only
an offline curve.

MODULES
--------------------------------------------------------------------------------
    structure.py   the robot's declared body: joints, anchors, load paths, gate 4
    driver.py      A(t), the dial, DialParams, certify() -- the four NS-2 gates
    coupling.py    the declared operator W, the r = 3 classes, the basis, the
                   geometric references
    channel.py     the per-step arithmetic: disturbance, sensor, estimator
                   (URB's core, vendored), the exact inverse, the arms
    layer.py       the severity MIXIN over MujocoMulti; hooks ``do_simulation``,
                   which is BELOW the reward
    ceiling.py     Part C -- the partition, the N-scaling, the references
    selftest.py    the offline conformance suite; no mujoco, no torch
    smoke.py       in-simulator identities; needs mujoco
    calibrate.py   the sigma ladder against a trained B0 checkpoint
    check_plumbing.py  yaml <-> dataclasses <-> layer <-> runner <-> registries
"""

from .coupling import Coupling
from .driver import DialParams, ThermalDriver
from .structure import JOINT_NAMES, PARTITIONS

__all__ = ["Coupling", "DialParams", "ThermalDriver", "JOINT_NAMES", "PARTITIONS",
           "make_ant_ns_env"]


def make_ant_ns_env(env_args, rank=0, n_threads=1):
    """Build the host with the severity layer mixed in (or the stock host when
    ``ns_on: 0``).

    Across parallel rollout threads the driver clock is DE-PHASED so a rollout
    batch is a true cycle average rather than one phase of a 20000-step cycle;
    the same rule is used for the eval envs.

    Imported lazily: ``layer`` needs mujoco, and the offline conformance suite
    must run without it.
    """
    a = dict(env_args)
    period = int(a.get("ns_period", 20000))
    if n_threads > 1:
        a["ns_phase0"] = int(a.get("ns_phase0", 0)) + int(rank * period / max(1, n_threads))
    on = a.get("ns_on", 1)
    if isinstance(on, str):
        on = on.strip().lower() in ("1", "true", "yes", "on")
    if not int(on):
        # B0: the stock host, byte for byte.  Every ns_* key is stripped so the
        # host is constructed from exactly the arguments it ships with.
        from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
        stock = {k: v for k, v in a.items() if not str(k).startswith("ns_")}
        return MujocoMulti(env_args=stock)
    from .layer import make_ant_ns_env as _mk
    return _mk(a)
