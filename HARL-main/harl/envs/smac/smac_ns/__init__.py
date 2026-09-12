"""SMAC-NS -- the Coupling-Under-Drift form, instantiated naturally in SMAC.

CELL: (C, no inverse) -- interaction-mediated, steering only.
--------------------------------------------------------------------------------
A lone agent feels NOTHING, structurally: the medium is the enemy line's damage
absorption, the harm is overkill, and a solitary unit cannot overkill.  There is
no inverse -- you cannot un-waste a shot any more than you can subtract minutes
off a congested road -- so the method may claim IDENTIFICATION AND STEERING ONLY,
and the margin is capped by the coordination gap.  That is the spec's bounded
cell, and conflating it with the compensation cell is the one thing the spec says
is not publishable.

Why steering and not compensation, in one line: SMAC's action space is
``no-op / stop / move x4 / attack[e]``, and its only continuous magnitude is the
move stride, which was measured to be inert on 3s5z (a movement-competent control
scores 0.83x a focus-fire one there, and a perfect oracle on a stride channel
gained 1.03).  So the honest channel is a shift over the K attack options.

THE FOUR OBJECTS
--------------------------------------------------------------------------------
  medium    the enemy line's absorbable damage.  Element = an enemy unit;
            capacity = its PUBLISHED effective hit points.
  operator  incidence over capacity, weighted by published per-unit damage.
            Zero-diagonal, asymmetric (0.545 measured), never fitted.
  driver    the guard cycle.  Shrinks absorption, never adds to the loss; reaches
            exact zero over half of every episode, which is the placebo.
  harm      overkill, handed back to the engine as restored hit points.  The
            reward function is untouched byte for byte.

MODULES
--------------------------------------------------------------------------------
    driver.py      A(t), the dial, and certify() -- the four NS-2 gates
    coupling.py    the declared operator W, the r classes, the basis
    layer.py       the severity MIXIN: below the method, above the host
    ceiling.py     PART C -- the coordination gap, with NO training
    selftest.py    the offline conformance suite; no StarCraft II, no torch
    pact1_core.py  VERBATIM vendored copy of URB's pact1/core.py.  Do not edit.
"""

from .coupling import Coupling
from .driver import GuardDriver


def make_smac_ns_env(args):
    """Build the host with the severity layer mixed in.

    Imported lazily: ``layer`` needs StarCraft II's protobufs, and I.7's
    conformance suite must run without them.
    """
    from .layer import make_smac_ns_env as _mk
    return _mk(args)


__all__ = ["Coupling", "GuardDriver", "make_smac_ns_env"]
