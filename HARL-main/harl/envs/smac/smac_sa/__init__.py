"""SMAC-SA -- SURFACE AREA UNDER DRIFT.  The Coupling-Under-Drift form in SMAC.

CELL: (C, no inverse) -- interaction-mediated, identification and steering only.
--------------------------------------------------------------------------------
    medium    the ring of contact positions around each enemy unit
    operator  footprint / capacity between melee attackers on the same target,
              from published unit radii; zero diagonal, never fitted
    driver    the enemy line closing ranks; sigma = 1 is a line formation
              (2 of 6 hex neighbours = a third of the ring), exactly zero over
              half of every cycle (the placebo)
    harm      an attack order that finds no free slot is executed as STOP.  No
              hit points are ever written; the reward function is untouched.

A lone attacker always finds a slot, at any severity -- category C from the
j != i, not from a small number.  This replaces the overkill instance in
``smac_ns``, whose harm could never reward spreading (see coupling.py).

MODULES
--------------------------------------------------------------------------------
    driver.py      the formation driver (the reference waveform, re-anchored)
    coupling.py    capacities, the operator, the BPR share basis
    layer.py       the severity MIXIN: the slot lottery before the tick,
                   sensing / estimation / prediction after it
    story.py       offline Lanchester battle through the real layer
    selftest.py    the offline conformance suite; no StarCraft II, no torch
Shared with smac_ns: pact1_core.py (verbatim URB), channel.py (the policy's
shift), toy.py (the engine-order stand-in), check_actor.py (the torch actor).
"""


def make_smac_sa_env(args):
    """Build the host with the surface-area layer mixed in (StarCraft II needed)."""
    from .layer import make_smac_sa_env as _mk
    return _mk(args)


__all__ = ["make_smac_sa_env"]
