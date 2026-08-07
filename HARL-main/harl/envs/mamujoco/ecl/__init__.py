"""ECL — Equilibrium Continuation Learning for category-C non-stationary MARL.

Agent side: the local harm-envelope adapter (pure numpy, spec Part 4 / 3.5) and
the HARL/MAMuJoCo env shim (envelope obs augmentation; NO probe, action
untouched). Trainer side lives in the runner/buffer (identifier + localizer +
anchor). See ``ECL_implementation_spec.md``.
"""

from harl.envs.mamujoco.ecl.envelope_adapter import EnvelopeAdapter

__all__ = ["EnvelopeAdapter"]
