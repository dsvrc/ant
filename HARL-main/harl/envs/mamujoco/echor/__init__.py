"""ECHO-R: driver-estimation + conditioning layer for category-C non-stationary MARL.

This package contains the framework-agnostic ``EchoRAdapter`` (pure numpy, spec
Part 3/4) and the HARL/MAMuJoCo integration shim ``EchoRMujocoMulti`` (spec Part
6.1 / Part 4 "env-wrapper slot").

See ``ECHO-R_implementation_spec.md`` for the full specification.
"""

from harl.envs.mamujoco.echor.echor_adapter import EchoRAdapter

__all__ = ["EchoRAdapter"]
