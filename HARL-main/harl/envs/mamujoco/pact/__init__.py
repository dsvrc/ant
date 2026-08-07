"""PACT — Peer-Action Compensation with a Trained gain.

The deployable method that reaches the O-MAX (O1) ceiling WITHOUT reading the
env's privileged disturbance. It replaces O1's true ``pcr_d_next`` with an
*arithmetically computed* coupling waveform ``x2 = leak_ρ(Σ_{j≠i} τ_j)`` (exact
from the peers' executed torques — a declared one-step-delayed communication of
2 scalars/agent), and compensates with a single learned scalar gain
``β``:  ``u_i = clip(a_i − β·x2_i)``.

Only ``β`` is learned (from return, as one extra bounded action dimension); the
waveform is exact by construction. See ``pact_mujoco.py`` for the wrapper and
``harl/runners/on_policy_pact_runner.py`` for the diagnostic logging + the one
hard (arithmetic, gait-independent) exactness gate.
"""
