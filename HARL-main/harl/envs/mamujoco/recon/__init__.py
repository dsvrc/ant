"""RECON — a separation principle for interaction-mediated non-stationarity.

    [ID] identify c centrally  ->  [RE] reconstruct ℓ̃ in hindsight
    ->  [DI] distill into a local causal filter  ->  [CE] condition the policy
    ->  [CP] conjugate the channel back to the stationary env

Modules:
  ``relabel.py``      [RE] + [ID]'s window  (pure numpy, unit-testable)
  ``filter.py``       [F] + [DI]            (torch; the only learned addition)
  ``recon_mujoco.py`` the env shim          (spaces + readout stash; no method)
The orchestration ([CP], obs augmentation, the debug trace) is in
``harl/runners/on_policy_recon_runner.py``. [ID] itself is
``harl/envs/mamujoco/ecl/ecl_identifier.py``, reused as a library.
"""
