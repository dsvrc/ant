"""Phase-1 (PACT pipeline) sigma-star certification for SMACv2-CWD.

Certify solvability and find sigma-star for the Concussion-Coupled Wake
Displacement non-stationarity: turn off learning, hand a scripted controller the
true hidden shove ``d_i``, sweep severity, and measure the largest severity at
which compensation still recovers the undisturbed baseline B0.

Pieces
------
* ``probe_env.SMACv2ProbeEnv`` -- the CWD env + a scripted, env-side compensation
  controller (discrete nearest-cardinal re-aim, or idealized continuous re-aim)
  and a freeze knob.  No learning, no gradients.
* ``sigma_star`` -- the sweep driver: load a trained blind baseline, roll it
  through the probe over a severity x gain grid, take max over gain per severity,
  and report sigma-star (+ the best-gain crossover, residual, saturation, falls).
* ``test_probe`` -- pure-numpy unit tests of the controller arithmetic (no SC2).

See ``README.md`` for the runbook and ``CWD_non_stationarity.md`` (one dir up) for
the mechanism this certifies.
"""
