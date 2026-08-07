"""Phase-2 (PACT) for SMAC v1 on the Coupled Weapon Overheat (CWO) non-stationarity.

CWO (in ``StarCraft2_Env.py``) is a category-C, dynamics-only NS with ONE knob
(``SEVERITY``): units share a weapon power/cooling bus, so the more the OTHERS fire the
more YOUR weapon overheats and a commanded shot is DROPPED.  The shared load

    x2_i <- RHO*x2_i + (1-RHO) * (sum_{j!=i} fire_j)/(N-1)        (RHO=0.85, in [0,1])

is a PURE peer-action scalar; the drop probability is ell_i = min(A(t)*SEVERITY*x2_i,
LMAX).  PACT appends the COMPUTED x2_i to the policy's obs (``env_args.snd_pact=1``), so
a recurrent HAPPO policy learns to STAGGER firing on the shared load -- the
tragedy-of-commons recovery a blind agent cannot do.  `--algo pact` trains stock
recurrent HAPPO on the augmented obs; the runner (``on_policy_pact_smac_runner.py``)
only adds ``pact_debug.csv`` (rich CWO + coordination telemetry).

Pieces: ``test_pact.py`` (pure-numpy arithmetic certificate), ``calibrate.py`` (pick
SEVERITY on the real env), ``README.md`` (runbook + the debug-file guide).
"""
