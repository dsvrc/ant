"""Phase-2 (PACT pipeline) for SMACv2-CWD: the discrete soft variant.

Certified in Phase 1 (``harl/envs/smacv2/phase1``): CWD's move-target shove is a
pure translation, fully invertible at every severity; on ``protoss_5_vs_5`` it
meaningfully collapses the blind policy's win-rate only at severity >= ~1.5, where
perfect (continuous) cancellation fully recovers it.  So we build PACT at
**severity 1.5**.

Because SMACv2 actions are discrete, a continuous re-aim isn't expressible on the
action, so PACT here is the **soft variant** (pipeline Part V): the env computes the
EXACT disturbance waveform

    x2_i(t+1) = rho * x2_i(t) + (1 - rho) * S_i(t)         (rho = 0.5)

-- the same leaky accumulator the CWD env runs, MINUS the one hidden scalar
c(t) = A(t) * severity (so the true shove is d_i = c * x2_i on the unsaturated set)
-- and appends ``[x2_i, |x2_i|]`` to agent i's observation (and the stacked ``x2``,
plus optionally the true driver ``A(t)`` for a CTDE critic, to the centralized
state).  A recurrent policy then learns to pre-empt the shove and to infer the
within-episode-almost-constant scalar ``c`` itself.  ``x2_i`` is decentralizable
arithmetic (peers' firing bits, one-step delayed, + their relative positions); the
true ``d`` is never exposed (that is the oracle).

This lives INSIDE ``harl/envs/smacv2/smacv2_env.py`` (gated by ``cwd_pact``),
mirroring the oracle obs-append, so **stock recurrent HAPPO/MAPPO trains on it with
no runner or algorithm change**.  Floor property: ignore the appended dims and it is
exactly the blind policy.

Pieces here: ``test_pact.py`` (pure-numpy arithmetic certificate -- no SC2),
``gate.py`` (the real-env per-step cosine gate, pipeline §VI), ``README.md`` (runbook).
"""
