# PACT — Peer-Action Compensation with a Trained gain

PACT reaches the O-MAX **O1 ceiling (6104)** *without* reading the env's
privileged disturbance, by **computing** what the oracle was told instead of
estimating it.

O1's only privileged input is `d_i = c(t)·x2_i(t)`, `x2_i = leak_ρ(Σ_{j≠i} τ_j)`.
`x2_i` is **pure arithmetic** over the peers' executed torques (a declared,
one-step-delayed communication of 2 scalars/agent); only the slow global scalar
`c(t)=A(t)·σ` is genuinely hidden. So PACT compensates with the exact waveform and
a learned gain:

```
u_i = clip(a_i − β_i · x2_i,  −1, +1)         β_i = β_max · sigmoid(w_i)   [direct mode]
```

`β_i` is set by **one extra bounded action dimension** per agent (`w_i`), learned
by PPO from the return. Perfect play is `β_i(t) ≈ c(t)`. Nothing is estimated; the
waveform is exact, and with `β≡0` the executed torque is `clip(a)` — plain HAPPO
(**floor property**: no configuration craters below blind).

**v2 default (direct-β + recurrent) — why:** the v1 headline (global integrator
`β←clip(β+δ·mean(w),0,β_max)`, memoryless policy) reached only 4442 (below blind).
The trace diagnosed it precisely: the arithmetic gate was perfect (`cos=1.0`) and β
*transiently* reached 0.36 at the peak by 5M — but then **collapsed to ~0**. Cause:
the optimal β is phase-dependent (~0.44 at peak, **0 at trough**, where `x2≈0.2`
but `c=0` so any β>0 self-injects harm), the memoryless policy can't sense the
hidden phase, and a *single global* β can't be both — the definite trough penalty
drives it to zero. Fix: (i) a **recurrent policy** so it can estimate `c` (within
an episode `c` is ~constant — an easy scalar), and (ii) **direct per-agent β** so
each agent sets β phase-appropriately from the RNN, starting at `β_max/2` (partial
compensation from step 1 — breaks the fall-fast/short-episode trap a β=0 start
hits) with no global integrator to collapse.

## Why this is not any of the methods that failed

| Post-mortem failure (RECON / ECL) | PACT status |
|---|---|
| teacher label mis-scaled; identifier anti-correlated with c; scalar-gain plant model false | **no teacher, no identifier, no plant model** — `x2` is arithmetic, `β` is learned from return and bounded ≤ `β_max` |
| gait-dependent readout-scan crashed good runs | **no readout, no scan**; the one hard gate is arithmetic (gait-independent) |
| policy fought a wrong compensation | escape valve *is* the knob: `β→0` recovers blind exactly |
| whitening confound (oracle obs mangled by per-step norm) | all PACT features appended **post-normalization, native units** |

## Run

```bash
# The method (10M, seed 1). Deploy ant_pcr045.py (σ=0.45) over gym's ant.py first.
PYTHONPATH=$PWD python examples/train.py \
  --load_config tuned_configs/mamujoco/Ant-v2-4x2/pact/config.json --exp_name PACT --seed 1
```

Comparators are already on disk — **blind HAPPO ≈ 5000**, **O1 = 6104**. No new
baseline is needed. The headline number is the eval cycle-average in
`progress.txt` (the eval envs are C4 de-aliased via `pcr_eval_dephase`, so each
row is a true cycle-average, exactly as O1's was).

**Unit tests first** (pure numpy, seconds, no simulator):
```bash
python -m harl.envs.mamujoco.pact.test_pact      # expect: ALL PACT UNIT TESTS PASSED
```

## Acceptance & gates

- **Acceptance:** ≥ **5500** cycle-average at 10M, seed 1 (≥45% of the O1−blind
  gap); then confirm with 2 more seeds. Report the three-bar figure blind /
  PACT / O1.
- **The one hard gate (arithmetic, gait-independent):**
  `cos_x2_dnext > 0.999`, checked once past 200k steps. It is the mean **per-step
  cosine similarity** between the `x2` and `pcr_d_next` 8-vectors on payload>0.3
  steps — ~1 when the peer-action recursion is wired right, ~0 under any index /
  reset / one-step-timing bug, and invariant to how `c` drifts across steps.
  (A Pearson corr *pooled* across a cycle would read only ~0.95 from the c-fan
  even when every point is exactly `d=c·x2` — so cosine, not pooled corr, is the
  correct metric.) A fact independent of the gait, hence safe to abort on (unlike
  the readout scans that burned earlier methods). Default `pact_cfg.gate_abort:
  true`; set `false` to warn-only.
- **Soft read (the whole ballgame — does β phase-track?):** in `pact_debug.csv`,
  `beta` should rise toward `c_true` at the peak and → 0 at the trough
  (`beta_peak` high, `beta_trough` → 0). If the gate is green but the return
  stalls near blind, β is **not** phase-tracking — see the fallback ladder below.

## `pact_debug.csv` column key (one row / rollout)

`env_step, rollout, payload_mean, beta, c_true (=payload·σ, diagnostic),
beta_minus_c, cos_x2_dnext (the gate; per-step cosine), x2_absmean, resid_absmean
(=mean|pcr_d_next − β·x2|), u_clip_frac, dbeta_absmean, r_total, r_forward,
r_ctrl, r_contact, r_survive, ep_len_mean, actor_std, beta_peak, beta_trough,
r_total_peak, r_total_trough`.

The two paper figures come straight from the per-rollout rows: **β(t) overlaid on
c_true(t)** (the whole cycle is swept across the 2500 training rollouts, so this is
the clean tracking figure — read it from `beta`/`c_true`, not the thin peak/trough
bins), and **resid vs blind |d|** (the disturbance removed).

## Pre-registered predictions (write into the run log before launching)

- **P1** `cos_x2_dnext ≈ 1.0000` by 200k (arithmetic exactness — near-certain).
- **P2** blind (5000) < PACT ≤ O1 (6104); PACT ≥ 5500 if β tracks even coarsely.
- **P3** `beta_peak` climbs toward `c_true≈0.45` while `beta_trough → 0`.
- **P4** `resid_absmean` at the peak drops to ≈ ⅓ of the blind peak |d| (≈0.095)
  when β tracks within ±0.15 of c.

## If the v2 default still stalls (fallback ladder, in order)

The v2 default (`beta_mode: direct` + `use_recurrent_policy: true`) directly
attacks the diagnosed cause (phase-blindness + global collapse). If the eval curve
still plateaus with `cos_x2_dnext≈1` but β not tracking (`beta_peak` not climbing
toward ~0.44, `beta_trough` not → 0):

1. **Stronger smoothing / slower β** if β is jittery: raise `pact_cfg.beta_ema`
   (0.9 → 0.95) so the compensation is steadier for the base gait to exploit.
2. **Privileged critic (CTDE-legal analysis arm).** Give the *critic only* the
   true payload (never the actor) so value estimation is clean and credit
   assignment for β accelerates — the honest analogue of O-MAX's O3. Label it as
   an analysis arm, not the headline.
3. **More steps.** Recurrent policies converge slower; if still rising at 10M,
   extend `num_env_steps` (O1 was still rising at 10M too).

Ablations for the paper: `beta_mode: integrator` (reproduces the v1 collapse —
the negative result that motivates direct+recurrent); memoryless + direct (isolates
the recurrent contribution); `pact_cfg.beta_max: 0` (≡ blind, the floor check);
`beta_driver` mean vs agent0 (integrator mode); ρ robustness (`pact_cfg.rho:
0.7 / 0.9`).

## Honesty & scope (verbatim into the paper)

- PACT is a **communication method**: peers share their executed torques (O(N)
  scalars/step, one-step delay). This is **declared**, and justified by the
  measured impossibility that precedes it — central *and* local *passive*
  identification both fail on the trained coordinated gait (the RECON
  post-mortem's Wall A). The message is not learned; it is the provably minimal
  one that makes the coupling waveform computable exactly.
- No env `pcr_*` info key enters the control path; they appear only in logging
  columns and the arithmetic gate.
- **Generality (one theory, many envs):** every category-C NS admits the same
  construction — broadcast own exertion (the Φ-coordinate), run the known leak to
  get the exact waveform, apply the channel's compensation/re-aim with one learned
  bounded gain. Ant is the reference; SMAC/SMACv2 only redefine Φ and the
  compensation channel.

## Config keys (`env_args.pact_cfg`)

`beta_mode` (**`direct`** default | `integrator`), `beta_max` (0.6),
`beta_ema` (0.9 — direct-mode smoothing), `rho` (0.8, env structural),
`ema_hl` (200), `delta` (0.01 — integrator speed), `beta_driver`
(`mean`|`agent0` — integrator agg), `gate_abort` (true), `gate_min_corr` (0.999),
`gate_after_steps` (200000). Pair `beta_mode: direct` with
`algo_args.model.use_recurrent_policy: true` (the v2 default config already does).
The `std_floor` contingency lives in `algo_args.model.std_floor` (0.0 = inert).
