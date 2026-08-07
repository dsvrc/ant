# SMACv2-CWD — Phase 2 (PACT), the discrete soft variant

Peer-Action Compensation with a Trained gain, for the Concussion-Coupled Wake
Displacement NS. Built **after** Phase 1 certified the frontier
(`harl/envs/smacv2/phase1`). Follows `~/Desktop/PACT_complete_pipeline.md` Part III /
Part V / Part IX steps 13–17.

---

## Why severity 1.5 (the Phase-1 verdict)

Phase 1 (peak-frozen, scripted, privileged) found:

- **Solvable at every tested σ.** The *continuous* re-aim certificate (perfect
  cancellation) recovered B0 win-rate at all σ up to 2.0 — CWD's shove is a pure
  translation, invertible, and `_DCAP` keeps the inverse from ever saturating. There
  is **no actuator-saturation σ\*** for CWD; the only limit is discrete expressibility.
- **Return is the wrong yardstick for SMAC** (shaped damage-reward stays high even in
  losses → the printed `σ*(return)=2.0` is a vacuous "the line never fell"). **Read
  win-rate.**
- On **win-rate**, the NS collapses the blind policy only at **σ ≥ 1.5**
  (0.575 → 0.20), and perfect cancellation fully recovers it (→ 0.675). σ ≤ 1.0 shows
  no gap (all sampling noise).

So **σ = 1.5** is the operating point: the largest, cleanest blind→recoverable gap.
(σ = 1.25 is a gentler alternative if 1.5's blind win ≈ 0.20 proves too harsh to learn
through — one number in the config.)

**Ceiling to beat / approach** (from Phase-1, peak, win-rate): blind ≈ **0.20**,
compensation ceiling (perfect continuous re-aim) ≈ **0.675 ≈ B0**. Note Phase-1 is the
*frozen-peak* slice; Phase-2 runs the *live* ramp, so its cycle-average sits above the
peak-only numbers — compare arms to each other, and to the trained oracle, not
literally to 0.20/0.675.

---

## The three declarations (Part V) for CWD

| # | Declaration | CWD instance |
|---|---|---|
| 1 | **Exertion Φ_i** | `S_i = Σ_{j≠i, firing} w_ij·u_ij` — range-weighted (`w=1/(1+d/R)`, R=5) radial push from **firing** peers (action ≥ n_no_attack). Shared message = peers' firing bits (1-step delayed); relative positions come from own obs. |
| 2 | **Leak / coupling** | `x2_i ← ρ·x2_i + (1−ρ)·S_i`, ρ=0.5 (impulsive), reset each episode. |
| 3 | **Harm channel + inverse** | additive move-**target** drift → re-aim by −β·x2. Actions are **discrete**, so use the **soft variant**: append `[x2_i, |x2_i|]` to obs and let the recurrent policy pick the corrected discrete action (Part V). |

## The mechanism (all env-side; host RL untouched)

The obs-augmentation is env-side (`env_args.cwd_pact=1` in `smacv2_env.py`, mirroring
the oracle append), so **training is bit-identical to recurrent HAPPO**. `--algo pact`
adds only a thin runner (`on_policy_pact_smac_runner.py`) that logs `pact_debug.csv`
and enforces the cosine gate — the actor/critic/loss are unchanged HAPPO:

- The env maintains the exact waveform `x2_i` (the CWD accumulator **minus** the hidden
  scalar `c = A(t)·σ`; the true shove is `d_i = c·x2_i` on the unsaturated set).
- **Actor obs** `o_i ⊕ [x2_i (2), |x2_i| (1)]` — agent i's OWN waveform (decentralized).
- **Critic state** `s ⊕ [stacked x2 (2N)]`, and with `cwd_pact_ctde=1` also `⊕ [A(t)]`
  (the true driver, **critic-only, training-only** — standard CTDE).
- **One-step timing:** the obs returned after step *t* carries `x2(t+1)`, which the
  policy consumes to act at *t+1*, whose move is shoved by `d(t+1)=c·x2(t+1)` — the
  policy sees the waveform *before* committing (verified by the gate).
- **Floor property:** ignore the appended dims ⇒ exactly the blind policy. No estimator
  in the control path.
- **Recurrence is required** (Part III.3): the policy must infer the within-episode-
  almost-constant scalar `c` (the hidden phase) from memory. The config has it ON.

---

## Run (server)

`pact` is a first-class algorithm (`--algo pact`, dispatched by env to the SMAC
soft-variant runner); blind is plain `--algo happo`. Every NS arm sets severity
**explicitly** — a naked run / the shipped `happo/config.json` is STOCK SMACv2
(severity 0). `--exp_name` / `--seed` override the config like any HARL run.

```bash
cd /path/to/HARL-main

# 0) Arithmetic certificate (no SC2). Must end "ALL TESTS PASSED".
python -m harl.envs.smacv2.pact.test_pact

# 1) Real-env wiring gate (few min). LOW severity => cosine must be ~1.000.
python -m harl.envs.smacv2.pact.gate --map_name protoss_5_vs_5 --severity 0.5 --episodes 6

# 2) The arms (host hyperparameters identical across all; ~10M steps each).
#    stationary reference (STOCK, severity 0):
python examples/train.py --load_config tuned_configs/smacv2/protoss_5_vs_5/happo/config.json      --exp_name stationary --seed 1
#    blind @ severity 1.5 (the lower reference PACT must beat):
python examples/train.py --load_config tuned_configs/smacv2/protoss_5_vs_5/happo/cwd_blind.json   --exp_name blind      --seed 1
#    PACT (fully decentralized):
python examples/train.py --load_config tuned_configs/smacv2/protoss_5_vs_5/pact/config.json       --exp_name PACT       --seed 1
#    PACT + CTDE critic (true driver A(t) in the critic only):
python examples/train.py --load_config tuned_configs/smacv2/protoss_5_vs_5/pact/config_ctde.json  --exp_name PACT_CTDE  --seed 1
#    oracle (true d in obs) — learner-given-truth reference:
python examples/train.py --load_config tuned_configs/smacv2/protoss_5_vs_5/happo/cwd_oracle.json  --exp_name oracle     --seed 1
```

The `pact` runner writes `pact_debug.csv` (per-rollout `cos_gate`, `x2load`, ...) and
enforces the cosine gate (hard-abort by default; `pact_cfg.gate_abort:false` warns).
Read `results/smacv2/protoss_5_vs_5/<algo>/<exp>/<run>/progress.txt` (win-rate column)
or TensorBoard `eval_win_rate`. Expected ordering (cycle-average win-rate):

```
blind  <  PACT (decentralized)  ≤  PACT+CTDE  ≲  oracle  ≈  stationary B0
```

The **gap between decentralized PACT and the oracle** is the phase-tracking frontier
for CWD (Part III.4): expect PACT to be a robust win over blind from the exact
waveform; CTDE + recurrence chase the last stretch.

---

## Gates & what to watch (Part VI)

- **Cosine gate (the one hard gate):** `gate.py` mean per-step cosine(x2, d) must be
  ~1.0 at low severity. A drop = a leak-wiring bug (index order / reset / timing), not
  a frontier — fix before trusting any training curve. (At σ=1.5 it dips a hair on the
  ~2% saturated steps — the honest leak, in `test_pact.py::Sat`.)
- **Floor/safety:** eval < blind is structurally impossible if wired right — suspect a
  bug, not the method.
- **β-tracking is implicit here** (soft variant: the policy IS the gain). Watch the
  eval win-rate oscillate with the bombardment cycle for blind and flatten upward for
  PACT.

## Caveats (honest, and worth a follow-up)

- **Eval aliasing.** The persistent clock is phase-synced across eval threads, so a
  40-episode eval is a phase snapshot, not a clean cycle-average. For the paper number,
  eval over many more episodes (several full 3000-step cycles) or stratify clocks.
- **Weak/noisy B0.** Phase-1 B0 was win 0.575 with ±0.08 noise. A stronger, converged
  B0 (and win-rate-headlined Phase-1 at ~100 eps/cell) sharpens every comparison.
- **Move-direction map.** `phase1/probe_env._DIR` (N/S/E/W ↔ x,y) is the standard SMAC
  convention but was not verifiable offline; the cosine gate here does **not** depend
  on it (it's obs-side), but the Phase-1 discrete ceiling did.
- **`cwd_severity` default fix.** The env now honours `SMACV2_CWD_SEVERITY` and defaults
  to 0.5 (the doc value); the old code hard-coded 5 and ignored the env-var.
