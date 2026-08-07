# RECON — implementation notes, run commands, and the debug-column key

> A separation principle for interaction-mediated non-stationarity:
> **identify centrally → filter locally → act certainty-equivalently**, with a
> conjugating compensation layer. Reference host: HAPPO (`recon`).
> Spec: `RECON_implementation_spec.md` v1.0. Benchmark: Ant-PCR at **σ = 0.45**.

```
        TRAINER (per rollout)                         AGENT i (execution, O(1)/step)
 ┌──────────────────────────────────────┐        ┌──────────────────────────────┐
 │ [ID] windowed (ρ,c) clipfit over the │        │ [F]  f_ψ(o_i, u_i(t−1)) → ℓ̂_i │
 │      stored EXECUTED joint actions   │        │ [CE] π_θ(o_i ⊕ ℓ̂_i) → a_i    │
 │ [RE] ℓ̃_i(t) = ĉ·F_ρ̂[Φ_i(u_{−i})](t)  │        │ [CP] u_i = clip(a_i − β⊙ℓ̂_i) │
 │ [DI] min_ψ Σ‖f_ψ(h_i) − ℓ̃_i‖² (own   │        └──────────────────────────────┘
 │      Adam, decoupled from every RL   │
 │      loss); host update UNCHANGED    │
 └──────────────────────────────────────┘
```

## Files

| file | role |
|---|---|
| `relabel.py` | **[RE]** + [ID]'s rolling window + the readout scan. Pure numpy. |
| `filter.py` | **[F]/[DI]** — the shared causal GRU, its Adam, the [CP] gain β. |
| `recon_mujoco.py` | the env shim: augmented space declaration + raw readout stash. No method logic. |
| `../ecl/ecl_identifier.py` | **[ID]** — reused as a library; extended with `rho_grid` + `do_scan` (both default-off, so ECL is bit-unchanged). |
| `../../../runners/on_policy_recon_runner.py` | orchestration: [CP], obs augmentation, per-iteration [ID]→[RE]→[DI], the debug trace, de-aliased eval. |
| `test_recon.py` | Stage-V0 unit tests U1–U4 (numpy+torch, no mujoco). |
| `../diag/ant_pcr045.py` | the deployable env (σ=0.45). **Already present**; copy it over `gym/envs/mujoco/ant.py` on the run machine. |

## Before anything else (run machine)

```bash
# 1. deploy the repaired benchmark env — the ONLY file ever copied there
cp harl/envs/mamujoco/diag/ant_pcr045.py $(python -c "import gym,os;print(os.path.dirname(gym.envs.mujoco.__file__))")/ant.py
# 2. V0 — must print "V0 PASS" before any 10M run is launched
python -m harl.envs.mamujoco.recon.test_recon
python -m harl.envs.mamujoco.ecl.test_ecl        # regression: ECL is bit-unchanged
```
Every process prints `[DIAG ENV] SEVERITY=0.45 ...` from the env constructor and
`[RECON] host=HAPPO ...` from the runner. If a run dir's log has no `[DIAG ENV]`
banner, it is not the benchmark and the numbers are void.

## Run commands

All arms share **one** host config (Prohibition 1). Everything below is a flag.

```bash
cd examples
CFG=../tuned_configs/mamujoco/Ant-v2-4x2/recon/config.json

# ---- U5 FIRST (seconds, offline): validates the self-supervised target ------
python -m harl.envs.mamujoco.recon.test_recon      # expect "U5 PASS" and "V0 PASS"

# ---- SMOKE GATE (~1.2M steps = one payload cycle, minutes). Never skip it. ---
python train.py --load_config ../tuned_configs/mamujoco/Ant-v2-4x2/recon/config_smoke.json
#   For label_mode: self_supervised, PASS iff in results/.../recon_smoke/.../recon_debug.csv:
#     scan_corr_at_idx > 0.25       (the configured readout responds to own torque)
#     label_r2 > 0.4 AND label_r2 is HIGHEST at the peak, NOT negative at the trough
#                                   (label_r2 now = corr of the self-sup target d̃ vs true d)
#     filter_true_r2 > 0.4 across ALL payload phases (incl. trough) — NOT just the peak
#     u_minus_a > 0 at the peak and ~0 at the trough (compensation tracks the load)
#     eval trough-slice >= peak-slice (natural order; the run #1 inversion is GONE)
#   c_corr / c_hat columns are the CENTRAL identifier, now DIAGNOSTIC ONLY (still
#   anti-correlated — that's the negative-result evidence, not a failure of this run).
#   Only then launch the 10M runs below.

# ---- Stage A headline: recon (blind, full) ------------------------- 3 seeds
python train.py --load_config $CFG --exp_name recon_s1 --seed 1
python train.py --load_config $CFG --exp_name recon_s2 --seed 2
python train.py --load_config $CFG --exp_name recon_s3 --seed 3

# ---- Stage A: conditioning-only ([CP] off; measures the layer's value) ----
python train.py --load_config $CFG --exp_name recon_condonly_s1 --seed 1 --compensate False

# ---- Stage A: oracle-filter ceiling (ℓ̂ := true d). LABELED — prints a banner.
python train.py --load_config $CFG --exp_name recon_oraclefilter_s1 --seed 1 --filter_oracle True

# ---- Stage A: blind HAPPO baseline ---------------------------------------
python train.py --load_config ../tuned_configs/mamujoco/Ant-v2-4x2/happo/config.json \
                --exp_name happo_blind_s1 --seed 1

# ---- Stage 0 / Stage A: d-oracle HAPPO ceiling. LABELED.
ANT_PCR_ORACLE=1 python train.py \
  --load_config ../tuned_configs/mamujoco/Ant-v2-4x2/happo/config.json \
  --exp_name happo_doracle_s1 --seed 1

# ---- Stage A: ECL v2 @ 0.45 (the slow-chart representative) ---------------
python train.py --load_config ../tuned_configs/mamujoco/Ant-v2-4x2/ecl/config.json \
                --exp_name ecl_s1 --seed 1

# ---- P4 stress probe: approach the frontier (2 seeds) --------------------
ANT_PCR_SEVERITY=0.55 python train.py --load_config $CFG --exp_name recon_sig055_s1 --seed 1

# ---- Ablations (after G1) -------------------------------------------------
python train.py --load_config $CFG --exp_name recon_beta_learned_s1 --seed 1 --beta_mode learned
python train.py --load_config $CFG --exp_name recon_rho06_s1 --seed 1 --rho_grid "[0.6]"   # wrong-ρ
python train.py --load_config $CFG --exp_name recon_rho095_s1 --seed 1 --rho_grid "[0.95]"
python train.py --load_config $CFG --exp_name recon_mlp_s1 --seed 1 --arch mlp             # stacked-frame fallback
```

`--severity_diag` must track `ANT_PCR_SEVERITY` (it only labels the `c_true`
diagnostic column; it never touches training). For the σ=0.55 arm add
`--severity_diag 0.55`.

**U4's env-side half** (needs mujoco, so it runs here rather than in V0): `recon`
with the filter's output pinned to zero and `compensate: false` must track plain
HAPPO. The zero-init head gives ℓ̂ ≡ 0 for free at iteration 0, so the check is:
launch `--compensate False` and confirm the first `recon_debug.csv` rows show
`u_minus_a = 0` and `lhat_rms = 0`, and that the early learning curve overlays
blind HAPPO's. Exact bit-identity is *not* expected and is not claimed: the
policy's input layer is 2 columns wider, so its orthogonal init consumes the RNG
differently. That is inherent to every obs-augmentation method.

## label_mode: the central identifier is anti-observable; use self_supervised

Runs #1–#4 established, with data, that the central identification of the severity
`c` **fails on the trained gait**: `corr(ĉ, c_true) = −0.35` (peak `ĉ` 0.42 < trough
`ĉ` 0.58), because the coordinated gait (`sumzero_frac` 0.69) makes the disturbance
`d ≈ −c·leak(a)` partly indistinguishable from an own-gain change, and the per-window
fit folds the coupling into the gain. Feeding that anti-correlated `ĉ` to `[CP]`
compensates **backwards** — worse than blind HAPPO.

So `label_mode: self_supervised` (the default) drops central `c` and trains the filter
on a **local disturbance-observer target**:

```
d̃_i(t) = ( y_i(t) − ĝ0_i · a_i(t) ) / ĝ0_i
```

`y_i` = own joint-velocity readout, `a_i` = own executed torque, `ĝ0_i` = robust global
nominal gain (median per-window `<y,a>/<a,a>`). The readout residual after own action
**rises and falls with the disturbance** (phase-correct, no anti-correlation), and the
causal filter uses the disturbance's distinguishing *lag* (`d` is the leaked history,
`a` is current) to pull `ℓ̂ ≈ d` out of it. `[CE]`/`[CP]` are unchanged. The central
path is kept as `label_mode: central` for the ablation / the paper's negative result.

**U5** validates this offline (`python -m harl.envs.mamujoco.recon.test_recon`): on a
synthetic coordinated gait it shows `ĉ` anti-/un-correlates while `d̃` tracks the true
disturbance and the distilled `ℓ̂` recovers `d` at corr > 0.5. Run it before the smoke.

Honest caveat: on the coordinated gait `d` is only *partially* observable from
proprioception, so self-sup gives phase-correct **partial** compensation. It should
clearly beat the phase-wrong central runs and blind HAPPO; whether it reaches the full
no-NS ~7k depends on how much of `d` is locally recoverable. The **d-oracle arm**
(`ANT_PCR_ORACLE=1`, below) is the definitive ceiling: if even it falls short of 7k,
7k is not recoverable at σ=0.45 by any method and the target must be reset.

## `recon_debug.csv` — the column key

The point of this file is to **localize a failure**, not to celebrate a success.
The chain is `[ID] → [RE] → [DI] → [F] → [CP]`; read the columns left to right
and blame the **first** link that goes bad. `recon_debug.log` is the same trace in
prose plus the gate warnings; `recon_eval.csv` is per-episode eval.

| column | meaning | healthy | if it's bad, the broken premise is |
|---|---|---|---|
| `c_hat` | [ID]'s raw windowed ĉ = **c_physical(const torso coupling) + c_PCR(t)** | modulates with the payload on top of a constant floor | flat constant ⇒ only the torso coupling is being read (run #3) |
| `c_floor` | trough-baseline estimate of the constant torso coupling (low pct of recent `c_hat`) | settles near `c_hat`'s trough value after ~1 cycle | — |
| `c_pcr` | `max(0, c_hat − c_floor)` — **what [RE] actually uses** | ≈ 0 at the trough, rises to a fraction of `c_true` at the peak | doesn't vanish at trough ⇒ [CP] injects harm there |
| `rho_hat` | leak rate | 0.8 (fixed) | — |
| `c_true` | `severity_diag · payload`. **diagnostic only** | oscillates over the cycle | — |
| `c_corr` | running corr(**c_pcr**, c_true) | **> 0.95** (gate G2); ≥ ~0.45 is ECL's measured floor | T4 / A3 — excitation, observability. **`nan` is a FAILURE, not a pass**: c_pcr has zero variance |
| `lock_gain`, `clip_frac`, `sumzero_frac` | clipfit quality, rail fraction, common-mode power | gain > 0.01 | T4 — the gait is unidentifiable (sum-zero or no clipping) |
| `locked` | did this window produce labels? | 1 | if 0 for long: [DI] is idle, ℓ̂ ≈ 0, recon **degenerates to HAPPO** |
| `label_r2`, `label_err` | ℓ̃ vs the env's true d | R² > 0.9 | Φ/index map, ρ̂, or Δ_c·W (= T3's ε_id) |
| `filter_mse`, `filter_mse_first` | [DI]'s MSE to ℓ̃ (after epochs−1 fits / before any) | falling | [DI] capacity or lr |
| `filter_true_r2` | ℓ̂ vs the true d — **E5's κ, measured live on-policy** | ≥ 0.5 (Stage-0 gate) | **A4** — local decodability. This is the method's real risk. |
| `filter_true_err` | rms‖ℓ̂ − d‖ = **T3's ε**; the return gap is linear in it | inside E3's r* | A4 |
| `lhat_rms` vs `ltrue_rms` | scale check | comparable | a scale mismatch means ĉ is mis-scaled (v1's failure mode) |
| `beta_hip`, `beta_ank` | the [CP] gain | 1.0 (fixed) | learned β collapsing ⇒ T5's β-inversion ⇒ near the frontier |
| `u_minus_a` | rms‖u − a‖ — how hard [CP] is pushing | > 0 once locked | 0 ⇒ [CP] is inert |
| `sat_frac` | env's `|τ+d| > 1` fraction | low | **A2** — authority exhausted ⇒ T5's frontier ⇒ *no* method can win |
| `scan_corr_at_idx` | does the **configured** readout column respond to own torque? | **> 0.25** | the index map is wrong — [ID] is regressing on a column it doesn't drive (the run #1 failure) |
| `scan_best_offset` | offset of the single best-correlating column | 0 **or** ±(a whole obs-block): own torque drives both the joint's velocity and its position, so the argmax legitimately flips between them by gait. **Not** an error as long as `scan_corr_at_idx` is high. | — |

### Post-mortem of run #1 (10M, cycle-avg 5212 — the mechanism never fired)

Worth keeping, because it is what the columns are calibrated against:

```
scan_best_offset = -2 in 45/50 scans          the readout map was 2 slots off
 -> [ID] regressed own torque on a NEIGHBOUR's joint velocity
 -> c_hat RAILED at the grid ceiling 1.2 and stayed CONSTANT all run
    (rho_hat pinned at 0.6; c_corr = nan, so the lock gate was vacuous)
 -> lock_gain 0.05 still cleared the 0.01 gate => bad labels shipped anyway
 -> label_rms 0.44 vs ltrue_rms 0.093 (label_r2 = -14)
 -> [DI]/[F] were HEALTHY (filter_mse 0.022 vs label var 0.19 = 88% of its teacher)
    -- they simply learned the wrong teacher
 -> [CP] subtracted u_minus_a = 0.42 against a true liability of 0.093:
    RECON injected ~3.7x MORE disturbance than the NS it was cancelling
 -> eval slices INVERTED: trough 5125 < peak 5500. The trough is where A(t)=0 and
    there is nothing to cancel; a constant c_hat means l~ = 1.2*x2 never vanishes
    there, so [CP] injected pure noise exactly where the env was benign.
```

It still reached 5212 only because ℓ̂ is *in the observation*, so HAPPO learned to
undo [CP] by emitting a + ℓ̂ — i.e. `recon` degenerated to blind HAPPO with wasted
capacity. Two lessons are now enforced in code: the scan aborts on a wrong map
(`_check_scan`), and a railed/constant ĉ is a loud warning rather than a `nan`
that slips through the gate.

Reading it as a decision tree:

* `locked = 0` forever → **T4**, not the filter. Nothing downstream is even running.
* `locked = 1` but `label_r2` low → **[ID]/[RE]**: ĉ scale or the Φ/ρ̂ map.
* `label_r2` high, `filter_mse` high → **[DI]**: the filter can't fit an exact target.
* `filter_mse` low, `filter_true_r2` low → **A4**: the labels are right and learned,
  but ℓ is not decodable from local history. This is the E5/κ escalation.
* everything green, return still flat → **A2/T5**: check `sat_frac`, and the
  `recon` − `conditioning-only` gap.

## Pre-registered predictions (spec Part 5 — copy into the run README before launching)

* **P1** `recon` ≈ d-oracle HAPPO ≈ `recon` oracle-filter, within CI (separation holds).
* **P2** blind HAPPO < ECL v2 < conditioning-only ≤ `recon` (the timescale ordering).
* **P3** filter true-error at peak ≤ r*, and its per-phase profile mirrors E5's R².
* **P4** removing [CP] costs little at σ=0.45 (the margin is generous) but the gap
  widens as σ → σ*; the σ=0.55 arm is the frontier-approach figure.

**Gates.** G1: cycle-average ≥ 0.9 × min(path ceiling, d-oracle arm) and ≥ blind +
3× seed std; trough-slice non-decreasing over the last 5 cycles; peak-slice ≥
0.8·B0. G2: `filter_true_err` inside (k*, r*); `c_corr` > 0.95; conditioning-only ≥
blind. G3: the measured return gap vs (ε_loc + ε_id) lies below T3's line.
A failed gate stops the line — escalate with data, do not tune past it.

## Implementation decisions worth knowing (deviations are placement, not theory)

1. **The filter lives in the runner, not the env shim.** With
   `n_rollout_threads > 1` HARL runs envs in subprocesses (`ShareSubprocVecEnv`),
   so a torch module inside the shim would be a pickled copy that never receives a
   gradient — ECL could keep its adapter env-side only because it was pure numpy.
   The spec's §4 wording ("[CP] … inside the RECON env wrapper") is therefore
   implemented as: shim declares dims + stashes the readout; runner runs [F] and
   applies [CP] before `envs.step`. **The information sets are unchanged**: [F]
   consumes only `(o_i, u_i(t−1))`, [CP] only `ℓ̂_i`, and the batch axis never
   mixes agents, so N filters batched in one process is arithmetically N
   decentralized filters. Decentralization is a property of the *inputs*, not of
   the process table.
2. **The buffer stores `a`, the env receives `u`.** The PPO ratio is evaluated on
   the policy's own sample; [ID]/[RE] and the env see the executed action (spec
   §2.1 ext 2). This required overriding `run()` rather than adding a hook to the
   shared on-policy base runner.
3. **`o_i` is what HARL provides — the per-timestep-normalized vector.** Note
   `MujocoMulti.get_obs()` ignores `agent_obsk` and hands every agent the *full*
   state plus a one-hot id, then normalizes the whole vector by its own mean/std.
   That normalization looks like it would destroy the torque-scale information the
   filter needs — but it does not: the one-hot block is a known constant pattern,
   so `(1−μ_t)/σ_t` and `(0−μ_t)/σ_t` appear in the observation and (μ_t, σ_t) are
   exactly recoverable from it. The map is invertible; the normalization is a
   conditioning nuisance the GRU must learn through, not an information loss.
   `filter_true_r2` measures whether it managed.
4. **ℓ̂ is appended *after* the shim's parent normalizes** (ECL's precedent), so ℓ̂
   reaches the policy in torque units — which is what [CP] needs. Contrast the
   `ANT_PCR_ORACLE` arm, which appends `d` *inside* `_get_obs()` and therefore
   hands the policy `(d − μ_t)/σ_t`; that is a real handicap on the ceiling arm
   and it is the env's choice, not RECON's.
5. **Zero-init output head.** ℓ̂ ≡ 0 at iteration 0, so [CP] starts as the identity
   and the compensation fades in exactly as fast as the filter earns it. No
   warm-up knob, and a mis-locked early ĉ cannot inject a large action offset.
6. **[ID] refreshes every `W / episode_length` iterations** (10 at the tuned
   config) — ECL's "every W new steps" cadence, and exactly the window over which
   [RE] treats ĉ as constant. Between refreshes ĉ is held; that staleness *is*
   T3's Δ_c·W term, and `label_r2` measures it.
7. **[RE]'s leak state carries across iterations** rather than restarting each
   rollout, so labels are exact from step 0 (the on-policy buffer is contiguous —
   the reason HAPPO is the reference host). [ID] keeps ECL's 25-step warm-up
   margin instead, unchanged.
8. **`filter.adam_lr` / `filter.grad_clip`, not `lr` / `max_grad_norm`.** HARL's
   `update_args` rewrites a CLI override into *every* nested dict by key name, so
   a `--lr` meant for the host would silently retune the filter.
9. **Learned β** (`beta_mode: learned`, ablation (v)) is fit on the cancellation
   objective `min_β E‖ℓ̃ − β⊙ℓ̂‖²` — the least-squares gain that best cancels the
   liability given the filter's own error. This is the shrinkage factor T3
   predicts (β* → 1 as the filter's MSE → 0) and the mechanism T5 blames for the
   measured β-inversion. It uses only trainer-side quantities and needs no host
   change; a gradient through the PPO loss does not exist for β, because the env
   is not differentiable and the stored action is `a`.
10. **The ctrl-cost caveat, stated honestly.** Ant charges `0.5‖·‖²` on the torque
    it is *commanded*, so a compensating agent pays for `u`, not for `a`. This is
    not an env edit — it is the same env, same reward, and the blind agent
    likewise pays for what it commands. But it means T2's corollary "the optimal
    value of the compensated game equals the stationary optimum" holds for the
    *dynamics*, not exactly for the return. The d-oracle ceiling arm has the same
    property, which is why G1 compares against it rather than against B0.

## Not done here (per the spec's build order)

`recon_hasac` (off-policy: labels at insert + the v2 retag cadence) and the SMAC /
SMACv2 instantiations (Part 6) come **after** Ant passes G1. Build order is a
prohibition, not a preference.
