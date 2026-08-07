# SMACv2-CWD — Phase-1 (σ\*) certification runbook

This is Phase 1 of the PACT pipeline (`~/Desktop/PACT_complete_pipeline.md`) applied
to the SMACv2 non-stationarity **Concussion-Coupled Wake Displacement (CWD)** (see
`../CWD_non_stationarity.md`). It answers one question **before** any method is built:

> **What is the largest severity σ at which a *privileged, scripted* controller —
> handed the true hidden shove `d_i` — still holds the frozen-peak return to ≥ 90 %
> of the undisturbed baseline B0?** That σ is **σ\***. A learner can at best match a
> perfect-information controller, so if the scripted controller fails at σ, no method
> can succeed at σ. Certify existence before spending compute.

Everything here is **eval-only, no learning, no gradients**, and reuses one SC2
process for the whole sweep. It runs in minutes once you have a trained B0.

---

## The three declarations for CWD (pipeline Part V)

| # | Declaration | CWD answer |
|---|---|---|
| 1 | **Exertion `Φ`** (what of the *others* feeds the liability) | the range-weighted radial push from **firing** peers, `S_i = Σ_{j≠i,firing} w_ij·û_ij` |
| 2 | **Leak / coupling** | impulsive leaky integrator, `ρ=0.5`; `d_i ← ρ·d_i + (1−ρ)·A·σ·S_i`, capped at `_DCAP=2.0`/axis |
| 3 | **Harm channel `g` + inverse** | **additive move-target drift** `delivered = commanded + d_i`; inverse = **re-aim** (subtract `β·d`) |

Because the harm is a pure translation of the move target, the inverse is exact —
**with continuous re-aim (`β=1`) the delivered target is byte-identical to stock
SMACv2 at every σ**, and `_DCAP` guarantees it never saturates (design doc §5.1/§5.4).
So for CWD the *solvability-in-principle* frontier is unbounded; **σ\* is entirely a
property of the discrete action set** — a unit can only pick from {stop, N, S, E, W},
so it cancels the shove only coarsely. That is exactly the recoverability regime the
design doc §5.4 describes ("the oracle chooses the least-bad cardinal move").

### The two controllers this harness runs

- **Discrete re-aim — the σ\* headline.** Keep action *selection* discrete: replace a
  commanded cardinal MOVE with the available move (or STOP) whose post-shove
  displacement best matches intent. `β=0` ⇒ blind; `β=1` ⇒ full re-aim. The bounded
  resource is discreteness: along-intent shove cannot be undone (you cannot "move
  less"), so a residual survives and grows with `|d|` until the `_DCAP` cap.
- **Continuous re-aim @ β=1 — the invertibility / transparency certificate.** Subtract
  `d` from the world target directly. Should recover B0 at **every** σ. If it does
  not, there is a wiring/units bug — fix that before believing any discrete number.

---

## Files

| File | Role |
|---|---|
| `../smacv2_env.py` | CWD env, now with per-instance knobs (`cwd_severity/oracle/freeze`), a **freeze** knob, full `d` in `info`, and a `_cwd_delivered_shift` override hook. |
| `probe_env.py` | `SMACv2ProbeEnv` — the scripted discrete/continuous compensation + freeze. |
| `sigma_star.py` | the σ×β sweep driver (loads B0, rolls it through the probe, computes σ\*). |
| `test_probe.py` | pure-numpy arithmetic tests (no SC2 binary). |

---

## ⚠️ One behavior fix in `smacv2_env.py` (read this)

The committed env previously hard-coded `CWD_SEVERITY = 5` **and ignored the
documented `SMACV2_CWD_SEVERITY` env-var**. Both are fixed: knobs are now resolved
per instance as `env_args["cwd_severity"] → $SMACV2_CWD_SEVERITY → default 0.5`
(the design-doc default). **Consequence:** a plain run with no env-var now trains at
σ=0.5, not σ=5, and `SMACV2_CWD_SEVERITY=0` now genuinely turns the NS off (needed for
a correct B0). Set the dial explicitly for every run.

---

## Run order (on the run machine)

### 0. Arithmetic sanity (seconds, no SC2)
```bash
python -m harl.envs.smacv2.phase1.test_probe        # must print "All ... PASSED"
```

### 1. Train the baseline B0 — CWD OFF (reused at every σ; never retrained)
```bash
SMACV2_CWD_SEVERITY=0 python examples/train.py --algo happo --env smacv2 \
    --exp_name cwd_b0 --map_name protoss_5_vs_5
# (mappo works too; use the tuned config if you have one)
```
B0 = its undisturbed eval return (and win-rate). Note the results dir; it contains
`config.json` and `models/actor_agent*.pt`.

### 2. Sweep σ×β and read σ\* (minutes)
```bash
python -m harl.envs.smacv2.phase1.sigma_star \
    --load_config results/smacv2/protoss_5_vs_5/happo/cwd_b0/<run>/config.json \
    --model_dir   results/smacv2/protoss_5_vs_5/happo/cwd_b0/<run>/models \
    --episodes 40 \
    --sigmas 0.0,0.25,0.5,0.75,1.0,1.5,2.0 \
    --betas  0.0,0.25,0.5,0.75,1.0 \
    --out results/smacv2_cwd_phase1
```
Outputs: `results/smacv2_cwd_phase1/sigma_star.csv` (per cell) and
`sigma_star_summary.md` (the table + σ\* headline). The console prints B0, the two
mandatory checks, every cell, and the final σ\* (return and win-rate).

The driver is **frozen at its peak** (`--freeze 1.0`), so effective coupling
`c = σ`; use `--freeze -1` to instead run the live bombardment ramp.

---

## What the harness checks for you (pipeline §II.4 / Pitfalls)

- **Transparency** — σ=0 discrete β=1 must reproduce B0. Guaranteed by construction
  (σ=0 ⇒ no drift ⇒ re-aim is a no-op); the run asserts it and warns otherwise.
- **Works-when-it-should** — at the lowest σ>0, best-β must beat blind (β=0). This is
  the guard on the **one unverified detail**: the N/S/E/W → (x,y) map in
  `probe_env._DIR`. If this check fails, the direction/sign convention is wrong for
  your installed smacv2 — open its `get_agent_action` and fix `_DIR`.
- **Re-optimize β per σ** (Pitfall #1) — the summary takes `max_β` at every σ and
  reports `best_β(σ)`; never read a fixed-β frontier.
- **Reset reaches the probe** (Pitfall #2) — the probe inherits the base `reset()`,
  which zeroes `d`; `configure_probe()` also zeroes it between cells.
- **Native units** (Pitfall #3) — moot here: `d` is consumed **env-side**, never
  piggybacked through a normalized obs, so it is never mangled.

## Reading the result (four things, not just σ\*)

1. **Return-vs-σ crossing** — the discrete table's `max_β return` vs the bar.
2. **best_β vs σ** — a drift from 1.0 toward 0 is the loop-gain fingerprint
   (over-cancellation starts to hurt).
3. **Residual / dsat onset** — `resid` should climb with σ until `|d|` pins at
   `_DCAP` and flattens; `dsat` is the fraction of `|d|` axes at the cap.
4. **Win-rate beside return** — SMAC's natural headline; σ\* is reported on both.

## The decision (pipeline §II.6)

- **σ_target ≤ σ\*** → well-posed → build **PACT** (Phase 2) at σ ≤ σ\*. For discrete
  actions PACT uses the *soft* variant: feed `β·x2` and `x2` as obs features (native
  units, post-normalization) and let the recurrent policy pick the corrected discrete
  action — the discrete controller certified here is its ceiling.
- **σ_target > σ\*** → redesign: lower σ to σ\*−0.05, attenuate the channel, or cap the
  driver — then re-run this sweep.

## A priori expectation (to be replaced by the real run)

Because `|d|` is capped at `_DCAP = 2.0` = one move step, and the discrete controller
can always oppose the *cross-intent* part of `d`, expect discrete compensation to
recover well at low σ and degrade as `A·σ·|S|` approaches the cap (design-doc
calibration puts peak `cwd_load` at 0.5–1.0 near σ≈0.5). The likely picture: σ\*
somewhere in the **0.5–1.0** band on `protoss_5_vs_5`, with the continuous certificate
flat at ≈B0 throughout. The real numbers come from step 2 — treat this only as a
bracket for the σ grid.
