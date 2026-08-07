# PCR diagnosis campaign — RUNBOOK

Package reference:
[`harl/envs/mamujoco/diag/README.md`](../../../../harl/envs/mamujoco/diag/README.md).

**Every stage writes a markdown debug file** next to its CSV/NPZ. Those are the
artifacts to read while ideating; `diag_report.py` bundles the lot.

---

## Just run the whole thing

```bash
# 1. deploy the env (once)
ANT=$(python -c 'import gym.envs.mujoco.ant as m; print(m.__file__)')
cp "$ANT" "$ANT.backup.$(date +%s)"
cp harl/envs/mamujoco/diag/ant_diag.py "$ANT"

# 2. look at the plan without running anything
python scripts/diag_campaign.py --dry_run

# 3. run it, unattended
nohup python -u scripts/diag_campaign.py --results_root ./results > campaign.out 2>&1 &

# 4. come back
cat diag_out/_campaign/STATUS.md
```

`diag_campaign.py` resolves every checkpoint path, threads β* and PC between
stages, tees each stage to `diag_out/_campaign/logs/<stage>.log`, and rewrites
`STATUS.md` after every stage. It is **resumable**: each stage drops a marker, so
re-launching after an interruption picks up where it stopped (`--force` to redo).

**It is not a list of commands — it implements the spec's gates and branches.**
V0 failure hard-stops. A **V1 fail skips F1c/F2/F2b/F2c/F3a/F3b/D2 entirely**
(~30M steps) and routes to the repaired-env track. F2 fail caps D2 at 3M. E5
R²<0.3 skips E3-DOB. A0 finding a d-oracle run is *flagged for you*, not
auto-decided — that one turns on whether the prior run failed, which is a
judgement about an experiment, not a number.

Useful flags: `--priority P0` (~45M steps; default P1 ≈65M; P2 adds F4b),
`--jobs N` (parallel training arms — each is a full HASAC run, so raise only if
the GPU has room), `--f0 <models>` (use an inherited walker), `--only`/`--skip`,
`--dry_run`, `--force`.

Stage names for `--only` / `--skip`:

```
configs configs_verify v0_probes v0_sysid v0_recorder v0_runner v0_golden a0
f0 e1 d0 e2 e2b e2b_dcap e3 e4 e6 e5_random_A0.5 e5_random_A1.0 e5 e3dob
f1a f1b f1c_s1 f1c_s2 f2_s1 f2_s2 f2b f2c f0o f3a f3b f4 f4b_collect f4b
g x1 d2_s1 d2_s2 d1 d3 report
```

**Budget: this is days, not hours.** Tier 0 finishes in hours; the ~45M training
steps are the long pole. Frozen arms parallelize freely.

---

## Resolving paths by hand

If you drive the stages yourself, do not hand-copy checkpoint paths — a run dir
is `.../diag_<arm>/seed-<NNNNN>-<timestamp>/models` and the timestamp is
wall-clock:

```bash
eval "$(python scripts/diag_resolve.py --exports)"   # sets F0, F1A, F1C, BETA_STAR, PC...
python scripts/diag_resolve.py --list                # what exists, and what is measured
```

`diag_resolve.py` verifies rather than globs: it returns a checkpoint only if the
actor weights and `config.json` are there and — when a `run.log` survives — the
run's `[DIAG ENV]` banner matches the env vars that arm should have carried.
Passing the wrong checkpoint is the E-5 error one level down.

```bash
export DIAG=./diag_out          # all debug files land under here
```

---

## Stage 0 — generate the arm configs (seconds)

```bash
python scripts/diag_make_configs.py                  # writes the 14 arm JSONs
python scripts/diag_make_configs.py --verify         # asserts Prohibition 2
python scripts/diag_make_configs.py --print_manifest # the run table + budget
```

Every arm JSON is the frozen `../hasac/config.json` plus that arm's diag flags
and step budget. **Re-run `--verify` after any hand-edit**: it fails loudly if an
arm has drifted from the host hyperparameters, which is the difference between a
campaign and a pile of incomparable runs.

## Stage V0 — self-tests (minutes, no GPU) — **gate: everything must PASS**

```bash
python -m harl.envs.mamujoco.diag.probes         --selftest   # -> $DIAG/v0/v0_probes.md
python -m harl.envs.mamujoco.diag.sysid          --selftest   # -> $DIAG/v0/v0_sysid.md
python -m harl.envs.mamujoco.pcr_diag                         # -> $DIAG/v0/v0_recorder.md
python -m harl.runners.off_policy_diag_runner                 # -> $DIAG/v0/v0_runner.md
```

Then **deploy the env and run the golden test** (Prohibition 5 — nothing else may
run until this passes):

```bash
ANT=$(python -c 'import gym.envs.mujoco.ant as m; print(m.__file__)')
cp "$ANT" "$ANT.backup.$(date +%s)"                          # keep the old one
cp harl/envs/mamujoco/diag/ant_diag.py "$ANT"
env -u ANT_PCR_SEVERITY -u ANT_PCR_FREEZE_A -u ANT_PCR_MASK \
    -u ANT_PCR_DCAP -u ANT_PCR_ORACLE -u ANT_PCR_CORACLE \
    python -m harl.envs.mamujoco.diag.test_ant_diag           # -> $DIAG/v0/v0_ant_diag.md
```

The golden test requires a clean environment — it asserts `ant_diag` with no env
vars is byte-identical (1e-12) to `ant_pcr_v1`, the frozen copy of what is
deployed today. It also runs the knob tests, the info-key invariants and the two
category-C litmus asserts (N=1 vanishing; frozen-partner persistence).

## Stage A0 — forensics (minutes) — **do first, do not skip**

```bash
python scripts/diag_a0_inventory.py --results_root ./results --out $DIAG/a0
# -> $DIAG/a0/a0_summary.md, a0_runs.csv, a0_phase_corrected.csv, a0_checkpoints.csv
```

Read `a0_summary.md` before spending a single GPU-hour. It decides two things:

* **E-5**: was the "ECL oracle" the *tag*-oracle or the env *d*-oracle? If A0
  shows a d-oracle at σ=0.9 / 10M / de-aliased that already failed, **abort rule 2
  fires**: do not re-run D2 at 10M — run a 3M confirmation and promote F2/F2b to
  double seeds.
* **F0**: is there already a confirmable SEVERITY=0 / MASK=off HASAC walker? If
  yes, Tier 0 starts immediately. If not, F0 must be trained first — every Tier-0
  probe needs it and B0 is the denominator of V1/V2/V4/V7.

---

## Tier 1 — F0 (5M) — only if A0 found no walker

```bash
ANT_PCR_MASK=off python examples/train.py \
    --load_config tuned_configs/mamujoco/Ant-v2-4x2/diag/f0.json \
    --exp_name diag_f0 --seed 1
```

Frozen arms parallelize freely — F0/F1a/F1b/F1c/F2/F2b/F2c can all run at once.

---

## Tier 0 — the decisive tier (eval-only, ~1–2 days wall-clock)

```bash
F0=$(python scripts/diag_resolve.py --arm f0)   # or your inherited walker's models/

# E1 (+ B0) and D0 — the collapse profile and the no-adaptation floor
python scripts/diag_tier0.py --ckpt $F0 --stage e1 --out $DIAG/e1
python scripts/diag_tier0.py --ckpt $F0 --stage d0 --out $DIAG/d0

# E2 — THE EXISTENCE EXPERIMENT (gate V1). Prints beta*.
python scripts/diag_tier0.py --ckpt $F0 --stage e2 --out $DIAG/e2

# E2b / E3 — the severity frontier and the information frontier (need beta*)
python scripts/diag_tier0.py --ckpt $F0 --stage e2b --beta_star "$(python scripts/diag_resolve.py --beta_star)" --out $DIAG/e2b
python scripts/diag_tier0.py --ckpt $F0 --stage e2b --beta_star "$(python scripts/diag_resolve.py --beta_star)" --dcap_leg --out $DIAG/e2b  # P1
python scripts/diag_tier0.py --ckpt $F0 --stage e3  --beta_star "$(python scripts/diag_resolve.py --beta_star)" --out $DIAG/e3

# E4 / E6 — the information-free escape and the harm-channel attribution
python scripts/diag_tier0.py --ckpt $F0 --stage e4 --out $DIAG/e4
python scripts/diag_tier0.py --ckpt $F0 --stage e6 --out $DIAG/e6
```

`--stage all` runs E1→E6 in one process and orders them correctly (E1 first,
because B0 is the denominator of every gate; β* is threaded from E2 to E2b/E3):

```bash
python scripts/diag_tier0.py --ckpt $F0 --stage all --out $DIAG/tier0
```

> **Ordering rule 1 (§10.2).** If **E2 fails at σ=0.9 (V1 fail)**: skip
> F1c/F2/F3/D2 at σ=0.9 **entirely**. Run E2b to locate σ*, then re-run the Tier-0
> suite + F1c at the repaired setting (R-a or R-b). The campaign continues on the
> repaired env and the report records the ill-posedness finding (§9.3).

### E5 + E3-DOB (hours)

E1's cells dump contiguous NPZ trajectories (source **(a)**, competent
on-manifold). Add the other two sources, then fit:

```bash
# (b) blind-collapsed on the drifting env — D0 already dumped it
# (c) 50k excitation-rich random-action steps at A in {0.5, 1}
for A in 0.5 1.0; do
  python scripts/diag_tier0.py --ckpt $F0 --probe identity --A $A \
      --episodes 50 --dump_traj --out $DIAG/e5_random_A$A
done

python -m harl.envs.mamujoco.diag.sysid \
    --data "e1_frozen:$DIAG/e1/recorder/**/traj_*.npz" \
    --data "blind_drift:$DIAG/d0/recorder/**/traj_*.npz" \
    --data "random_excite:$DIAG/e5_random_A*/recorder/**/traj_*.npz" \
    --out $DIAG/e5 --export_dob $DIAG/e5/dob_filter.npz
# -> $DIAG/e5/e5_sysid.md, e5_r2.csv, dob_filter.npz

python scripts/diag_tier0.py --ckpt $F0 --stage e3dob --beta_star "$(python scripts/diag_resolve.py --beta_star)" \
    --dob $DIAG/e5/dob_filter.npz --out $DIAG/e3dob
```

> **Abort rule 4.** E5 R² < 0.3 everywhere on source (a) ⇒ **skip E3-DOB**, log V6
> fail early, do not spend the eval pass.

---

## Tier 1 — frozen-slice trainings (≈25M steps, parallelizable)

| arm | launch |
|---|---|
| **F1a** | `ANT_PCR_FREEZE_A=0 python examples/train.py --load_config .../diag/f1a.json --exp_name diag_f1a --seed 1` |
| **F1b** | `ANT_PCR_FREEZE_A=0.5 python examples/train.py --load_config .../diag/f1b.json --exp_name diag_f1b --seed 1` |
| **F1c** | `ANT_PCR_FREEZE_A=1.0 python examples/train.py --load_config .../diag/f1c.json --exp_name diag_f1c_s1 --seed 1` (also seed 2) |
| **F2** | `ANT_PCR_FREEZE_A=1.0 ANT_PCR_ORACLE=1 python examples/train.py --load_config .../diag/f2.json --exp_name diag_f2_s1 --seed 1` (also seed 2) |
| **F2b** (P1) | `ANT_PCR_FREEZE_A=1.0 python examples/train.py --load_config .../diag/f2b.json --exp_name diag_f2b --seed 1` |
| **F2c** (P1) | `ANT_PCR_FREEZE_A=1.0 python examples/train.py --load_config .../diag/f2c.json --exp_name diag_f2c --seed 1` |
| **F0o** (P1) | `ANT_PCR_MASK=off ANT_PCR_ORACLE=1 python examples/train.py --load_config .../diag/f0o.json --exp_name diag_f0o --seed 1` |
| **F3a** | `ANT_PCR_FREEZE_A=1.0 python examples/train.py --load_config .../diag/f3a.json --exp_name diag_f3a --seed 1 --model_dir $F0` |
| **F3b** | `ANT_PCR_FREEZE_A=1.0 ANT_PCR_ORACLE=1 python examples/train.py --load_config .../diag/f3b.json --exp_name diag_f3b --seed 1 --model_dir $F0O` (`eval "$(python scripts/diag_resolve.py --exports)"` sets `$F0O`) |

F3b initializes from **F0o**, not F0 — see deviation 2 in the package README.

### F4 / F4b — the forgetting curve

```bash
ANT_PCR_FREEZE_A=1.0 python scripts/diag_f4.py \
    --config tuned_configs/mamujoco/Ant-v2-4x2/diag/f4.json \
    --model_dir $F0 --exp_name diag_f4 --seed 1 --out $DIAG/f4

# F4b (P2) — two phases, because the env var is read once per process
ANT_PCR_FREEZE_A=0.0 python scripts/diag_f4.py --config .../diag/f4.json \
    --model_dir $F0 --collect_pool $DIAG/f4/trough_pool.npz --pool_steps 200000
ANT_PCR_FREEZE_A=1.0 python scripts/diag_f4.py --config .../diag/f4.json \
    --model_dir $F0 --rehearse_pool $DIAG/f4/trough_pool.npz --rehearse 0.25 \
    --exp_name diag_f4b --out $DIAG/f4
```

## G + X1

```bash
eval "$(python scripts/diag_resolve.py --exports)"   # F0, F1A, F1B, F1C, F3A, F2, F3B

python scripts/diag_crosseval.py --out $DIAG/g \
    --policy "f0:$F0" \
    --policy "f1a:$F1A@0.0" \
    --policy "f1b:$F1B@0.5" \
    --policy "f1c:$F1C@1.0" \
    --policy "f3a:$F3A" \
    --oracle_policy "f2:$F2" --oracle_policy "f3b:$F3B"
# -> $DIAG/g/g_matrix.{md,csv,png} + g_summary.json; prints V4 and **PC**

python scripts/diag_distill.py --out $DIAG/x1 \
    --pc "$(python scripts/diag_resolve.py --pc)" \
    --expert "f1a:$F1A@0.0" \
    --expert "f1b:$F1B@0.5" \
    --expert "f1c:$F1C@1.0"
# -> $DIAG/x1/x1_distill.md, x1_eval.csv; prints V5
```

`@A` marks a policy as the diagonal's expert at slice A — that is what PC is
computed from. **PC replaces the arbitrary 6500 as the target for any future
method: `target := 0.9·PC`.**

## Tier 2 — drift arms (≈35M)

```bash
# D2 (P0) — the instrumented, banner-labeled replacement for the ambiguous E-5 run
ANT_PCR_ORACLE=1 python examples/train.py --load_config .../diag/d2.json \
    --exp_name diag_d2_s1 --seed 1        # also seed 2
# D1 (P1) — the forensic re-run of the baseline failure, under the microscope
python examples/train.py --load_config .../diag/d1.json --exp_name diag_d1 --seed 1
# D3 (P1) — the ratchet
python examples/train.py --load_config .../diag/d3.json --exp_name diag_d3 \
    --seed 1 --model_dir $F0
```

> **Abort rule 3.** F2 fail ⇒ cap D2 at 3M. Checkpoint-abort: if D2's 2M-step
> trough slice is below F1c's at the same step, stop it.

## The verdict

```bash
python scripts/diag_report.py --diag_out $DIAG --results_root ./results \
    --bundle ./diag_bundle
# -> diag_bundle/verdict.md, verdict.json, checkpoint_index.json, diag_out/, runs/
```

Prints the eight axis readings, the scalar facts (B0, PC, the E1 deficit
decomposition, the fall-cause table, σ*, E4, the F2b/F2c three-way), the
**decision-tree walk with the fired leaf**, and the WP-1..5 certificates. The
walk is total over a *complete* axis tuple; if an axis is still UNKNOWN it stops
rather than guessing.

---

## Banners — what each arm must print

Every arm prints these at startup. **If a banner is missing, the run is not
classifiable and does not count** (Prohibition 3 — this is the direct fix for the
E-5 ambiguity that caused this campaign).

```
[DIAG ENV] SEVERITY=0.9 FREEZE_A=<...> MASK=<...> DCAP=None ORACLE=<0|1> CORACLE=0 RHO=0.8 P=40000 B=0.2
[DIAG ARM] d_to=<none|share|obs|both> (RAW, post-normalization; torque units) obs A->B  share C->D  clock_offset=N
[DIAG RUN] algo=hasac_diag telemetry=True interval=10000 ... | auto_alpha=False alpha=0.2 | eval_dephase=<...> d_to=<...>
```

| arm | expected `[DIAG ENV]` fragment |
|---|---|
| f0 | `FREEZE_A=None MASK=off ... ORACLE=0` |
| f0o | `FREEZE_A=None MASK=off ... ORACLE=1` |
| f1a / f1b / f1c | `FREEZE_A=0.0 / 0.5 / 1.0 MASK=both ORACLE=0` |
| f2 | `FREEZE_A=1.0 MASK=both ORACLE=1` |
| f2b / f2c | `FREEZE_A=1.0 MASK=both ORACLE=0` + `[DIAG ARM] d_to=share` / `d_to=both` |
| f3a | `FREEZE_A=1.0 ... ORACLE=0` |
| f3b | `FREEZE_A=1.0 ... ORACLE=1` |
| d1 | `FREEZE_A=None MASK=both ORACLE=0` |
| d2 | `FREEZE_A=None MASK=both ORACLE=1` |
| d3 | `FREEZE_A=None MASK=both ORACLE=0` |

Capture stdout so A0 can parse it later:

```bash
... python examples/train.py ... 2>&1 | tee results/.../run.log
```

## Artifact map

| stage | debug file (read this) | machine-readable |
|---|---|---|
| V0 | `$DIAG/v0/v0_*.md` | — |
| A0 | `$DIAG/a0/a0_summary.md` | `a0_runs.csv`, `a0_phase_corrected.csv`, `a0_checkpoints.csv` |
| Tier 0 | `$DIAG/<stage>/tier0_<stage>.md` | `tier0_cells.csv`, `recorder/<cell>/pcr_diag_*.csv`, `*_episodes.csv`, `traj_*.npz` |
| E5 | `$DIAG/e5/e5_sysid.md` | `e5_r2.csv`, `dob_filter.npz` |
| training arms | run dir | `progress.txt`, `eval_debug.csv`, `diag_telemetry.csv`, `diag_qcal.csv`, `diag_probes.npz` |
| F4 | `$DIAG/f4/f4_*.md` | `f4_*.csv` |
| G | `$DIAG/g/g_matrix.md` | `g_matrix.csv`, `g_matrix.png` |
| X1 | `$DIAG/x1/x1_distill.md` | `x1_eval.csv` |
| verdict | `diag_bundle/verdict.md` | `verdict.json`, `checkpoint_index.json` |

## Budget (`--print_manifest`)

P0 ≈ 45M env steps + evals — comparable to the v2 stage (3×10M). Full ≈ 65M.
Frozen arms parallelize freely; Tier 0 is eval-only and finishes in hours.

## Seeds

2 for the verdict-critical arms (F1c, F2, D2), 1 elsewhere. **A verdict that
flips between the two seeds triggers a third.** Report per-seed numbers always;
never average across protocols or env configs (§10.2 rule 6).
