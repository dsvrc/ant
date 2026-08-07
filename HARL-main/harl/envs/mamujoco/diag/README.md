# PCR diagnosis campaign — package reference

Implements `PCR_diagnosis_campaign_spec.md` ("Measure everything once. Then build
once."). **Nothing here is a method component** (Prohibition 1): no learned
identifier, no replay shaping, no reward edit, no host-loss change. The
deliverable is an artifact bundle plus a verdict; the next method spec is written
*from* that bundle.

**The runbook — ordered commands, banners, artifact map — is
[`tuned_configs/mamujoco/Ant-v2-4x2/diag/README.md`](../../../../tuned_configs/mamujoco/Ant-v2-4x2/diag/README.md).**
This file is the reference: what each module is, what every logged column means,
and where the implementation deviates from the spec (and why).

---

## Files

| file | role |
|---|---|
| `ant_pcr_v1.py` | Frozen **golden copy** of the deployed `ant.py` (SEVERITY=0.9). Never edited. Byte-identical to `~/Desktop/ant.py` at commit time (sha256 `101f65d1…`). |
| `ant_diag.py` | **The only file ever copied over `gym/envs/mujoco/ant.py`.** `ant_pcr_v1` + env-var knobs + info keys + the mandatory banner. Defaults are byte-identical to the golden copy. |
| `test_ant_diag.py` | Stage V0: golden equivalence, knob isolation, info-key invariants, the two category-C litmus asserts. |
| `knobs.py` | Knob access on the **deployed** ant module. Always go through this (see the trap below). |
| `probes.py` | `ProbeShim` + the Tier-0 probe/transform library. Pure numpy, `--selftest`. |
| `sysid.py` | E5 offline ridge system-id + the E3-DOB filter export. Pure numpy, `--selftest`. |
| `diag_mujoco.py` | `DiagMujocoMulti`: the training-arm env shim (recorder + the F2b/F2c privileged-input schemas + the eval clock offset). |
| `report_io.py` | Shared debug-file / report / bootstrap-CI helpers. Home of the campaign's **CI rule**. |
| `../pcr_diag.py` | The flight recorder (extended per §3.1; append-only). |
| `../../../runners/off_policy_diag_runner.py` | `hasac_diag`: HASAC + read-only, **RNG-transparent** telemetry. |
| `../../../common/buffers/diag_off_policy_buffer.py` | Payload-aligned replay buffer (diagnostic arrays; the sampler never reads them). |
| `../../../../scripts/diag_*.py` | The drivers: `make_configs`, `a0_inventory`, `tier0`, `f4`, `crosseval`, `distill`, `report`. |

---

## Three traps this package exists to close

**1. `ant_diag` is not the running env.** `MujocoMulti` builds its env with
`gym.make("Ant-v2")` → `gym.envs.mujoco.ant.AntEnv`. Deployment copies
`ant_diag.py` *over* that file, so the two are the same **source** but different
**module objects with different globals**. `ant_diag.set_freeze_a(1.0)` therefore
does *nothing* to a running env, and every Tier-0 cell would silently measure
whatever the env var said while the report labelled it otherwise. Always go
through `diag/knobs.py`, which resolves the deployed module and refuses to run if
`ant_diag.py` was never deployed.

**2. `MujocoMulti.reset()` bypasses `wrapped_env`.** It resets via
`self.timelimit_env`, a reference captured at construction — so a `ProbeShim`
installed on `wrapped_env` never sees a reset, and per-episode probe state (delay
rings, EMAs, DOB history) leaks across episodes. Use `probes.install_probe()`,
which hooks both.

**3. Telemetry that consumes randomness is part of the experiment.** Every
telemetry pass in `hasac_diag` runs inside `_rng_frozen()`, which saves and
restores the torch / CUDA / numpy states. So telemetry-ON and telemetry-OFF
produce the **same** training trajectory for a given seed — which is what makes
D1 a re-run of the baseline failure rather than a different run.

---

## Finding: the privileged obs never arrived in torque units

Not a design choice — something this implementation found while wiring F2b, and
it changes how axis V3 must be read.

```python
# harl/envs/mamujoco/multiagent_mujoco/mujoco_multi.py
obs_i = np.concatenate([state, agent_id_feats])
obs_i = (obs_i - np.mean(obs_i)) / np.std(obs_i)     # whole-vector, every step
```

`ANT_PCR_ORACLE=1` appends `d` **inside** `AntEnv._get_obs()`, i.e. *before* that
line. So every oracle arm ever run fed its policy `(d − mean_t)/std_t`, not `d`.
A feed-forward canceller needs `d` in torque units — it commands
`tau = desired − d` — and recovering it requires inverting a per-step scale set
largely by cfrc_ext contact spikes. `ANT_PCR_CORACLE=1` has it worse: the scalar
`c` becomes a *different function of c at every timestep*.

Consequences:

* **E-5 / E-7 ("the oracle failed") inherit a competing explanation** that has
  nothing to do with information being unhelpful.
* **Axis V3 as specified (F2 vs F1c) cannot separate** "info is toxic" (H-C4)
  from "info arrived mangled".
* **X1's specified eval** (`ANT_PCR_CORACLE=1` supplying c) would have measured a
  normalization artifact, not representational capacity.

So, alongside the spec's arms exactly as written:

| arm | schema | isolates |
|---|---|---|
| **F2** (spec) | `ANT_PCR_ORACLE=1` → normalized d, actor+critic | the spec's reading |
| **F2c** (added, P1) | `d_to=both` → **raw** d, actor+critic | F2 vs F2c = **the units** |
| **F2b** (spec) | `d_to=share` → raw d, critic only | F2c vs F2b = actor conditioning vs critic variance reduction |
| — | — | F2b vs F1c = the critic effect alone |

and `diag_distill.py` defaults to `--c_source info` (blind env, exact c from
`info`), with `--c_source coracle` available as the literal-spec contrast.

Verified: `use_feature_normalization` is **inert** for HASAC — its actor
(`SquashedGaussianPolicy`) and critic (`ContinuousQNet`) are built on `PlainMLP`,
which has no input LayerNorm; only `MLPBase` (the on-policy side) honours the
flag. So raw `d` does reach the first `Linear` in torque units.

`diag_report.py` prints `F2c − F2` explicitly and says what it means.

---

## Other deviations from the spec (each deliberate, each recorded)

| # | Spec says | Implemented | Why |
|---|---|---|---|
| 1 | NPZ dump is "decimated 1-in-K full records" | decimated **per episode** (`dump_every_k_episodes`), contiguous | Step-decimation makes E5 *impossible*: its object is the last-L **consecutive** window, which a 1-in-K stream no longer contains. Episode-level decimation honours the size cap for the same reason while preserving lag structure. |
| 2 | F3b = "F3a + d-oracle, `model_dir=F0`" | added arm **F0o**; F3b inits from it | `ANT_PCR_ORACLE=1` grows the obs by 8, so F0's actor weights do not fit — and because the whole vector is renormalized, an F0 policy is not even approximately the right init. F0o is the same 5M stationary walker in the oracle schema (`MASK=off` ⇒ d≡0 ⇒ the 8 oracle dims are constant zeros). |
| 3 | knobs are module constants | knobs **snapshot per env instance** + runtime setters | Required by the spec's own design: `diag_tier0.py` "owns the full grid loops" over FREEZE_A/SEVERITY (one process, many settings), and F4 trains at FREEZE_A=1 while scoring at FREEZE_A=0. Module constants can express neither. The env-var interface is unchanged and is still the only thing a training arm uses. |
| 4 | F4b "repeat with 25% of every minibatch from a frozen buffer of F0 trough transitions" | **two-phase**: `--collect_pool` at `FREEZE_A=0`, then `--rehearse_pool` at `FREEZE_A=1` | The env var is read once per process at import, so the collection envs (A=0) and training envs (A=1) cannot be the same processes. The pool is loaded into the buffer's **oldest** slots so n-step targets stay inside trough episodes. |
| 5 | `dc(halflife=64)` listed beside `ema(64)` | both implemented; `dc:h` **is** `ema:h` | Kept as separate cells on purpose — they answer different questions (how fast must an estimator be, vs is a *slow chart* sufficient at all). `probes.py`'s self-test asserts they compute the same thing so nobody "optimizes away" the duplicate later. |
| 6 | F-joint = "last-L ALL agents' obs ⊕ actions" | last-L **own** obs ⊕ **all** actions | Every agent's obs is the *same* vector: `normalize(concat([state, onehot_i]))`, and the normalizing mean/std do not depend on `i` (the one-hot always contributes one 1 and three 0s). So `obs_i` and `obs_j` differ **only** in 4 one-hot slots — concatenating them quadruples D for zero information. The genuine CTDE increment is the teammates' *actions*, which is what the recursion feeds on. Asserted at load time (`_assert_obs_shared`), not assumed. |
| 7 | configs: "one JSON per manifest arm" | generated by `scripts/diag_make_configs.py` + `--verify` | Makes Prohibition 2 ("host hyperparameters identical across every arm") a mechanical property. `--verify` catches a later hand-edit that quietly retunes an arm before it costs 45M steps. |

---

## Column dictionary

### `pcr_diag_*.csv` — the flight recorder (per step)

v1 columns, unchanged: `step_global`, `ep_step`, `done`, `fall`, `pcr_payload`,
`pcr_load`, `pcr_loadmax`, `d_app_mean`, `d_app_max`, `torso_height`, `sat_frac`,
`tau_absmean`, `delivered_absmean`, `reward`, `r_forward`, `r_ctrl`, `r_contact`,
`r_survive`.

| column | one line |
|---|---|
| `d_hip_common` | `mean(d[0,2,4,6])**2` — common-mode disturbance power (eigenvalue +3 of M = 11ᵀ−I). |
| `d_hip_diff` | `var(d[0,2,4,6])` (population) — difference-mode disturbance power (eigenvalue −1). |
| `d_ank_common` / `d_ank_diff` | the same for ankles. |
| `tau_hip_common` / `tau_hip_diff` | the same decomposition of the **commanded** torque — the E2 derivation says the difference-mode **DC** content is amplified ×9 into d, so this is the column that predicts whether cancellation can work. |
| `tau_ank_common` / `tau_ank_diff` | the same for ankles. |
| `clip_frac_cmd` | fraction of joints with `\|a\| > 0.999` — the **policy** riding the rails. |
| `sat_frac` | fraction with `\|tau+d\| > 1` — the **disturbance** exceeding actuator authority. Different thing; both are logged. |
| `fwd_vel` | `(xposafter − xposbefore)/dt` (equals `reward_forward`). |
| `sumzero_resid` | `\|mean(tau_hip)\| + \|mean(tau_ank)\|` — distance from the sum-zero manifold. E4 needs it because the post-projection clip can re-break sum-zero. |
| `pcr_clock` | the global shift clock after this step's tick. |
| `term_cause` | `fall_low` / `fall_high` / `nonfinite` / `timeout` / `other` / `""`. |

### `pcr_diag_*_episodes.csv`
`ep_index`, `step_global_end`, `ep_return`, `ep_len`, `term_cause`,
`payload_start`, `payload_end`, then `mean_*` of every per-step column above.
This is what E1's deficit decomposition is computed from.

### `diag_telemetry.csv` — `hasac_diag` (one row per 10k env steps)

| column | one line |
|---|---|
| `env_step` | buffer rows inserted = env steps (warmup included). |
| `reward_mean`, `payload_mean` | last insert's mean reward / payload. |
| `td_q0..q4` | median \|TD\| over an independent uniform replay draw, binned by payload quintile. |
| `age_q0..q4` | mean replay age (env steps) per quintile — how stale each phase is (H-C2). |
| `n_q0..q4` | draw counts per quintile. **Under a frozen arm only q0 is populated** — that is one bin, not four empty phases. |
| `collect_q0..q4` | fraction of *this window's* collection in each quintile. Crossed with `drift_*` this is the who-overwrites-whom matrix. |
| `drift_b0..b4` | `‖π_t(bank) − π_{t−1}(bank)‖` per frozen probe bank (H-C2). |
| `entropy_b0..b4` | MC estimate of `E[−log π(a\|s)]` per bank (H-C5's live half). |
| `alpha_mean` | mean per-agent α. |
| `auto_alpha` | 0/1. **The tuned config pins `auto_alpha: false`, α=0.2**, so H-C5's "one auto-tuned α cannot serve trough and peak" half is *moot for this host* — the column is a constant and must not be reported as evidence. The live half is `entropy_b*`. |
| `feature_rank` | 99%-energy effective rank of the critic's penultimate activations (plasticity loss, P2; every 200k steps). |
| `banks_ready` | 1 once the probe banks are frozen. |

### `diag_qcal.csv`
`env_step`, `thread`, `payload_start`, `payload_end`, `q0` (min over twin critics
at s₀,a₀), `disc_return` (realized, discounted), `ep_return`, `ep_len` — §3.2.4.

### `eval_debug.csv`
`step`, `thread`, `payload_end`, `ep_return`, `ep_len` — the C4 protocol (§3.3).

### `diag_probes.npz`
`bank_payloads`, `bank_bins`, `bank_obs`, and the per-checkpoint mean action
vectors (`log_step`, `log_bank`, `log_agent`, `log_action`) — the raw material for
the drift matrix.

---

## The CI rule (§3.3), in one place

`report_io.compare()` implements it so no script can quietly weaken it: every
reported comparison carries a bootstrap 95% CI over episodes, and a claim
"X > Y" requires **non-overlapping CIs or a gap ≥ 3× the pooled std**. Its
verdict `~` means *not separated at the campaign's evidence bar* — never "equal".
40 episodes gives ±~150–250 on Ant (return std 400–800), sufficient for gates
that are all ≥ 500-point contrasts.

## Standing prohibitions (§11.2)

1. No method components in this campaign.
2. Host hyperparameters identical across every arm — mechanically enforced by
   `diag_make_configs.py --verify`.
3. Every privileged/diagnostic arm prints its banner; every run dir must be
   classifiable by A0's parser. **Hard requirement, not hygiene.**
4. `FREEZE_A` / `MASK` / `DCAP` / `SEVERITY≠0.9` are diagnostic configs — never a
   headline training arm.
5. `ant_diag.py` defaults must pass the golden test before any deployment.
6. Tier-0 probes: deterministic actors; the A=0 control is mandatory per probe.
7. No protocol mixing in any figure; phase-corrected historical numbers always
   carry the `reconstructed` flag.
8. The report states measurements and axis readings; **the method choice is made
   from Part 8's tree with the user, not inside the report.**
