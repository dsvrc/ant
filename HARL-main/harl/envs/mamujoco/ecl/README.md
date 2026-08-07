# ECL in HARL — Equilibrium Continuation Learning (PCR Ant)

ECL is the successor to ECHO‑R, built on its post‑mortem. The failure of
conditioning was an **equilibrium‑selection / learning pathology under drift**,
not an information deficit (the c‑oracle was flat across the cycle). So ECL does
**not** rely on adding driver info to the policy input. Its four components:

- **[I] Identifier** (trainer‑side, `ecl_identifier.py`): recovers `ĉ(t)=A(t)·σ`
  *passively* and *centrally* from the replay buffer's **joint actions** — a
  windowed two‑regressor ridge whose coefficient ratio `b2/b1 = c` cancels the
  unknown plant gain. **No probe** (the natural gait disturbance is 30–50× any
  tolerable probe; this is why it works where ECHO‑R's per‑agent probe couldn't).
  `ĉ` is used only to *tag* transitions and *steer sampling* — never a network input.
- **[L] Localizer** (`ecl_off_policy_buffer.py`): every minibatch is kernel‑localized
  around the current `ĉ` (Gaussian in the tag), so no gradient ever averages across
  distant contexts — the average‑game trap is deleted by construction.
- **[A] Anchor** (`ecl_off_policy_buffer.py`): a fraction `β_A` of every minibatch
  rehearses trough (`c≈0`) experience, so the stationary walking competence is
  never surrendered — each cycle costs a short adaptation, not a relearn.
- **[E] Envelope** (`envelope_adapter.py` + `ecl_mujoco.py`, agent‑side): each agent
  appends one scalar `ε_i` — the residual‑power fraction of its own readout that its
  own effort doesn't explain — a decentralized monotone chart of the path. **No
  probe; the action is untouched.**

The host (HASAC) losses/critic/update are unchanged (spec P6). Two real knobs:
kernel width `h` (`h_frac`) and anchor fraction `β_A`.

## ⚠️ Part 0 — benchmark repair (mandatory, do this FIRST)

The spec's Part 0 derives a **phase boundary at `c = A·σ = 1`**: for `c > 1` the
difference (sum‑zero) leg modes become non‑minimum‑phase (`delivered = (1−c)·τ`
flips sign) — **no stable causal policy can compensate, with any information.**
This is *separate* from the saturation boundary we already fixed. At `σ = 1.5` the
payload peaks still spend time at `c > 1`, so even the d‑oracle cannot recover
there and **no method has an existence proof.** Fix it (env‑side config, sanctioned
by the PCR doc — not a method edit):

- **Route A (config‑only, simplest):** set `SEVERITY ≤ 0.9` in `ant.py` so
  `c(t) < 1` for the whole cycle, then verify the blind collapse still survives.
- **Route B (recommended if you want a deep collapse):** keep `SEVERITY` high and
  enable the PCR §9 load cap `_DCAP = 0.5` in `ant.py` (bounds `|d| ≤ 0.5`; the
  loop saturates instead of diverging and the oracle stays feasible).

**Stage‑G gate before any full ECL run** (Protocol B — init all arms from a
pretrained `SEVERITY=0` walker): `d‑oracle ≫ blind` (existence restored) and the
`c‑oracle` trough return stays ≈ baseline. If the d‑oracle still fails after the
repair, the benchmark needs redesign — stop.

## Fast verification (do this before the slow full run)

ECL adds only tiny overhead — the slowness is HASAC itself (off-policy, per-agent
sequential critic+actor updates). Two cheap ways to confirm it works first:

1. **Unit test (seconds, pure numpy, no simulator/torch/mujoco)** — checks the
   *core* logic: the identifier recovers `c` and the envelope is monotone in `c`:
   ```bash
   python -m harl.envs.mamujoco.ecl.test_ecl      # expect "T1 PASS" and "T3 PASS"
   ```
2. **Short training run (~10 min, real env)** — full integration + on-env identifier
   tracking, 1.25 payload cycles, small buffer, no eval:
   ```bash
   python examples/train.py \
       --load_config tuned_configs/mamujoco/Ant-v2-4x2/ecl/config_smoke.json --exp_name ecl_smoke
   ```
   Then read the debug CSV written to **`<run_dir>/ecl_debug.csv`** (path printed
   at startup; `<run_dir>` is under `./results/.../ecl_smoke/...`). One row per
   identifier refresh, columns:
   `env_step, c_now, c_max_seen, pcr_payload, eps_mean, cond_number, trough_frac, reward_mean`.
   The **D1 check**: `c_now` and `pcr_payload` should rise and fall *together*
   (`c_now ≈ pcr_payload·σ`, so `c_now/pcr_payload ≈ c_max_seen ≈ σ`). If they
   track, the identifier works and the pipeline is wired. Also: `cond_number`
   should stay well below ~1e3 (else raise the SAC temperature), and `trough_frac`
   should be a healthy fraction (the anchor's supply). Only then launch the full run.

   Quick look: `column -t -s, <run_dir>/ecl_debug.csv | less`, or plot
   `c_now` vs `pcr_payload`.

## Run (v2)

```bash
# --- Stage V0: unit tests (this machine, pure numpy, seconds) ---
python -m harl.envs.mamujoco.ecl.test_ecl          # expect "T1' PASS" and "T2' PASS"

# --- Stage V1: smoke run (~10 min, real env) — check gates BEFORE the 10M runs ---
python examples/train.py \
    --load_config tuned_configs/mamujoco/Ant-v2-4x2/ecl/config_smoke.json --exp_name ecl_smoke

# --- Stage V2: the three headline arms (10M each, all with de-aliased eval) ---
# (a) ECL v2 blind — the headline
python examples/train.py --load_config tuned_configs/mamujoco/Ant-v2-4x2/ecl/config.json --exp_name ecl_v2
# (b) ECL v2 oracle-tag — upper-bounds what a perfect identifier buys (labeled ORACLE arm)
python examples/train.py --load_config tuned_configs/mamujoco/Ant-v2-4x2/ecl/config_tag_oracle.json --exp_name ecl_tag_oracle
# (c) G2 d-oracle ceiling — set ORACLE=True in ant.py, run plain hasac (env-side arm)
#     python examples/train.py --load_config <hasac Ant-4x2 config> --exp_name g2_doracle

# envelope-only degraded variant on HAPPO (on-policy: no buffer, so no [L]/[A])
python examples/train.py --load_config tuned_configs/mamujoco/Ant-v2-4x2/ecl_happo/config.json \
    --exp_name ecl_happo
```

### Stage V1 smoke gates (from `ecl_debug.csv`; if the first fails, don't launch 10M)
- **`tag_payload_corr > 0.9`** sustained after ~200k steps — the D1 pass, the v1 killer.
- `c_max_seen ∈ [0.7, 1.1]`; `c_now` at `pcr_payload > 0.95` reads **≥ 0.6**.
- `lock_gain` median **> 0.05**; `scan_best_offset == 0` and `scan_best_offset_ankle == 0`.
- ε̃ (`eps_norm_mean`) spans **≥ [0.15, 0.85]** within a cycle.
- If `tag_payload_corr` stays low → set `identifier_mode` unchanged but check
  `clip_frac` (if ~1 the gait rides the rails — expected) and consider the
  instrument-variant fallback (spec C1 fallback).

### `eval_debug.csv` (C4) — the de-aliased, per-episode eval
Columns `step, thread, payload_end, ep_return, ep_len`. Each eval round covers 20
evenly-spaced payload phases (eval clocks are stratified, C4.1), so a round mean is
a **true cycle-average**. The runner also prints per round the **trough-slice**
(mean return, payload lowest decile = the ratchet metric) and **peak-slice**
(highest quintile = collapse depth).

`ecl` is the flagship (Part 5: the replay‑shaping mechanism needs an off‑policy
buffer, and HASAC's entropy gives the identifier its excitation and the tracked
optimum its uniqueness/smoothness). `ecl_happo` gets `[E]` only — report it as the
degraded variant, not ECL.

## What to watch (`<run_dir>/ecl_debug.csv` — same data on TensorBoard `ecl/*`)

Every run writes `ecl_debug.csv` (columns `env_step, c_now, c_max_seen,
pcr_payload, eps_mean, cond_number, trough_frac, reward_mean`), one row per
identifier refresh — no TensorBoard needed.

- `c_now` vs `pcr_payload` — the identifier must track the driver
  (**D1**: `corr > 0.95`, lag `< 5%` of period). This is the go/no‑go for [I].
- `ecl/cond_number` — the identifier's 2×2 design condition number (§3.2
  collinearity watch). If it sits above ~1e3, raise the SAC temperature floor.
- `ecl/trough_frac` — fraction of the buffer tagged trough (the anchor's supply).
- The **ratchet plot** (spec Stage E): per‑cycle trough‑slice return. ECL's should
  be **non‑decreasing** across cycles; blind's decays. That is the headline claim.

## Ant instantiation (exact, spec §6.1)

Effort coordinate = own **hip torque** (identifier `x1`/`S` from stored joint
actions; envelope `x1`). Readout `y` = own **hip qvel one‑step delta**. On this
env's `ant.py` the hip qvels sit at obs indices **`[17,19,21,23]`** (the qvel
block starts 2 slots earlier than a stock Ant‑v2; confirmed by the identifier's
`scan_best_offset = -2` diagnostic — every agent's own‑effort response peaked 2
indices below the old `[19,21,23,25]`). `ρ = 0.8`, `W = 2000`, `h = 0.15·c_max`,
`β_A = 0.25`, `β_U = 0.10`, `c_low = 0.10·c_max`, envelope half‑lives 200/2000.

> **Readout‑index self‑check.** `ecl_debug.csv` logs `corr_x1y` (own effort ↔
> readout at the *configured* index), `scan_best_corr`, and `scan_best_offset`
> (offset to the best‑correlating obs coord). A correct index shows
> `corr_x1y ≈ 0.5` and `scan_best_offset = 0`. If `scan_best_offset ≠ 0`, shift
> `ecl_cfg.readout_qvel_idx` by that offset — a wrong index makes `b1≈0`, which
> sends the `b2/b1` estimate bang‑bang and pins the envelope near 1.

## HARL‑fit deviations (core theory preserved; documented per §)

- **Localization** = kernel‑weighted importance resampling from a uniform candidate
  pool (realizes the exact §3.3 Gaussian tag‑kernel in O(pool) without maintaining
  live per‑bin index lists under circular eviction).
- **Anchor** rehearses trough‑tagged transitions **from the main buffer** rather
  than a separate reservoir — faithful on PCR (the trough game is *stationary* and
  the 1e6 buffer always holds ≳1 payload cycle of trough experience; the
  separate‑reservoir machinery guards rare/drifting troughs, which PCR lacks).
- FP `state_type` not yet supported (mamujoco is EP); on‑policy `ecl_happo` is
  envelope‑only by design.
