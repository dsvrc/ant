# ECHO-R in HARL (PCR non-stationary Ant)

ECHO-R is a **driver-estimation + conditioning layer** for the category-C
Payload-Coupled Chassis-Reaction (PCR) non-stationarity. Each agent injects a
microscopic, zero-mean orthogonal probe into the exact action coordinate the PCR
coupling reads (its leg's **hip** torque), demodulates its own local readout
against (a) the *others'* codes — which can only have reached it through the
liability channel — and (b) its *own* code, and takes the **ratio** to recover a
self-calibrating estimate `ĉ ≈ c(t) = A(t)·σ` of the hidden severity. That one
scalar is appended to the observation; the host MARL algorithm is otherwise
**unchanged** and learns a `ĉ`-conditioned policy (spec: `ECHO-R_implementation_spec.md`).

## Integration (why there is almost no host-code change)

The whole method lives in an **env wrapper**, `EchoRMujocoMulti` (the spec's
preferred HARL slot, Part 4). It subclasses `MujocoMulti` and, inside
`step`/`reset`:

1. injects the probe into the hip action *before* the sim (so it lands in the
   commanded torque both the simulator and the PCR coupling `S_i` read);
2. reads out each agent's own hip joint-velocity delta and demodulates it into
   `ĉ` (pure numpy, `EchoRAdapter`);
3. appends `ĉ` to the observation and the shared/critic state.

Because the probe is added *inside* `step`, the host runner passes its raw
(pre-probe) action to the env and stores exactly that — the buffer keeps the
**pre-probe** action automatically (spec 5.2), and the actor/critic/buffer are
sized from the wrapper's already-augmented spaces. Nothing in `ant.py`, the
runner, or the learner is modified. The probe runs at **both train and eval**
(the eval envs are wrapped too), satisfying train/exec symmetry automatically.

`ant.py` (the PCR NS) is **not touched** — deploy it on the run machine exactly
as in the PCR doc, blind (`ORACLE=False`).

## Run

Two backbones (both solve the *same* wrapped env; the algo name only selects the
learner):

```bash
# ECHO-R on HAPPO (on-policy)
python examples/train.py --load_config \
    tuned_configs/mamujoco/Ant-v2-4x2/echor/config.json \
    --exp_name mujoco_ant_echor_happo

# ECHO-R on HASAC (off-policy)
python examples/train.py --load_config \
    tuned_configs/mamujoco/Ant-v2-4x2/echor_hasac/config.json \
    --exp_name mujoco_ant_echor_hasac
```

Enabling is via `env_args.echor: true` (+ `env_args.echor_cfg`) in those JSONs.
Use `agent_conf 4x2` (agent == leg) — the reference configuration.

## Status / what this chat taught us (read before running)

This is **pure ECHO-R** — probe → demodulate → append `ĉ` → the host learns a
`ĉ`-conditioned policy. It does **not** cancel anything (spec P1 / Prohibition 9;
the explicit-cancellation experiment we tried destabilised training and was
reverted). Improvements kept from the debugging in this chat:

- **SNR gate (`snr_gate`, default 1.0)** — holds `ĉ = 0` on any agent whose
  direct-channel SNR `|E[G]|/std[G] < snr_gate`. So when the decentralised probe
  can't detect the driver, ECHO-R degrades to **blind (never worse)** instead of
  feeding the policy noise. Set `snr_gate: 0` for the literal spec (no gate).
- **SNR diagnostics** (`snr_G`, `snr_H`) in the debug trace — the at-a-glance
  read on whether the estimator is actually working.
- **c-oracle D0 mode** (`c_oracle: true`) — condition on the *true* driver from
  `info` (the sanctioned D0 hygiene exception) to test conditioning-sufficiency.

**The severity must be recoverable for any of this to matter.** We found the
original `SEVERITY=4` was *past the actuator-saturation boundary* (peak `|d|`
far exceeds ±1), so nothing — blind, ECHO-R, or oracle — can recover; the
recoverable ceiling is `SEVERITY ≈ 1.8` and `ant.py` is now calibrated to **1.5**
(verified: peak mean `|d| ≈ 0.39`). Run ECHO-R only at that calibrated severity.

**Recommended workflow (in order):**

1. **D0 gate — run first.** `--c_oracle True`: does conditioning on the *true*
   driver beat blind at `SEVERITY=1.5`?
   - beats blind → conditioning is sufficient; go to step 2 (make the estimator
     detectable).
   - ≈ blind → the scalar driver alone isn't enough for this NS (the per-joint
     load matters); report honestly — no scalar estimator can beat this bound.
2. **Make the probe detectable.** In the debug trace, watch `snr_G`. With the
   default `ε` it sits well below 1 (the hip-qvel readout is contact-noisy), so
   the gate keeps `ĉ = 0` (= blind). Sweep `ε` up (0.03 → 0.05 → 0.08) until
   `snr_G > 1`; then the gate opens and `ĉ` starts tracking `pcr_payload` (D2).
   Keep the smallest `ε` that clears 1 (larger `ε` disturbs the gait — check D1).
3. **Headline.** blind vs ECHO-R vs c-oracle at `SEVERITY=1.5`, identical host
   configs (Part 8, Stage E).

```bash
# D0: conditioning-sufficiency at the recoverable severity
python examples/train.py --load_config tuned_configs/mamujoco/Ant-v2-4x2/echor/config.json \
    --exp_name echor_d0_coracle --c_oracle True
# ECHO-R proper (sweep eps until snr_G>1 in the debug trace)
python examples/train.py --load_config tuned_configs/mamujoco/Ant-v2-4x2/echor/config.json \
    --exp_name echor --eps 0.05
```

## The ONE thing to verify on the run machine (spec 6.1)

The readout reads each leg's **hip joint velocity** from the raw Ant
`_get_obs()`. For the standard layout `[qpos[2:] (13), qvel (14), ...]` the hip
velocities are at obs indices **19, 21, 23, 25** (default `echor_cfg.readout_qvel_idx`).
This is correct for the stock Ant / PCR `ant.py`. If your `ant.py` uses a
different obs layout, set `echor_cfg.readout_qvel_idx` accordingly (the wrapper
prints a warning if an index is out of range). The injection targets the hip
(even) action coordinate; override via `echor_cfg.inject_mask` if needed.

## Calibration & knobs (spec Part 7 — two real knobs)

* `eps` (default `0.01`) — probe amplitude. Gait torques ≈ 0.2, so the probe is
  ~5% of the operating scale; its reward cost is `O(ε²)` ≈ 5·10⁻⁵/step. Sweep
  `{0.5×,1×,2×}` if the tracking SNR is weak (Stage-E ablation iv).
* `lam_halflife` (default `1500` steps) — demodulator EMA half-life. Raise it in
  low-SNR regimes (the driver is slow, so the budget exists).

Everything else (`T_chip=3`, `L=127`, `delta_shift=16`, `K=15`, `rho_hat=0.8`,
`W_anchor=100000`, anchor quantile `0.05`, `center_halflife=200`) is a derived
constant — leave it unless you change the code structure. `rho_hat=0.8` is the
PCR structural leak (`KNOWN_RHO`); the method assumes only the *class* of the NS,
never its hidden state/phase/severity.

## Diagnostics (emitted in `info`, never fed back)

Per step the wrapper writes to `info`: `echor_chat_mean` (mean `ĉ`),
`echor_chat` (per-agent), `echor_G_absmean`, `echor_H_absmean`. The PCR `ant.py`
already emits `pcr_payload`, `pcr_load`. For the spec's **D2 tracking test**,
overlay `echor_chat_mean` against `pcr_payload · SEVERITY` — expect `corr > 0.9`
once the estimator has warmed up (a few thousand steps). Note: at the very first
evaluation the per-thread eval estimators are still warming up, so early eval
points are slightly pessimistic; the training-env curve is the clean signal, and
late-training eval is fully converged.

## Debug trace + calibration workflow (find out *why* it isn't working)

Set `echor_cfg.debug: true` (already on in the tuned configs) to write a focused
per-step CSV trace. Each env instance writes its **own** file
`./echor_debug/echor_debug_pid<PID>_inst<N>.csv` (no cross-process contention);
the path is printed at startup. The columns are the ones that actually explain a
non-working run:

| column | meaning | what it tells you |
|---|---|---|
| `pcr_payload` | ground-truth driver `A(t)` | the target `c_hat` must track (ramps up over ~`0.2·_P` steps) |
| `pcr_load` | true mean liability `|d|` | is the NS even active yet? peak should reach ≈0.3–0.5 |
| `c_hat_i` | per-agent estimate fed to the policy | does it track `pcr_payload`, or is it noise? |
| `H_i` | echo-path gain `Ĥ` | ≈0 ⇒ probe isn't reaching the liability |
| **`snr_G_i`** | **direct-channel SNR `\|E[G]\|/std[G]`** | **`< 1` ⇒ the agent can't detect its own probe → ratio is noise → `c_hat` meaningless. This is the root-cause readout.** |
| `snr_H_i` | echo-channel SNR | `> 1` ⇒ the estimate is real |
| `reward`,`done` | step reward / episode end | collapse vs recovery |

**What the first run showed (why nothing improved):** `pcr_payload` was still
~0 (payload ramping), yet `c_hat` was 2–10 and hitting `c_clip` — i.e. the policy
was fed *noise*. Cause: `snr_G < 1` (the `eps=0.01` probe is buried under the
±6 `qvel`-delta gait/contact noise), so `H/G` is garbage. Two mechanisms now
guard against this:

- **`snr_gate` (default 1.0)** — `c_hat` is held at **0** on any agent whose
  `snr_G < snr_gate`, so a noise-dominated probe degrades to *blind* (never
  worse) instead of injecting garbage. With `eps=0.01` this means `c_hat≈0`
  until you raise `eps`; that is expected and visible in the trace.
- higher default `eps=0.03` and `lam_halflife=4000` to give the channel a chance.

**Calibration workflow (do these in order):**

1. **D0 / c-oracle gate — run this first.** Set `echor_cfg.c_oracle: true`: the
   policy is conditioned on the *true* `pcr_payload` instead of the estimate.
   - if this **beats blind** → conditioning on the driver works, so it *is* worth
     driving the estimator's SNR up (step 2);
   - if this **≈ blind** → conditioning doesn't help here (check that
     `pcr_load` peaks ≈0.3–0.5; if not, the NS is too weak — raise `SEVERITY` in
     `ant.py` per the PCR doc). No estimator can beat this bound.
2. **Drive `snr_G` above 1.** In a focused run (below), watch `snr_G_i`. Sweep
   `eps` up (0.03 → 0.05 → 0.1) and/or `lam_halflife` up (4000 → 8000). When
   `snr_G > 1` the gate opens and `c_hat` starts tracking `pcr_payload`. Watch
   the probe cost: `eps` too high disturbs the gait (return drops at
   `SEVERITY=0`); keep the smallest `eps` that gives `snr_G > 1`.
3. **Confirm tracking (D2).** With the gate open, `c_hat_0..3` should rise/fall
   with `pcr_payload` (`corr > 0.9`).

```bash
# focused, per-step debug session -> two CSVs (train + eval)
python examples/train.py --load_config \
    tuned_configs/mamujoco/Ant-v2-4x2/echor/config.json --exp_name echor_debug \
    --n_rollout_threads 1 --n_eval_rollout_threads 1 --debug_interval 1 --eps 0.05
```

If `snr_G` cannot be pushed above 1 without an `eps` that visibly disturbs the
gait, the decentralized-probe approach is SNR-limited on this contact-rich
readout — report the c-oracle bound honestly (spec Part 8 D0). Set `debug: false`
for clean production runs.

## Expected result (spec Part 8, Stage E)

`stationary ≈ d-oracle ≥ ECHO-R ≫ blind`. ECHO-R should recover a large fraction
of the (oracle − blind) gap and show a materially smaller residual ripple against
the payload phase than the blind policy.
