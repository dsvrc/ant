# PCR failure diagnostic (plain HAPPO, no ECHO-R)

A read-only logging wrapper (`PcrDiagMujocoMulti`) that changes **nothing** about
the dynamics/action/obs/reward — it is plain HAPPO on the PCR ant — and records
per step exactly how and where the blind policy fails. Enable via
`env_args.pcr_diag: true`.

## Run

```bash
# focused single-thread trace (one clean CSV, every step)
# (--interval overrides pcr_diag_cfg.interval; --n_rollout_threads etc. work too)
python examples/train.py --load_config tuned_configs/mamujoco/Ant-v2-4x2/happo_diag/config.json \
    --exp_name ns_happo_diag --n_rollout_threads 1 --n_eval_rollout_threads 1 --interval 1

# or the full 20-thread run (each worker writes its own CSV; interval 10)
python examples/train.py --load_config tuned_configs/mamujoco/Ant-v2-4x2/happo_diag/config.json \
    --exp_name ns_happo_diag
```

CSVs land in `./pcr_diag/pcr_diag_pid<PID>_inst<N>.csv` (path printed at startup).
The terminal step of every episode is always logged (even when subsampling).

## Columns

| column | meaning |
|---|---|
| `step_global`, `ep_step` | global step; step within the current episode (resets on done) |
| `done`, `fall` | episode ended; ended by a **fall** (height out of `[0.2,1.0]`) vs a time-limit truncation |
| `pcr_payload` | driver `A(t)` ∈ [0,1] (ramps to peak over ~0.2·40000 ≈ 8000 steps, then sheds) |
| `pcr_load`, `pcr_loadmax` | mean / max parasitic load `|d|` |
| `d_app_mean`, `d_app_max` | mean / max `|d|` actually **applied** this step |
| `torso_height` | torso z; the env terminates when it leaves `[0.2, 1.0]` |
| `sat_frac` | fraction of joints where `|τ + d| > 1` → **actuator saturated by the load** |
| `tau_absmean`, `delivered_absmean` | mean commanded vs mean delivered `|torque|` |
| `reward` | total step reward |
| `r_forward`, `r_ctrl`, `r_contact`, `r_survive` | reward decomposition (ctrl/contact are ≤ 0) |

## How to read the failure (what to look for)

1. **Is it falling or just walking slowly?** Group by `fall`/`ep_step`. Short
   episodes ending in `fall=1` ⇒ it tips over (catastrophic). Long episodes with
   low `r_forward` ⇒ it survives but barely moves (soft failure).
2. **Is the load overwhelming the actuators?** Watch `sat_frac` and
   `d_app_max`. `sat_frac` rising toward the payload peak ⇒ the disturbance
   exceeds the ±1 authority — the "information-recoverable" claim breaks here
   (even the oracle can't cancel what it can't command).
3. **Does the collapse track the payload?** Overlay `reward` / `torso_height`
   against `pcr_payload`. If falls cluster near `pcr_payload` peaks, it's the
   NS; if they're phase-independent, it's a training issue.
4. **Which term drives the return down?** Compare `r_forward` (moves less),
   `r_ctrl` (fights the load harder), and `r_survive` (goes to ~0 on the fall
   step — losing the survival stream is usually the dominant collapse term).

Read these together to classify the failure: *saturation-driven tipping at
payload peak* (sat_frac↑, height→bound, fall=1, r_survive lost) vs *soft
slow-down* (sat_frac≈0, height ok, r_forward↓). That distinction decides whether
any driver-conditioned method could even help.
