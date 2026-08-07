# O-MAX — the unfair-advantage ceiling ladder

**Measurement instruments, not a deployable method.** Each rung hands the learner a
privileged quantity (the env's true disturbance `d`) **and** the machinery engineered
to exploit it, to find where the reachable return at σ=0.45 actually tops out. The
headline number is the **eval return** in `progress.txt`.

Why this exists: the `ANT_PCR_ORACLE` conditioning arm reached ~4k — *below* blind
(~5k) — which only proved that **handing HAPPO information doesn't make HAPPO use it**
(and the info was scrambled by per-step normalization). An oracle *arm* is a lower
bound, not a ceiling. These rungs deliver the privileged quantity through mechanisms
built to exploit it — chiefly **hardwired feed-forward cancellation** — so their top
is the real ceiling.

## How it works (no custom runner, no method)

Everything is `--algo happo` + env-args. One env wrapper (`OmaxMujocoMulti`) does it:

- **`comp_beta`** — before each step, execute `u = a − β·d_true` using the env's own
  `pcr_d_next`. The *delivered* torque is then `clip(clip(a−d)+d) = clip(a)` on the
  unsaturated set: **the training env becomes byte-equivalent to the stationary Ant.**
  So rung O1 is literally *HAPPO on the stationary channel*, and its return is the
  reachable ceiling. (β=1 exact; the β-grid {0.5,0.75,1} is the O0 sweep.)
- **`aug_actor` / `aug_critic`** — append `[A(t), d·d_scale]` to the actor obs / critic
  share_obs **after** MujocoMulti's normalization, in torque units (fixes the
  `ANT_PCR_ORACLE` whitening confound). Each is a list of `["payload","d"]`.
- **`std_floor`** (actor) + **`model_dir`** (warm-start) — rung O4's extra advantages.

`ant.py` is **untouched** — it already exposes `pcr_d_next` / `pcr_payload` in `info`.

## The ladder

| Rung | Config | Advantages (cumulative) | Steps | Expected if theory holds |
|---|---|---|---|---|
| O1 | `o1.json` | `comp_beta=1`, from scratch | 10M | ≈ stationary (~7k − ≤1% ctrl-cost wedge) |
| O2 | `o2.json` | + `aug_actor=[payload,d]` | 10M | ≥ O1 |
| O3 | `o3.json` | + `aug_critic=[payload,d]` | 10M | ≥ O2 |
| O4 | `o4.json` | + warm-start (O1 ckpt) + `std_floor=0.05` | 5M | the absolute reachable number |

## Run order

**O1 is the decisive test — run it first.** It needs no checkpoint and answers the
whole question:

```bash
cd examples
python train.py --load_config ../tuned_configs/mamujoco/Ant-v2-4x2/omax/o1.json
```

- **If O1 → ~6.5–7k**: the ceiling is real and learnable. Perfect cancellation during
  training recovers the stationary return, so the disturbance IS compensable, and the
  deployable problem reduces to **estimating `d`** (which is what a real method must
  do — the O1 wrapper's `d_true` gets replaced by an estimate). 6.5k is reachable.
- **If O1 ≪ 7k** but O4 ≈ 7k: the gap is optimization-under-drift, not information —
  run O2/O3 to attribute which advantage closes it.
- **If even O4 ≪ 7k**: RL cannot exploit even a perfect, hardwired, warm-started
  advantage → the binding constraint is training-under-drift itself; investigate
  value/normalization churn before any method.
- **If O1 ≈ 5k**: the strong gait's disturbance cannot be canceled to 7k at σ=0.45 →
  **7k was never reachable; the ceiling number replaces 7k as the target for every
  future method** (grade against the ladder top, not the stationary return).

O2/O3 are attribution rungs — run them only if O1 lands meaningfully below O4. For O4,
set `train.model_dir` to O1's `.../models` checkpoint dir (O4 has no obs augmentation,
so its input layer matches O1's — no loader shim needed). O0 (scripted β-grid on a
frozen walker) = evaluate O1's checkpoint with `comp_beta ∈ {0.5,0.75,1.0}`; its number
should bracket O1's from above.

## Reading `omax_debug.csv` — why a rung tops out where it does

Every rung writes `omax_debug.csv` (one row per rollout) built to answer two
questions directly. **Q2 first, then Q1, then the reward decomposition:**

**Q2 — is the mechanism using the RIGHT information?**
- `timing_err_max` = `|d_used − d_applied|`. The compensation cancels `d_used`
  (cached from last step's `pcr_d_next`); the env applies `d_applied`. These must
  be the same d. **Healthy: ≈ 0** (< 1e-9). If it's nonzero, the comp is cancelling
  the wrong step's disturbance — a timing/reset bug, and every ceiling number is
  void until it's fixed.

**Q1 — is the disturbance actually removed from what the policy experiences?**
- `residual_absmean` = `|delivered − clip(a_intended)|`, and its `_peak`/`_trough`
  split. This is the disturbance the policy STILL feels after compensation.
  **Healthy: ≈ 0** ⇒ the policy trains on the stationary env exactly (the info is
  fully delivered). **If `residual_peak ≫ residual_trough`**, the comp is leaking
  at high load — look at `u_clip_frac` (compensated command railing) and
  `a_absmax` (policy driving the rails): that is the saturation frontier, and it
  means the strong gait's disturbance *cannot* be fully cancelled at σ=0.45.
- `d_absmean/absmax` — the disturbance magnitude there is to cancel (context).

**If Q1 and Q2 are both clean (residual≈0, timing_err≈0) but the return is still
low, the info IS delivered and used correctly — so the ceiling is genuinely that
number**, and the reward decomposition says why:
- `r_forward` low → the gait itself is weak (decentralized 4×2 optimization).
- `r_ctrl` large / cycling `peak` vs `trough` → the control cost of compensating
  (`u = a − d`) is eating the return and drifting the reward with the payload.
- `r_survive` low / `ep_len` short → the Ant is falling over.
- `actor_std` collapsed → exploration died (why O4 adds a std floor).

Decision tree: `timing_err`≠0 → bug (fix first). `residual_peak`≫0 → frontier
(can't cancel the strong gait; the σ=0.45 target is too hard for THIS gait).
Both≈0, return low → the reward decomposition names the binding term, and that
number is the true ceiling.

## The ≤1% wedge (don't chase it)

The env charges ctrl-cost on the received action `u = a − d`, so O1–O4 pay
`0.5‖a−d‖²` ≈ the stationary cost plus a small positive wedge (~<1.5% of return on the
coordinated gait). Also, when a compensated command rails (`|a−d|>1`) it leaks residual
`≤ |d|` (logged as `omax_u_clip`). **O1 landing at ~6900 instead of 7000 is consistent
with the theory, not a failure.**

## Validate the mechanic first (seconds, offline)

```bash
python -m harl.envs.mamujoco.omax.test_omax     # expect T1..T4 PASS, V0 PASS
```

T1 proves `delivered == clip(a)` (the env is made stationary); T2 the cache timing +
episode-reset; T3 that `comp_beta=0` is bitwise the blind env and β<1 leaves `(1−β)d`;
T4 the saturation leak bound. If T1 fails, the ceiling numbers are meaningless — do not
launch a training rung.

## Prohibitions (from the spec)

Reward untouched; host hypers frozen across rungs (only the listed advantage knobs
differ); every rung prints an `[ORACLE ARM][O<k>]` banner; these arms are **never** the
headline method — the deployable method is specced separately against whichever rung
the ladder selects as the ceiling.
