# GRF-NS — Coupling Under Drift, instantiated in Google Research Football

The cell this instance fills, stated first because the spec says conflating the
cells is the one thing that is not publishable:

|  | **invertible channel** | no inverse |
|---|---|---|
| **(C) interaction-mediated** | **GRF-NS ← this instance**, `simple_ns` (VMAS) | `smac_ns`, `urb_ns` / `road_ns` |
| (B) exogenous | `ns_direct: 1`, the control | — |

> **(C, invertible) — identification AND compensation.** The disturbance is a
> rotation of a player's commanded compass heading, and rotations of GRF's eight
> headings form a group: every rotation has an exact inverse and nothing
> saturates in the action space. A correct estimate therefore *cancels* the
> disturbance rather than routing around it. This is the demonstration cell,
> where a return curve can fall a long way and be brought back.

## The story, and every clause is a requirement

> *Three attackers share one half of a pitch. A player running with a teammate
> in his lane has to go round him: he swerves off the line he wanted. How early
> and how wide he swerves depends on the footing — on a firm dry surface he
> checks and cuts late; on a wet, greasy, cutting-up pitch he cannot check, so
> he gives everyone more room, earlier. The surface changes over the match, and
> nobody on the pitch controls it. A player **alone** in that half never swerves
> round anyone, however wet it is.*

| object | GRF | check |
|---|---|---|
| medium | my running lane: the cone within 135° of my heading, weighted by proximity `κ(d) = 1/(1+(d/λ)²)`, λ = 0.12 (~6 m) | — |
| occupancy | teammates in the lane, strictly `j ≠ i` | gate 3, asserted |
| operator `W[i,j]` | `recv_i · κ(d_ij) · 1[j in i's lane]`, zero diagonal; **asym 0.133, spread 0.484, ratio 4.1×** at spawn with the ball carrier | NS-1.2 |
| `r` classes | the sector a teammate occupies: **front** (≤45°) or **flank** (45°–135°); behind is not in the lane. `r = 2`, independent of N | P-1.1 |
| `β*` | `σ·L·A(t)·send_m`, `send = [1.4, 0.6]` (mean 1) — how much a teammate in that sector costs **today**. Unknown to the agent | P-1.2 |
| driver `A(t)` | pitch condition: a `sin²` bump over the wet half of a 12000-step cycle, **exactly 0** for the dry half | NS-1.3, NS-2.5 |
| harm | `d_i = β*·Q_i / load_norm` compass steps of swerve per step, enacted through the host's own action interface by a first-order sigma-delta; **direction away from the traffic, public**; GRF's reward untouched | NS-1.4 |
| sensor | `y_i = k_i + c_i`: executed heading − sent heading, along the public direction. Proprioception at the actuator, exactly `simple_ns`'s residual | P-2.1 |
| inverse | `c_i = g·d̂_i`, pre-rotate; with `d̂ = d` the net is exactly 0.0 and the executed action **is** the commanded action | II.6 row 1 |

Identities that hold **exactly** and are asserted, not argued (`selftest.py`, 47
checks, numpy only, ~50 s; `smoke.py` re-checks the load-bearing ones through
the real engine at the ACTION level -- GRF does not reproduce a trajectory
across two engine instances even under `env.seed()`, so "executed == commanded
on every step" and a shadow blind channel fed the same positions are the
in-engine statements, and they are exact):

```
sigma = 0            ->  executed == commanded          the stock task, byte for byte
A(t) = 0 (dry half)  ->  executed == commanded          the placebo, every sigma
N = 1                ->  d == 0.0 at sigma = 3          category C, structurally
trust = 0            ->  pactoff == blind, bit for bit  the floor property, P-7.1
oracle (c = d)       ->  executed == commanded          the channel inverse, II.6
```

**Measured offline (synthetic squad, σ=2, μ=0.99):** `fit_gain 0.45–0.70`,
`beta_cos 0.999`, `beta_relerr 0.04–0.05`, `cond_psi 1.7–3.3`; the compensated
arm enacts **2.4× fewer** swerves than blind; the intercept arm enacts *more*
than blind on (C) (a fleet-mean correction is wrong per agent) and removes
**93 %** of them on the (B) control, where the full arm's `fit_gain` is −0.07.
That pair is the measurement the classification rests on.

## Who the agents are: controller slots, not players

GRF exposes `n` controller **slots**; which player a slot drives is the
engine's decision (`active` in the raw observation) and it changes: at
kick-off the engine hands a slot the goalkeeper and auto-switches it onto the
designated (ball) player on the first tick, and it switches again whenever the
ball reaches an uncontrolled teammate. The layer keys everything on the slot,
re-reads each slot's position every step, drops a slot's lane state when it
changes player (`reset_slot`, counted as `switches` in the panel), and takes
the references from the scenario's **declared** attacker geometry
(`ceiling.SPAWN`) rather than from the kick-off frame -- then checks that
geometry against the engine once after the first tick and warns on drift. The
smoke suite prints the assignment sequence; if the steady state ever includes
the keeper, the design (not the code) has to be revisited.

## Why the heading, and not the pace

Sprint is a two-level effort and a trained attacker sprints nearly always, so a
pace derate has no headroom for an inverse — it would put this instance in the
bounded cell. The headings are a group. And a swerve round a teammate *is* a
heading error, which is what makes the compensation claim available and honest.

## Why the driver cycles in 12000 steps, not 400

URB's driver unit is the *day*, not the step: the surface is effectively
constant within an episode (~100–150 steps for a competent 3v1 policy) and
drifts across ~100 of them. The clock persists across episodes (NS-3.4) and is
de-phased across rollout threads so a batch is a cycle average. This is also
what keeps the estimator honest: a 100-step RLS memory (μ = 0.99) averages the
sigma-delta quantisation noise **and** tracks the 6000-step bump with a ~10 %
lag. A 400-step cycle sat on the tracking floor whatever μ was set to (trap 9);
the fix was physical, not a tuned μ.

## Files

| file | what |
|---|---|
| `actions.py` | GRF's default action set, the compass group `rotate`, gate 4 |
| `driver.py` | `A(t)`, the dial, `DialParams`, `certify()` — the four NS-2 gates over the whole domain |
| `coupling.py` | the declared `W`, the sectors, the basis, the geometric references |
| `channel.py` | the per-step arithmetic: disturbance, sigma-delta, sensor, estimator (URB's core), inverse, arms |
| `layer.py` | the severity **mixin** over HARL's `FootballEnv`; hooks `step`, inherits `obs`/`reward`/`done` |
| `keys.py` | every yaml key the layer reads, gfootball-free |
| `ceiling.py` | Part C: loading distribution, N-scaling, the references to commit (→ `ceiling.json`) |
| `selftest.py` | offline conformance; no gfootball, no torch |
| `smoke.py` | in-simulator identities; needs gfootball |
| `calibrate.py` | the σ ladder against a trained B0 checkpoint, all five arms |
| `check_plumbing.py` | yaml ↔ dataclasses ↔ layer keys ↔ runner arms ↔ registries |

The estimator (`AgentRLS`, `rls_confidence_pred`) is imported from the
**verbatim vendored** URB core at `harl/envs/smac/smac_ns/pact1_core.py` — the
same object `smac_ns` uses. The one addition is BenchMARL's covariance-windup
bound (`p_trace_max`), applied to the estimator's state from outside.

## Runbook

**0. Plumbing and conformance (seconds, no simulator).** Both must end green.

```bash
python -m harl.envs.football.grf_ns.check_plumbing
```
```bash
python -m harl.envs.football.grf_ns.selftest
```

**1. Part C, committed before any method run (NS-4.1).** Already committed as
`ceiling.json`; rerun after any change to the dial constants.

```bash
python -m harl.envs.football.grf_ns.ceiling --out harl/envs/football/grf_ns/ceiling.json
```

**2. Smoke, in the engine (minutes).**

```bash
python -m harl.envs.football.grf_ns.smoke
```

**3. B0 — stock GRF, dial off.** The competent controller the ladder needs.
Every arm below uses the SAME tuned host flags (`tuned_configs/football/
academy_3_vs_1_with_keeper`), so define them once:

```bash
export TUNED="--n_rollout_threads 50 --num_env_steps 25000000 --use_linear_lr_decay True --ppo_epoch 15 --critic_epoch 15 --actor_num_mini_batch 2 --critic_num_mini_batch 2 --hidden_sizes 64,64 --share_param False --fixed_order False --eval_episodes 100 --n_eval_rollout_threads 50 --eval_interval 25"
```
```bash
python examples/train.py --algo mappo --env football --exp_name b0 --ns_on 0 --seed 1 $TUNED
```

**4. Calibrate σ against B0, then commit the operating point.**

```bash
python -m harl.envs.football.grf_ns.calibrate --run_dir results/football/academy_3_vs_1_with_keeper/mappo/b0/seed-00001-<stamp> --sigmas 0,0.25,0.5,1,1.5,2,3 --episodes 100 --threads 10 --phase peak --out ladder_peak.json
```

Run it once more with `--phase cycle --out ladder_cycle.json` (the cycle average is
what training sees).  `calibrate.py` gives every thread a fixed
`game_engine_random_seed` shared across arms, so the ladder is a paired
comparison (`env.seed()` is a no-op in gfootball).  Commit the chosen
`ns_severity` and the table into `football.yaml`.

**5. The arms.** Five, through the identical layer (P-9.1). `blind` is
`--algo mappo` inside the dial; `pactoff` is provably bit-identical to it and
additionally writes the panel.

```bash
python examples/train.py --algo mappo --env football --exp_name blind --seed 1 $TUNED
```
```bash
python examples/train.py --algo pactoff --env football --exp_name pactoff --seed 1 $TUNED
```
```bash
python examples/train.py --algo pact --env football --exp_name pact --seed 1 $TUNED
```
```bash
python examples/train.py --algo pact_oracle --env football --exp_name oracle --seed 1 $TUNED
```
```bash
python examples/train.py --algo pact_intercept --env football --exp_name intercept --seed 1 $TUNED
```

**6. The existing algorithms, inside the identical physics.** These are what
should fall.

```bash
python examples/train.py --algo happo --env football --exp_name happo_ns --seed 1 $TUNED
```
```bash
python examples/train.py --algo hatrpo --env football --exp_name hatrpo_ns --seed 1 $TUNED
```

**7. The (B) control** — the pair the classification rests on.

```bash
python examples/train.py --algo pact --env football --exp_name pact_B --ns_direct 1 --seed 1 $TUNED
```
```bash
python examples/train.py --algo pact_intercept --env football --exp_name intercept_B --ns_direct 1 --seed 1 $TUNED
```

**8. Ablations.**

```bash
python examples/train.py --algo pact --env football --exp_name abl_mu995 --ns_mu 0.995 --seed 1 $TUNED
```
```bash
python examples/train.py --algo pact --env football --exp_name abl_meanpres --ns_mean_preserving 1 --seed 1 $TUNED
```
```bash
python examples/train.py --algo mappo --env football --exp_name mappo_residual --ns_observe_residual 1 --seed 1 $TUNED
```

Run 5 seeds per arm; compare host to host (`pactoff` vs `pact`, `pact_happo_off`
vs `pact_happo`).

## Reading `pact_debug.csv`

II.10: "is the method working" and "is it winning" are separate columns.

| column | question | healthy |
|---|---|---|
| `dial_live`, `harmed_steps`, `rot_sent` | **did the dial fire at all?** | non-zero; a σ>0 run with `dial_live = 0` refuses to be a severity arm |
| `sigma` / `A` / `amp` / `placebo` | is the dial live, and does it switch off? | `placebo ≈ 0.5` over a cycle |
| `d`, `k_abs`, `y` | is the NS biting? | non-zero outside the placebo |
| `fit_gain` | does the reduction hold here? | rises, stays up |
| `beta_cos`, `beta_relerr` | is β *identified*, or only used? | cos → 1 (scored only while A > 0.25) |
| `cond_psi` | can β be decomposed? | low; **warn only** |
| `trust_pol` / `trust_app` | configured vs applied reliance | together; divergence = the gate misfiring |
| `pred_err`, `corr_abs`, `corr_clipped` | how much was cancelled, and did the relief valve bind? | small / rare |
| `x_std`, `spread` | channel liveness; the commons (does compensation bunch the squad?) | non-zero / logged |
| `win_rate`, `ep_return` | is it winning? | — |

## Honest limits

1. **σ = 1 is a stated calibration, not a published anchor.** Football has no
   constant for "how much wider you swerve on a wet pitch"; 0.14 step/step at
   the peak at spawn geometry is a *procedure*, carried over from the HCM
   figure so ladders are comparable. Every σ > 1 is a beyond-physical stress
   test.
2. **Learned trust (P-6.2) is not implemented** — the correction is a
   deterministic transform of the sampled action, so `∂ log π / ∂g = 0`. Arms
   are `off` / `fixed` / `oracle` / `intercept`; `pact_fixed` and `pact`
   coincide. `simple_ns` has the same limitation.
3. **The Part-C endpoint carries no information** (100 % peer by
   construction: no uncontrolled participant loads a lane). Quote the
   N-scaling curve (`ceiling.json`: mean d 0 → 0.071 → 0.143 for N = 1, 2, 3).
4. **The oracle uses full trust (g = 1) and the same relief valve** as every
   other arm, so it is B0 wherever `d ≤ corr_clip` and shows σ* where the valve
   binds. `pact` keeps P-5.1's 0.9.
5. **The driver's shape is injected.** Pitch condition is what a practitioner
   names; the `sin²` cycle is ours. Say so.
6. **A rotation persists one step through a ball action or a toggle** (the
   layer never overrides a pass, shot, sprint or dribble command); it is
   counted as `stale` and is rare.
7. **Handedness is checked in the engine, not assumed** (`smoke.py`); the
   spawn table in `ceiling.py` is cross-checked there too.
