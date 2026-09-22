# ANT-NS — Coupling Under Drift, instantiated in MAMuJoCo Ant

The cell this instance fills, stated first because the spec says conflating the
cells is the one thing that is not publishable:

|  | **invertible channel** | no inverse |
|---|---|---|
| **(C) interaction-mediated** | **ANT-NS ← this instance**, `simple_ns` (VMAS), `grf_ns` | `smac_ns`, `urb_ns` / `road_ns` |
| (B) exogenous | `ns_direct: 1`, the control | — |

> **(C, invertible) — identification AND compensation.** The disturbance is an
> unmodelled torque at the agent's own joints, so it lives in exactly the space
> its action lives in: a correct estimate **subtracts** it, and the executed
> torque, the reward and the trajectory become the σ = 0 ones **bit for bit**.
> With VMAS point masses and GRF's discrete heading group, this is the third
> action space in which the same method cancels the same form — continuous
> joint torque, the highest-dimensional of the three.

## Why this is the cheapest instance in the paper to defend

In URB the medium is a road network, in SMAC an enemy line, in GRF a running
lane — structures we had to **name** before we could use them. Here the medium
is the robot: **four legs bolted to one torso**. Drive one and the trunk reacts;
every other leg hangs off that same trunk and feels it. That coupling is in
stock Ant at every severity including zero.

**What the dial injects is therefore not the coupling — only the drift in how
strongly the structure transmits.** That is the design guide's `C_i[θ(t)]` with
`F_i` untouched, and the lone-agent projection stays exactly stationary because
every sum runs over `j ≠ i`.

## The story, and every clause is a requirement

> *Four legs are bolted to one torso. When another leg drives hard the trunk
> reacts, and that reaction arrives at my hips as a torque I never commanded: I
> am holding against my neighbours. How much of a neighbour's push reaches me
> depends on the state of the drivetrain — a cold machine is stiff and crisply
> preloaded; over a run the gearbox and bearing grease warm, viscosity falls,
> preload drops and backlash opens, and the same push arrives through a softer,
> sloppier path. Nobody on the robot decides how warm it is. A leg alone on the
> torso — the others slack — feels nothing from the others, at any temperature.*

| object | Ant | check |
|---|---|---|
| medium | the trunk every leg is bolted to; element = a joint | — |
| operator `W[p,q]` | `recv_p · κ(p,q) · 1[agent(p) ≠ agent(q)]`, with `κ` = the machine's OWN nominal-pose `|M⁻¹|` (`operator.json`). **zero-diag, spread 0.805, ratio 29.5×, asym 0.194, 24 live links of 56** | NS-1.2 |
| `r` classes | the load path by joint type: **hip←hip / ankle←ankle / cross**. `r = 3`, independent of N *and* of the number of joints; **cross is pruned** (P-3.4) because the machine's own operator makes it exactly zero → `r_live = 2` | P-1.1, P-3.4 |
| `β*` | `σ·L·A(t)·send`, `send = [1.5, 0.9, 0.6]` (mean 1) — what a neighbour's torque on each path costs **today**. Unknown to the agent | P-1.2 |
| driver `A(t)` | drivetrain thermal state: a `sin²` bump over the warm half of a 20000-step cycle, **exactly 0** for the cold half | NS-1.3, NS-2.5 |
| harm | an unmodelled torque on the actuators, applied at `do_simulation` — **below the reward** | NS-1.4 |
| sensor | `y_i = ⟨u_exec − u_sent, e_i⟩`: what the drive was asked for against what it delivered. Motor current, which every servo already measures | P-2.1 |
| inverse | `u_sent = a − c·e`; with `c = |d|` the executed torque **is** the commanded one | II.6 row 1 |

Identities that hold **exactly** and are asserted, not argued (`selftest.py`, 48
checks, numpy only, ~60 s; `smoke.py` re-checks them through MuJoCo at the
**trajectory** level — MuJoCo is deterministic, so unlike `grf_ns` these are
bit-for-bit in the engine too):

```
sigma = 0             ->  executed == commanded          the stock task, byte for byte
A(t) = 0 (cold half)  ->  executed == commanded          the placebo, every sigma
lone agent            ->  d == 0.0 at sigma = 3          category C, structurally
trust = 0             ->  pactoff == blind, bit for bit  the floor property, P-7.1
oracle (c = |d|)      ->  executed == commanded          the channel inverse, II.6
```

The last one is exact only because the channel forms the **net** actuator error
`d − c·e` *before* adding it to the command. Writing it as `(a − c·e) + d` is the
same number in exact arithmetic and differs by an ulp in floating point, which
would have cost the ceiling identity.

**Measured offline** (synthetic gait, σ=2, μ=0.99, 4x2, machine's own
operator): `fit_gain 0.999`, `beta_cos 1.000`, `beta_relerr 0.004`, `cond_psi
2.6`. Mean `|executed − commanded|` over the warm peak: **blind 0.0456 → pact
0.0036** (12.7×), intercept 0.0186, against a disturbance of 0.0464. The (B)/(C)
pair, as a ratio of residuals `intercept / full`: **(C) 5.20× — the peer channels
are load-bearing; (B) 0.96× — they carry nothing.** That pair is the measurement
the classification rests on.

## NS-4.2: the N-scaling is *runnable* here, and that is the headline

`r = 3` is independent of the partition, so the identical dial and the identical
basis serve every MAMuJoCo Ant partition. Splitting a leg's hip from its own
ankle (`8x1`) moves the strongest load path in the machine out of "solvable
alone" and into "requires coordination" **without making the task lossier**:

| `agent_conf` | N | \|d\| per actuator | machine's coupling crossing an agent boundary |
|---|---|---|---|
| `1x8` | 1 | 0.0000 | 0.0 % — a lone agent, structurally |
| `2x4` | 2 | 0.0444 | 66.2 % |
| `2x4d` | 2 | 0.0502 | 67.6 % |
| **`4x2`** | **4** | **0.0551** | **100.0 %** |
| `8x1` | 8 | 0.0440 | 100.0 % — **saturated** |

Every row from N=2 down is a **real training config**, so the prediction is an
experiment here, not only an offline curve. (`1x8` is the structural N=1
projection — MAMuJoCo builds no one-agent Ant.)

**It saturates at `4x2`, and the reason is exact.** The only pair of joints
inside a `4x2` agent is its own hip and its own ankle — and on the machine's own
operator that load path carries **identically zero** (hip axes are vertical,
ankle axes are not, so no hip torque accelerates any ankle at the nominal pose).
So by `4x2` every coupled pair already crosses an agent boundary, and `8x1` can
add nothing. That makes `8x1` a **control with a sharp prediction — no further
change** — which is a stronger claim than "more agents is worse", and the
informative range of the experiment is `2x4` → `4x2` (66 % → 100 %).

`ceiling.py` asserts the coupling share is non-decreasing and that `|d|` rises
with it *up to saturation*; it prints the saturation point rather than
pretending the curve continues.

> Read `|d|` **per actuator**, never per agent. The per-agent vector norm grows
> with the number of joints an agent owns, so the per-agent number *falls* with
> N for the same per-actuator error and would report this prediction as refuted.
> `ceiling.py` asserts monotonicity on the per-actuator figure.

## The hook, and why it is below the reward

gym's `AntEnv.step(a)` calls `do_simulation(a, 5)` and *then* computes
`ctrl_cost = 0.5·|a|²` from the **policy's own** `a`. Wrapping `do_simulation`
therefore puts the disturbance on the actuators while the reward function keeps
its own arguments:

- **NS-1.4 is satisfied literally** — the reward function is byte-for-byte the
  host's own, nothing is subtracted, and the ant earns less only because it
  physically walked worse. `smoke.py` asserts `reward_ctrl == −0.5·|a|²` to 1e-9
  with the dial live, which is the check that separates "removes capability"
  from "penalty term".
- Handing the disturbed torque to `step()` instead would feed the disturbance
  straight into `ctrl_cost` — a penalty term in disguise, and the single thing
  NS-1.4 forbids.

**No host file is edited** — not `mujoco_multi.py`, not gym's `ant.py`.
`do_simulation` is re-bound on the instance the layer owns.

## Who the agents are: a MAMuJoCo quirk worth knowing

MAMuJoCo's `obsk` tags each joint with an `act_ids` and declares Ant's `2x4` as
act-ids `[(2,3,4,5), (6,7,0,1)]`. But `MujocoMulti.step` **never reorders by
`act_ids`** — it concatenates the agents' action slices and hands the result
straight to MuJoCo, so agent `a` drives `ctrl[sum(dims[:a]) : sum(dims[:a+1])]`.
`structure.PARTITIONS` follows the **positional** mapping, because that is where
the torques land. Measured against `obsk`: identical leg groups for `4x2`,
`2x4d` and `8x1`, and for `2x4` the same "two adjacent legs per agent" shape
rotated by one corner — equivalent under the ant's 4-fold symmetry.
`check_plumbing.py` compares by structure and prints the relabelling;
`smoke.py` settles it by driving one agent and reading `sim.data.ctrl`.

`ant.xml` also does not list its actuators in leg order — the first two are
`hip_4`, `ankle_4`. A table written from leg names would aim every channel at
the wrong joint while every diagnostic looked healthy, which is why gate 4 reads
the actuator→joint map out of the loaded model and aborts on mismatch.

## Why the driver cycles in 20000 steps

Ant runs at dt = 0.05 s, so 20000 steps ≈ 17 minutes — the time scale a
drivetrain actually warms over. The clock persists across episodes (NS-3.4) and
is de-phased across rollout threads.

**Measured, because the obvious claim is not quite true:** `A` moves at most
`3.1e-4` per step, `0.031` over the estimator's own 100-step memory (μ = 0.99),
and up to `0.31` over a full 1000-step episode. So the surface is slow against
the control loop and against the estimator, but it is **not** frozen within a
long episode — a gearbox does warm measurably inside a minute of hard work. Say
that, rather than "constant within an episode".

## Files

| file | what |
|---|---|
| `structure.py` | the robot's declared body: joints, anchors, load paths, partitions, gate 4 |
| `driver.py` | `A(t)`, the dial, `DialParams`, `certify()` — the four NS-2 gates |
| `coupling.py` | the declared `W`, the `r = 3` classes, the basis, the geometric references |
| `channel.py` | the per-step arithmetic: disturbance, sensor, estimator, exact inverse, arms |
| `layer.py` | the severity **mixin** over `MujocoMulti`; hooks `do_simulation` |
| `keys.py` | every yaml key the layer reads, mujoco-free |
| `ceiling.py` | Part C: the partition, the N-scaling, the references (→ `ceiling.json`) |
| `dump_operator.py` | reads the machine's own `|M⁻¹|` out of the model (→ `operator.json`) |
| `selftest.py` | offline conformance; no mujoco, no torch |
| `smoke.py` | in-simulator identities; needs mujoco |
| `calibrate.py` | the σ ladder against a trained B0 checkpoint, all five arms |
| `check_plumbing.py` | yaml ↔ dataclasses ↔ layer ↔ runner ↔ every registry |

The estimator (`AgentRLS`, `rls_confidence_pred`) is imported from the
**verbatim vendored** URB core at `harl/envs/smac/smac_ns/pact1_core.py` — the
same object `smac_ns` and `grf_ns` use. The one addition is BenchMARL's
covariance-windup bound (`p_trace_max`), applied from outside.

## Runbook

**0. Plumbing and conformance (seconds, no simulator).** Both must end green.

```bash
python -m harl.envs.mamujoco.ant_ns.check_plumbing
```
```bash
python -m harl.envs.mamujoco.ant_ns.selftest
```

**1. Part C, committed before any method run (NS-4.1).** Already committed as
`ceiling.json`; rerun after any change to the dial constants.

```bash
python -m harl.envs.mamujoco.ant_ns.ceiling --out harl/envs/mamujoco/ant_ns/ceiling.json
```

**2. The machine's own operator (once, where MuJoCo lives).** Replaces the
geometric surrogate; the references change, so `ceiling.py` and the committed
`ns_load_norm` must be redone after it.

```bash
python -m harl.envs.mamujoco.ant_ns.dump_operator
```
```bash
python -m harl.envs.mamujoco.ant_ns.ceiling --out harl/envs/mamujoco/ant_ns/ceiling.json && python -m harl.envs.mamujoco.ant_ns.selftest && python -m harl.envs.mamujoco.ant_ns.check_plumbing
```

**3. Smoke, in the engine (minutes).**

```bash
python -m harl.envs.mamujoco.ant_ns.smoke
```

**4. B0 — stock Ant, dial off.** The competent controller the ladder needs (Ant
ships no scripted gait, and a random policy has nothing to lose). Host flags are
`tuned_configs/mamujoco/Ant-v2-4x2`, defined once:

```bash
export TUNED="--n_rollout_threads 20 --num_env_steps 10000000 --episode_length 200 --hidden_sizes 128,128,128 --ppo_epoch 5 --critic_epoch 5 --actor_num_mini_batch 1 --critic_num_mini_batch 1 --entropy_coef 0 --share_param False --fixed_order False"
```
```bash
for S in 1 2 3; do python examples/train.py --algo happo --env mamujoco_ns --exp_name b0 --ns_on 0 --seed $S $TUNED; done
```

**5. Calibrate σ against B0, then commit the operating point.**

```bash
python -m harl.envs.mamujoco.ant_ns.calibrate --run_dir results/mamujoco_ns/Ant-v2-4x2/happo/b0/seed-00001-<stamp> --sigmas 0,0.25,0.5,1,1.5,2,3 --episodes 20 --threads 10 --phase peak --out ladder_peak.json
```

Run it again with `--phase cycle --out ladder_cycle.json` (the cycle average is
what training sees), then commit the chosen `ns_severity` and the table into
`mamujoco_ns.yaml`.

**6. The arms.** Five, through the identical layer (P-9.1). `blind` is
`--algo happo` inside the dial; `pactoff` is provably bit-identical to it and
additionally writes the panel.

```bash
for A in happo pactoff pact pact_oracle pact_intercept; do python examples/train.py --algo $A --env mamujoco_ns --exp_name $A --seed 1 $TUNED; done
```

**7. The existing algorithms, inside the identical physics.** These are what
should fall.

```bash
for A in mappo hatrpo haa2c; do python examples/train.py --algo $A --env mamujoco_ns --exp_name ${A}_ns --seed 1 $TUNED; done
```

**8. The (B) control** — the pair the classification rests on.

```bash
python examples/train.py --algo pact --env mamujoco_ns --exp_name pact_B --ns_direct 1 --seed 1 $TUNED
```
```bash
python examples/train.py --algo pact_intercept --env mamujoco_ns --exp_name intercept_B --ns_direct 1 --seed 1 $TUNED
```

**9. The N-scaling, as training runs (NS-4.2).**

```bash
for C in 2x4 4x2 8x1; do for A in pactoff pact; do python examples/train.py --algo $A --env mamujoco_ns --exp_name ${A}_$C --agent_conf $C --seed 1 $TUNED; done; done
```

**10. Ablations.**

```bash
python examples/train.py --algo pact --env mamujoco_ns --exp_name abl_mu995 --ns_mu 0.995 --seed 1 $TUNED
```
```bash
python examples/train.py --algo pact --env mamujoco_ns --exp_name abl_meanpres --ns_mean_preserving 1 --seed 1 $TUNED
```

Run 5 seeds per arm; compare host to host (`pactoff` vs `pact`).

## Reading `pact_debug.csv`

II.10: "is the method working" and "is it winning" are separate columns.

| column | question | healthy |
|---|---|---|
| `dial_live`, `harmed` | **did the dial fire at all?** | non-zero; a σ>0 run with `dial_live = 0` refuses to be a severity arm |
| `sigma` / `A` / `amp` / `placebo` | is the dial live, and does it switch off? | `placebo ≈ 0.5` over a cycle |
| `d`, `dmax`, `y` | is the NS biting? | non-zero outside the placebo |
| `fit_gain` | does the reduction hold here? | rises, stays up |
| `beta_cos`, `beta_relerr` | is β *identified*, or only used? | cos → 1 (scored only while `A > 0.25`) |
| `cond_psi` | can β be decomposed? | low; **warn only** |
| `trust_pol` / `trust_app` | configured vs applied reliance | together; divergence = the gate misfiring |
| `pred_err`, `corr`, `corr_clipped`, `actuator_clipped` | how much was cancelled; did the relief valve or the actuator bind? | small / rare |
| `x_hh`, `x_aa`, `x_cross`, `x_std` | channel liveness, per load path | non-zero |
| `tau_rms` | the commons: is compensating making everyone push harder? | **logged, never acted on** |
| `ep_return`, `r_forward`, `r_ctrl`, `r_contact`, `r_survive` | is it winning — and did it *fall* or just slow down? | — |

## Before anything else: the host must be stock

The layer runs **gate 0** at construction and aborts if the installed gym
`ant.py` carries a coupling of its own. This is not hypothetical — this repo has
shipped a diagnostic Ant that announces itself with

```
[DIAG ENV] SEVERITY=0.45 FREEZE_A=None MASK=both ... RHO=0.8 P=40000 B=0.2
```

and applies its own peer coupling. Stacking ANT-NS on that would mean
`ns_severity: 0` is **not** the stock task, so B0 and every ladder row would be
measured against an already-disturbed baseline — entirely plausible numbers,
wrong experiment. Restore the stock 59-line `gym/envs/mujoco/ant.py`, or set
that env's own severity to 0, before running anything. `ns_allow_patched_host: 1`
exists only to experiment and makes a run unreportable; `check_plumbing.py`
fails if it is left on.

## Gate 4 compares distances, not coordinates

Ant's `reset_model` perturbs the free joint's quaternion, so the whole body sits
at a small arbitrary yaw — measured at a fresh reset, every joint rotated by the
same ~10°. The operator only ever uses **pairwise distances** (`κ` is a function
of distance alone), so that is what the gate compares: radii agree to 2 %,
distances to 3.6 %, kernel correlation **0.9997**, and the ordering the
structure rests on holds exactly (own ankle 0.439 > adjacent leg 0.288 >
diagonal leg 0.164). Comparing world-frame coordinates would fail on a robot
that is entirely correct.

## Two things the server measured, and what changed because of them

**The host must be stock.** The layer runs **gate 0** at construction and aborts
if the installed gym `ant.py` carries a coupling of its own. This is not
hypothetical — the first server run reported

```
[DIAG ENV] SEVERITY=0.45 FREEZE_A=None MASK=both ... RHO=0.8 P=40000 B=0.2
```

a diagnostic Ant from this repo's earlier PACT work, applying its own peer
coupling. Stacking ANT-NS on that would make `ns_severity: 0` something other
than the stock task, so B0 and every ladder row would be measured against an
already-disturbed baseline. Restore the stock `gym/envs/mujoco/ant.py` first.
`ns_allow_patched_host: 1` exists only to experiment and makes a run
unreportable; `check_plumbing.py` fails if it is left on.

**`recv` is declared, and the model refuted the first version of it.** An
earlier draft declared a hip/ankle susceptibility ratio of 1.86, arguing that an
ankle carries only the foot. The model was asked and answered **1.01** — Ant's
hips and ankles are equally easy to accelerate. The argument was wrong and was
removed rather than defended. What stands in its place is a declared
per-actuator heterogeneity (build tolerance and wear, `recv_spread = 0.35`,
mean 1), stated as injected rather than measured, exactly as `simple_ns` states
`recv_spread` and `smac_ns` its per-enemy sensitivity. It is the only source of
the operator's asymmetry, because the transmission structure itself is
symmetric.

## Gate 4 compares distances, not coordinates

Ant's `reset_model` perturbs the free joint's quaternion, so the whole body sits
at a small arbitrary yaw — measured on two fresh resets, every joint rotated by
the same amount, once −10° and once +9°. The operator only ever uses **pairwise
distances**, so that is what the gate compares: radii agree to 2 %, distances to
3.6 %, and the ordering the structure rests on holds (own ankle 0.439 >
adjacent leg 0.288 > diagonal leg 0.164). Comparing world-frame coordinates
fails on a robot that is entirely correct, and did.

## The operator is the machine's own — and it is not what a proxy would guess

`κ` is the joint-space inverse inertia `|M⁻¹|` at the nominal pose, read out of
the model by `dump_operator.py` and committed as `operator.json`. It is literally
"how much does a unit torque at q accelerate p" — the mechanical counterpart of
POWER's PTDF and URB's incidence-over-capacity — so the injected disturbance
amplifies the machine's **own** coupling rather than laying a differently-shaped
one on top. The layer re-checks it against the live model at startup
(measured: corr **+0.999**).

It was worth going and getting, because the machine's structure is not what the
obvious geometric proxy predicts:

| path | measured `|M⁻¹|` | a distance proxy would say |
|---|---|---|
| hip ← hip, adjacent legs | **4.382** | strong ✓ |
| hip ← hip, diagonal legs | **2.823** | weaker ✓ |
| ankle ← ankle, **diagonal** legs | **1.795** | weakest ✗ |
| ankle ← ankle, adjacent legs | **0.309** | stronger ✗ |
| **cross** (hip ↔ ankle, incl. same leg) | **0.000** | *strongest* ✗✗ |

Diagonal legs have **parallel** ankle axes, so they couple 5.8× more strongly
than adjacent ones — the opposite of the distance ordering. And the cross path,
which a proximity kernel rates highest of all (a leg's own hip and ankle are the
closest pair on the robot), is exactly zero. The geometric surrogate correlates
only **+0.19** with this; it is kept in `structure.py` only as the offline
fallback, and a run on it must be described as using a declared transmission
*model* rather than the machine's own sensitivity.

## P-3.4: the cross path is pruned, and it had to be

A class carrying less than `MIN_SHARE = 1e-3` of the operator's weight is
dropped from the regressor (the physics keeps it — it contributes exactly zero
anyway). On Ant that is the cross path, and leaving it in was not cosmetic: a
dead column took the design matrix to `cond_psi = 9.3e10`, `scale = 0.0`, and
the placebo-reacquisition check to `fit_gain = -0.150`. With it pruned,
`r_live = 2`, `cond_psi = 2.6`, and reacquisition returns `fit_gain = 0.999`.

## Honest limits

1. **σ = 1 is a stated calibration, not a published anchor.** A simulated
   quadruped has no constant for "how much more of your neighbour's push
   reaches you when the gearbox is warm". 0.14 of the torque range at the peak
   under full peer torque is a *procedure* (measured: 0.1397), carried over
   from the HCM figure so the instances share a scale. Every σ > 1 is a
   beyond-physical stress test.
2. **Learned trust (P-6.2) is not implemented** — the correction is applied
   below the policy, so `∂ log π / ∂g = 0`. Arms are `off` / `fixed` /
   `oracle` / `intercept`; `pact_fixed` and `pact` coincide. `simple_ns` and
   `grf_ns` have the same limitation. (The repo's older `pact_1` arm makes
   trust an extra sampled action dimension, which *does* get a gradient but
   adds a sampled dimension the spec's P-6.2 asks not to add.)
3. **The energy cost of compensating is not modelled.** The feed-forward is
   applied below `ctrl_cost`, so the compensator is not charged for the extra
   torque it commands. That is what makes the conjugacy exact and the oracle a
   true ceiling; the honest bound on it is the relief valve `ns_corr_clip` (0.5
   of the range) and the actuator's own limit, both reported.
4. **The Part-C endpoint carries no information** (100 % peer by construction —
   every actuator on the machine belongs to an agent). Quote the N-scaling
   curve.
5. **`recv_spread` is declared and injected, not measured.** Ant's four legs
   are identical, so nothing in the simulator supports a per-actuator spread;
   it is a property of real hardware asserted here to give the operator its
   asymmetry. Say so. (The version of this claim the model refuted is recorded
   above rather than quietly deleted.)
6. **`operator.json` is machine-specific.** The one in the repo reproduces the
   dumping machine's `load_norm` to 2e-6, but re-run `dump_operator.py` after
   any sync so the operator is the one your MuJoCo actually has; the banner
   prints its provenance every run. Without it the layer falls back to the
   geometric surrogate, which correlates only +0.19 — usable, but then W is a
   declared transmission *model*, not the machine's own sensitivity, and the
   paper must say which was used.
7. **A stale `ns_load_norm` silently rescales the whole dial.** Changing the
   operator changes it (surrogate 0.4718 → machine's own 7.7255, a 16× jump
   that put `|d|` at 13.3 of a torque range of 1 before it was caught). The
   layer now **aborts** when the committed value disagrees with the reference
   partition's own, and only warns on other partitions where a difference is
   the N-scaling. It is committed at the `4x2` reference partition** and is
   deliberately *not* recomputed per partition: at `8x1` the same σ delivers
   more because there **are** more peers, which is the N-scaling prediction
   rather than a change of dial. The layer prints both and warns.
8. **The driver's shape is injected.** Thermal drift is what a practitioner
   names; the `sin²` cycle is ours.
9. **The operator is a nominal-pose linearisation.** `κ` is evaluated once
   from the declared geometry and held fixed; the real distances move as the
   ant walks. That is the standard declared-sensitivity choice (URB's
   incidence-over-capacity and POWER's PTDF are the same kind of object), and
   it is what keeps the operator declared rather than fitted — but it is a
   modelling choice and belongs in the ablation table.
10. **`Ant-v2` only.** `manyagent_ant` and other scenarios must run with
   `ns_on: 0` until a structure is declared for them — the layer raises rather
   than silently doing nothing.
