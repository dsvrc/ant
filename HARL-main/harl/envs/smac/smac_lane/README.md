# SMAC-LANE — Lane Swerve Under Drift (SMAC's invertible cell)

|  | **invertible channel** | no inverse |
|---|---|---|
| **(C) interaction-mediated** | **SMAC-LANE ← this instance**, GRF-NS | `smac_ns` (overkill), `smac_sa` (surface area), URB |
| (B) exogenous | the control | — |

> **(C, invertible) — identification AND compensation.** A unit ordered to move
> with a teammate in its lane swerves off the line it was ordered along. SMAC
> issues every move as a world point, so the swerve is a rotation of that point,
> and a rotation has an exact inverse: a correct estimate *cancels* the
> disturbance. Each unit undoes its own swerve, so no coordination is needed —
> which is exactly what the two steering designs lacked.

## The story

> *Three Stalkers kite five Zealots. A Stalker running with a teammate in its path
> has to go round it: it swerves off the line it wanted, away from the teammate.
> How wide depends on how much room the ground leaves — open ground, or a choke —
> and that drifts as the fight moves across the map; nobody controls it. A Stalker
> alone never swerves round anyone.*

| object | SMAC | check |
|---|---|---|
| medium | my lane: the cone within 135° of my ordered move direction, weighted `κ(d)=1/(1+(d/λ)²)`, λ = 2 | — |
| occupancy | living teammates in the lane, strictly `j ≠ i` | gate 3 |
| `r` classes | the sector a teammate occupies: **front** (≤45°) / **flank** (45°–135°) | P-1.1 |
| `β*` | `σ·L·A(t)·send/load_norm`, `send = [1.4, 0.6]` — unknown to the unit | P-1.2 |
| driver `A(t)` | how cramped the ground is; `sin²` bump over half of a 25000-step cycle (100 episode limits), **exactly 0** for the other half | NS-1.3, NS-2.5 |
| harm | the move order's world point rotated by `d` **away from the traffic** (direction public) | NS-1.4 |
| sensor | executed order vs the order sent, along the public side = `d`, exactly | P-2.1 |
| inverse | pre-rotate by `g·clip(β̂·ψ, 0, π/2)`; with `d̂ = d` the executed order **is** the chosen order | II.6 |

Asserted by `selftest.py` through the mixin's real `step` and `get_agent_action`:

```
sigma = 0 or placebo  ->  every executed order == the chosen order
no living teammate    ->  swerve == 0 at sigma = 5            (category C)
oracle (c = d)        ->  executed == chosen wherever the quarter-turn valve does not bind
trust 0 (pactoff)     ->  identical to blind, order for order  (P-7.1)
```

Measured offline on a moving 3-unit squad (σ = 3): the estimator reaches
`beta_cos 1.0000, relerr 0.05`, and compensation cuts the executed swerve
**8.2×** (11.05° → 1.34° per move).

## Why this cell, and not steering

| design | what the offline battle test showed |
|---|---|
| overkill (`smac_ns`) | harder focus fire won **more** at the peak of the harm (focus 1→0.00, 8→0.65 at σ=5): spreading never pays, so no steering can recover |
| surface area (`smac_sa`, shared slots) | a sharp host fell 1.00→0.32 and a centralized capacity-aware allocator held 1.00 — but PACT steering recovered ≤ 4 points: every unit read the same cost and moved together |
| surface area (nearest slots) | coordination became possible, but a sharp stack no longer fell (0.98) |

## Files

| file | what |
|---|---|
| `driver.py` | `A(t)`, the stated calibration `L`, NS-2 gates (reference waveform) |
| `coupling.py` | the lane operator, channels, brute force, filtered memory, references |
| `layer.py` | the mixin: rotation before the tick, the rotated move command, sensing + RLS after |
| `calibrate.py` | the σ ladder against a trained, frozen B0 (needs StarCraft II) |
| `selftest.py` | offline conformance; no StarCraft II, no torch |

The estimator is URB's verbatim core (`smac_ns/pact1_core.py`).

## Runbook

**0. Offline conformance.**

```bash
python -m harl.envs.smac.smac_lane.selftest
```

**1. B0 — stock MAPPO, dial off.** The competent kiter the ladder needs. Check it
wins ≥ 85% before calibrating.

```bash
python examples/train.py --algo mappo --env smac --exp_name b0 --ns_on 0 --num_env_steps 20000000 --seed 1
```

**2. Calibrate σ on the frozen B0** (evaluation only, no training).

```bash
python -m harl.envs.smac.smac_lane.calibrate --run_dir results/smac/3s_vs_5z/mappo/b0/<seed-dir> --sigmas 0,1,2,3,4,6 --episodes 64 --threads 8 --phase peak --out ladder.json
```

Commit the recommended σ (blind lost ≥ 20 points, oracle within 5 of B0) to
`smac.yaml`, with the table.

**3. The arms, trained inside the dial at that σ** (replace `3` with it).

```bash
python examples/train.py --algo mappo --env smac --exp_name blind --ns_severity 3 --num_env_steps 20000000 --seed 1
```
```bash
python examples/train.py --algo pactoff --env smac --exp_name pactoff --ns_severity 3 --num_env_steps 20000000 --seed 1
```
```bash
python examples/train.py --algo pact --env smac --exp_name pact --ns_severity 3 --num_env_steps 20000000 --seed 1
```
```bash
python examples/train.py --algo pact_oracle --env smac --exp_name oracle --ns_severity 3 --num_env_steps 20000000 --seed 1
```
```bash
python examples/train.py --algo pact_intercept --env smac --exp_name intercept --ns_severity 3 --num_env_steps 20000000 --seed 1
```
```bash
python examples/train.py --algo happo --env smac --exp_name happo_ns --ns_severity 3 --num_env_steps 20000000 --seed 1
```

5 seeds per arm.

## Reading `pact_debug.csv`

| column | question | healthy |
|---|---|---|
| `ns_dial_ratio`, `ns_A`, `ns_placebo` | is the dial live, and does it switch off? | ≈ 0.5 live |
| `design_swerve_deg_peak` | how much swerve the ground causes at the peak | clearly > 0 |
| `exec_swerve_deg_peak` | how much of it reaches the orders | blind ≈ design; pact ≪ design |
| `exec_swerve_deg_dry` | the placebo | **exactly 0** |
| `ns_lane_frac` | how often a moving unit has a teammate in its lane | not tiny |
| `ns_fit_gain`, `ns_beta_cos`, `ns_beta_relerr` | is the gain identified? | fit up, cos → 1 |
| `ns_corr_clipped` | did the quarter-turn valve bind? | rare |
| `win_rate_peak` / `win_rate_dry` | **the separation** | blind's gap ≫ pact's |

## Honest limits

1. **σ = 1 is a stated calibration, not a published anchor.** StarCraft has no
   constant for swerve round a teammate; the operating point comes from
   `calibrate.py` against B0, and σ > 1 is beyond-physical.
2. **The driver's shape is injected.** Terrain is what a player names; the cycle is ours.
3. **Trust is fixed, not learned** — the compensation is a deterministic transform
   of the sampled action, so `∂ log π / ∂g = 0` (as in the football instance).
4. **The sensor is exact proprioception at the actuator**: the layer knows the
   order it issued. That is the cleanest reading of P-2.1, and it is why
   identification is nearly exact.
5. **A recurrent host could infer the phase** from its own swerves; the PACT
   configs and B0 use the MLP host so every arm shares one architecture.
6. **Blind's fall is not measured offline.** A scripted kiting controller strong
   enough to win 3s_vs_5z could not be built without the game, so the fall is
   measured on the cluster by `calibrate.py` against B0 — before any arm trains.
