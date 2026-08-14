from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .multiagentenv import MultiAgentEnv
from .smac_maps import get_map_params

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

import os.path as osp
from pathlib import Path
import yaml

import random
from gym.spaces import Discrete

from .pact.pact1_core import (
    AgentRLS as _P1RLS,
    MIXNORM as _P1MIXNORM,
    R as _P1R,
    THETA_LEGACY as _P1THETA_LEGACY,
    ell_from_shift as _p1_ell_from_shift,
    predict_ell as _p1_predict_ell,
    shift_from_ell as _p1_shift_from_ell,
    theta_anchors as _p1_theta_anchors,
    theta_at as _p1_theta_at,
    type_split as _p1_type_split,
)

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}

import os

# ===========================================================================
# Non-Stationarity  ::  Coupled Targeting Interference (CTI)
# ---------------------------------------------------------------------------
# A category-C, dynamics-only non-stationarity for SMAC, controlled by ONE knob.
#
# Story: the squad's weapons share a targeting/fire-control bus.  The more the OTHER
# units are shooting, the more crosstalk on the bus and the further YOUR shot is
# DEFLECTED onto the wrong enemy.  An exogenous engagement-tempo A(t) sets how bad the
# crosstalk gets.  The shot still fires and still does full damage -- it just lands on
# a different target than the one you aimed at.  The reward is byte-for-byte stock
# SMAC; a unit earns less only because the squad stops focusing fire.
#
#   Phi_j  = 1 if unit j is ENGAGED (alive, with an enemy in weapons range)
#            -- NOT "j pulled the trigger": keying the load to trigger-pulls lets the
#            team switch the NS off by not shooting, which it learned to do instead of
#            learning to compensate (fire_frac 0.82 -> 0.50, fire_avail 0.90 -> 0.55,
#            ep_len 50 -> 86, win capped at 0.44).  SMAC_SND_PHI=fire restores it.
#   x2_i   <- RHO*x2_i + (1-RHO) * (sum_{j!=i} Phi_j)/(N-1)       the shared load, in [0,1]
#   ell_i  =  clip( A(t)*SEVERITY * (x2_i - KNEE)/(1 - KNEE), 0, LMAX )   deflection
#   harm   :  the delivered target is shifted s = round(ell_i*(K-1)) places along unit
#             i's own list of K currently-attackable enemies
#
# Category C:  A(t) MULTIPLIES the sum over the OTHERS -> at N=1 the sum is empty,
# x2 == 0, and it reduces EXACTLY to stock SMAC; frozen-but-firing partners still
# deflect you; a unit can never deflect itself (the sum excludes i).
#
# *** THE CHANNEL IS INVERTIBLE, AND THAT IS WHY IT IS THIS AND NOT AN OVERHEAT. ***
# The deflection is a deterministic PERMUTATION of the attack action, so an agent that
# knows s commands `desired - s` and lands exactly where it wanted, at ZERO COST.
# Compensation restores the stationary game byte for byte -- pipeline T2 conjugacy --
# so 0.9*B0 is reachable and sigma* is a real frontier.  This is the Part-II
# "Transform: delivered = T(a) -> inverse u = T^-1(a)" row, and the pipeline's own
# SMAC sketch ("channel = target displacement -> inverse re-aims by -beta*x2").
#
# The predecessor (CWO) instead DROPPED the shot, which is NOT invertible: the only
# response is to fire less, which buys damage with damage.  Phase 1 measured exactly
# that -- compensation genuinely helped (sigma=1.0: return 7.9 -> 11.3, throughput
# 0.489 -> 0.631) but the coordinated ceiling was ~0.68 against a stationary 0.88, so
# the 0.9*B0 bar of 16.4 was unreachable from a best of 11.3 at EVERY severity from 0.5
# to 3.2.  No conjugacy => no frontier => nothing Phase 1 can certify.  See HISTORY.
#
# Blind agents are hurt because SMAC rewards FOCUS FIRE: a mis-aimed squad spreads
# damage across the enemy line, kills nothing, and dies to full-strength opponents.
# Nothing is physically removed -- it falls because it cannot USE x2, not because
# capability was taken away.
#
# *** x2_i IS ALMOST A GLOBAL SCALAR. *** With N=8, x2_i and x2_j differ only by the
# excluded own-fire term: |x2_i - x2_j| <= (1-RHO)/(N-1) = 0.021 per step.  That is
# fine here (every agent needs the same deflection estimate, not a per-agent role), but
# it is why any diagnostic that splits agents by their own x2_i is splitting on a
# near-degenerate quantity -- see _cwo_fill_diag on `stagger_gap`.
#
#   * SMAC_SND_ORACLE=1        appends the TRUE liability ell_i to obs/state (proof)
#   * env_args["snd_pact"]=1   appends the COMPUTED x2_i to obs/state (the method),
#                              plus TWO raw leaky counters -- x3_jam (shots of mine
#                              deflected lately) and x3_try (shots I attempted lately)
#                              -- unless env_args["snd_pact_feedback"]=0.  Their ratio
#                              estimates ell_i, which is what makes the hidden driver
#                              A(t) locally observable: ell_i = A*SEVERITY*x2_i, so
#                              with x2_i known the agent is tracking ONE hidden scalar.
#                              Two counters rather than one rate so "I have no evidence"
#                              (x3_try~0) is distinguishable from "I fired and nothing
#                              was deflected".  NOT privileged -- the unit feels its own
#                              shot go astray.
# ===========================================================================


# ###########################################################################
# ###  THE ONE KNOB  --  edit SEVERITY to set how hard the NS is.          ###
# ###########################################################################
#   SEVERITY = 0.0   ->  stock SMAC, no NS      (use this to train the baseline B0)
#   higher           ->  larger target deflection  ->  harder task
#
#   The harm channel (see _snd_step and get_agent_action) is
#       ell_i = clip( A(t) * SEVERITY * (x2_i - _KNEE)/(1 - _KNEE),  0, _LMAX )
#       delivered target = commanded target shifted round(ell_i*(K-1)) places along
#                          unit i's K currently-attackable enemies
#   With _KNEE=0 and _LMAX=1 this is simply ell = A*SEVERITY*x2.
#
#   *** TUNING IS EASY NOW, because the channel is INVERTIBLE. ***  Re-aiming costs
#   nothing, so compensation restores the stationary game exactly and there is no
#   headroom/ceiling trade-off to balance -- pick SEVERITY so that a BLIND team is
#   clearly hurt at the driver peak and let Phase 1 find sigma*.  With the measured
#   greedy load x2 ~= 0.70 on 3s5z, SEVERITY=1.0 gives ell ~= 0.70 at the peak, i.e.
#   the shot lands ~70% of the way around the target list -- a badly scattered volley.
#     * CERTIFY it: `python -m harl.envs.smac.pact.phase1` sweeps a scripted,
#       privileged RE-AIM controller over (severity x gain).  beta=1 is the exact
#       inverse and should reproduce B0; sigma* is where even that stops working
#       (it degrades once K, the number of attackable enemies, varies enough that the
#       shift is coarse).  Run at sigma <= sigma*.  Skipping Phase 1 is Pitfall 5.
#     * blind return does NOT fall -> raise SEVERITY;  the scripted inverse cannot
#       recover -> lower it.
#   (Per-run override:  SMAC_SND_SEVERITY=1.0 python ...)
#
#   HISTORY -- the harm channel used to DROP the shot (Coupled Weapon Overheat).  Four
#   settings were measured before that idea was abandoned; do not re-derive them.
#
#   (1) `ell = A*sigma*x2`, SEVERITY=2.0, _LMAX=0.6.  NO coordination solution at all:
#       past f = _LMAX/(A*sigma) = 0.3 the cap makes throughput T(f) = f*(1-_LMAX),
#       strictly INCREASING, so greedy beat the "stagger" optimum by 3.2x.  20M steps:
#       firing fraction at the driver peak vs trough 0.335 vs 0.345 -- no modulation,
#       because none was ever profitable -- and win rate 0.95 -> 0.07, never recovered.
#
#   (2) SEVERITY=1.0, _KNEE=0.15, _LMAX=0.95.  Headroom 5.9x but CEILING only 36% of
#       stationary damage.  17.7M steps with de-aliased eval: win rate at the driver
#       peak was 0.000 in EVERY ONE of 221 eval rounds while the same policy won 93% at
#       the trough.  With no harm-free firing level the team also stopped engaging
#       (fire_avail 0.89 -> 0.22, ep_len 50 -> 141) and the trough win rate collapsed
#       0.93 -> 0.01.  During the ramp the policy moved the WRONG way (hold_gap -0.074:
#       it fired MORE at the peak, individually rational, collectively fatal).
#
#   (3) SEVERITY=2.2, _KNEE=0.40.  Ceiling 57%.  Phase 1 returned win 0.000 in EVERY
#       cell from sigma 0.5 to 2.8 -- 3s5z is a MIRROR match, so by Lanchester's square
#       law the outcome is a CLIFF in relative DPS (a 27% drop rate took win 0.600 ->
#       0.000 with ep_len FALLING 45.7 -> 37.3: the team dies faster, it does not time
#       out).  Win rate has no partial band and cannot express a frontier on this map.
#
#   (4) SEVERITY=2.5, _KNEE=0.55, ceiling 79%, read on RETURN with a working re-aim
#       sweep.  Compensation demonstrably WORKED -- at sigma=1.0 the scripted
#       controller took the shared load 0.650 -> 0.471, the drop rate 0.448 -> 0.050,
#       throughput 0.489 -> 0.631 and return 7.9 -> 11.3 -- and it STILL failed the bar
#       at every severity from 0.5 to 3.2, because 0.9*B0 = 16.4 and the best reachable
#       return was 12.5.  *** That is the decisive result: a DROP channel is not
#       invertible, so pipeline T2 conjugacy does not hold, so no gain can return the
#       team to B0 and Phase 1 has nothing to certify.  Holding fire buys damage with
#       damage. ***  Hence the move to a deflection (permutation) channel, which is
#       invertible at zero cost.
#   NOTE: with Phi = ENGAGEMENT the stationary load is ~0.88 (fire_avail), not the
#   ~0.63 that trigger-pulls produced, so the same SEVERITY now bites ~40% harder.
#   0.7 * 0.88 = 0.62 deflection at the peak, i.e. a blind shot lands ~2 enemies away
#   on a 5-target list.  Let phase1 confirm.
SEVERITY = float(os.environ.get("SMAC_SND_SEVERITY", "0.7"))

# ###########################################################################
# ###  CURRICULUM (WARMUP)  --  the NS ramps in AFTER a warmup.            ###
# ###########################################################################
# Introducing the NS from step 1 is believed to trap a from-scratch learner in a "farm
# damage, never finish the game" basin.  (pact/diagnose.py was cited as proving this,
# but it did not: it never set snd_eval, so _curr_severity ramped its few-thousand-step
# probe env from 0 and every cell ran at severity 0.  Both it and calibrate.py now pass
# snd_eval=1.  Re-run before relying on the claim.)
# So TRAINING severity ramps from 0: it is 0 for the first _WARMUP steps (the policy
# learns to win), then linearly rises to the full SEVERITY over the next _RAMP steps,
# then stays full.  EVALUATION always uses the full SEVERITY (so eval measures the
# harmed win rate as the policy adapts).  Set _WARMUP=0 to disable the curriculum.
#
# UNITS: PER-PARALLEL-ENV step counts (total = value * n_rollout_threads).  Defaults
# below, with n_rollout_threads=20: _WARMUP=500000 -> first 10M env steps stationary
# (learn to win -- 3s5z needs ~10M to reach a strong win-rate), _RAMP=250000 -> ramp
# over the next 5M, then full.  ***The warmup must reach a good stationary win-rate
# (the ceiling PACT can recover to), so give the run enough total steps: e.g.
# num_env_steps=30M => 10M warmup + 5M ramp + 15M under the NS.***  Increase _WARMUP if
# the eval win-rate is still climbing when the ramp starts.
#
# *** THE WARMUP IS A BASELINE CONFOUND -- prefer warm-starting from a shared B0. ***
# During the warmup every arm is running byte-identical stock SMAC, so if one arm's
# warmup succeeds and another's does not, the whole downstream comparison is about
# warmup luck rather than the NS.  That is what happened on the 20M-step 3s5z run: the
# blind arm sat at training win rate 0.000 for the *entire* 10M severity-0 warmup
# (ep_len ~135, ep_reward ~18 -- the farm-damage/timeout basin) while the PACT arm,
# same algorithm, same hyperparameters, same seed, differing only by the appended obs
# scalar, reached 0.95 by 4M.  "PACT 0.13 vs blind 0.00" then measures nothing about
# the method.  The clean protocol is: train one B0 per obs shape at severity 0, CHECK
# BOTH REACH A COMPARABLE STATIONARY WIN RATE, then run every arm from those
# checkpoints with SMAC_SND_WARMUP=0 (see harl/envs/smac/pact/README.md).
#
# BUDGET: the curriculum used to eat 15M of a 20M run (10M warmup + 5M ramp), leaving
# only 5M at full severity -- and the arm was still improving when the run ended
# (return 15.0 -> 17.0 and win 0.35 -> 0.44 over the last 8M).  Defaults now spend 6M
# on warmup + 3M on the ramp, leaving 11M under the NS in the same 20M budget.
_WARMUP = int(float(os.environ.get("SMAC_SND_WARMUP", "300000")))  # per-env steps at sigma=0
_RAMP   = int(float(os.environ.get("SMAC_SND_RAMP", "150000")))    # per-env steps to ramp in

# --- fixed internals (you normally NEVER touch these; what each does) ---
_RHO    = 0.85   # heat memory: how long an overheat lingers (~ 1/(1-RHO) ~= 7 steps)
_KNEE   = float(os.environ.get("SMAC_SND_KNEE", "0.0"))
#                  free-load allowance: shared load up to _KNEE costs NOTHING.  Gives
#                  the coordination target a crisp, learnable shape ("keep the bus
#                  under the knee") and keeps a lightly-engaged team completely
#                  unharmed, so the NS only bites when the team over-fires.
#                  SIZE IT AGAINST THE MEASURED GREEDY LOAD (~0.6*_GREEDY_LOAD).  It
#                  sets the CEILING: when SEVERITY >= (1-_KNEE)/_KNEE the optimum is
#                  the knee exactly, so a coordinated team takes zero drops and keeps
#                  throughput _KNEE.  A knee that is too small caps what perfect play
#                  can achieve -- at _KNEE=0.15 the peak ceiling was 36% of stationary
#                  damage and the win rate at the peak was 0.000 in 221 consecutive
#                  eval rounds.  It is also the team's SAFE HARBOUR: with no level of
#                  firing that is free, the policy learns to stop engaging altogether.
_GREEDY_LOAD = float(os.environ.get("SMAC_SND_GREEDY_LOAD", "0.88"))
#                  the shared load a trained STATIONARY team actually runs at, used
#                  only to report the coordination headroom honestly.  Measure it:
#                  x2_mean from a severity-0 run's pact_debug.csv, or read the
#                  "measured greedy load" line pact.phase1 prints for B0.  On 3s5z a
#                  B0 checkpoint runs fire_frac 0.88 -> x2 ~= 0.70.  Comparing against
#                  a hypothetical f=1 instead overstates the penalty greedy pays, and
#                  sizing _KNEE against the wrong number sets the wrong ceiling.
_LMAX   = float(os.environ.get("SMAC_SND_LMAX", "1.0"))
#                  cap on the drop probability.  MUST be close to 1: once ell is
#                  capped, T(f) = f*(1-_LMAX) is INCREASING in f, so any f past the
#                  cap point rewards firing MORE -- a low cap silently deletes the
#                  coordination solution (see the HISTORY note on SEVERITY).  The
#                  cap now only exists so the drop probability is never exactly 1.
_PERIOD = int(float(os.environ.get("SMAC_SND_PERIOD", "5000")))
#                  engagement-tempo cycle length in steps (collapse-and-recover once
#                  per cycle).  The clock persists across episodes.  Parallel envs are
#                  DE-PHASED across the cycle (see _snd_phase / snd_phase) so every
#                  rollout batch and every eval round is a true cycle-average rather
#                  than a single-phase snapshot -- see the _DEPHASE note below.
_ORACLE = os.environ.get("SMAC_SND_ORACLE", "0").lower() not in ("0", "false", "", "no")

# ###########################################################################
# ###  DE-PHASING  --  parallel envs are spread over the driver cycle.     ###
# ###########################################################################
# The driver period (_PERIOD=5000 steps) is ~30x a rollout (episode_length=160) and
# ~40x an episode.  If every parallel env shares a phase then
#   * every PPO batch contains ONE phase of A(t) -- the critic chases a moving
#     target and the policy is re-fit to a different game every update (the
#     train win-rate oscillates 0 <-> 0.9 with the driver period), and
#   * every eval round is a single-phase SNAPSHOT: with 40 episodes / 10 threads the
#     eval clock advances only ~4*ep_len ~= 250-600 of the 5000 steps, so consecutive
#     evals walk slowly around the cycle and the reported win-rate is a slow square
#     wave (0.0 for most evals, ~0.95 for a few) that measures the DRIVER PHASE, not
#     the policy.  Its beat period even shrinks from ~20 evals to ~10 as episodes get
#     longer -- the fingerprint of exactly this aliasing.
# So env rank r starts its clock at r*_PERIOD/n_threads.  The curriculum is driven by
# a SEPARATE age counter, so de-phasing cannot shift the warmup boundary.
# (Set SMAC_SND_DEPHASE=0 to reproduce the old in-phase behaviour.)
_DEPHASE = os.environ.get("SMAC_SND_DEPHASE", "1").lower() not in ("0", "false", "", "no")

# ###########################################################################
# ###  EXERTION Phi -- what the OTHERS do that loads your bus.             ###
# ###########################################################################
# This is THE design decision that determines whether a blind baseline can escape
# the non-stationarity through the CONTROL space instead of solving it.  Three
# settings, in increasing order of how hard they are to dodge:
#
#   "fire"    Phi_j = j pulled the trigger.  ESCAPABLE and measured: the team simply
#             fires less, which switches the NS partly off (fire_frac 0.82 -> 0.50,
#             ep_len 50 -> 86, win capped at 0.44).  It learned to stop fighting
#             instead of to re-aim.
#
#   "engage"  Phi_j = j is alive AND has an enemy in range.  Closes the trigger
#             hatch but opens a slower one: the squad can DISENGAGE -- back off so
#             nothing is in range.  Also measured (fire_avail 0.89 -> 0.22, ep_len
#             50 -> 141), and the runner prints [PACT][DISENGAGED] when it happens.
#
#   "alive"   Phi_j = j is alive.  *** DEFAULT.  UNCANCELLABLE. ***  Every powered
#             unit draws from the shared bus whatever it chooses to do, so the only
#             way to lower the load is to LOSE UNITS -- which costs the battle
#             directly.  There is no behavioural dodge left; the sole remaining
#             mitigation is the intended one, compensate the deflection.
#
# This is the exact analogue of the fix Ant needed.  There, the coupling read a
# SIGNED torque sum, so an anti-symmetric gait shrank it for free -- measured, blind
# halved its own disturbance over training (|d| 0.086 -> 0.044 at matched driver
# level).  Keying the load to something the team cannot re-shape is what made blind
# actually fail.  Same disease, same cure.
#
# NOTE ON IDENTIFIABILITY.  "alive" does not make the coupling constant: 3s5z is 3
# stalkers + 5 zealots, and the two types die at different rates, so B_same and
# B_cross keep varying INDEPENDENTLY through attrition.  That is precisely the
# regime test_pact1.py's T4b measures as well-conditioned; a constant-engagement
# process is the degenerate one.
_PHI = os.environ.get("SMAC_SND_PHI", "alive").lower()
assert _PHI in ("alive", "engage", "fire"), (
    f"SMAC_SND_PHI must be alive|engage|fire (got {_PHI!r})"
)
_PHI_ENGAGE = _PHI == "engage"   # kept so old comparisons/scripts still read right

# ###########################################################################
# ###  PACT-1 HARDENING -- the interference SPLIT is unknown.  DEFAULT OFF. ##
# ###########################################################################
# The squad shares a fire-control bus, but not every teammate loads it the same way:
# units on the same channel interfere differently from units on another.  On 3s5z
# (3 stalkers + 5 zealots) the natural basis is same-type vs cross-type, and HOW the
# squad's emissions split across those two channels is a property of the loadout --
# unknown, and drifting as the engagement develops.
#
#     x_m,i <- RHO*x_m,i + (1-RHO)*B_m,i        B_same, B_cross  (each /(N-1))
#     psi_i  = MIXNORM * [x_1,i, x_2,i]         computable by the agent, exactly
#     ell_i  = c(t) * sum_m theta_m psi_m,i  =  beta* . psi_i ,   beta* = c(t)*theta
#
# The agent is told the two channels EXIST (that is squad composition, which it can
# see) but not how the load splits between them.  So the unknown goes from one hidden
# scalar to a drifting r-vector that has to be tracked ONLINE, DECENTRALIZED, from
# each unit's own observation of where its shots actually landed.
#
# BACKWARD COMPATIBILITY IS EXACT.  B_same + B_cross == the legacy uniform average,
# so theta=(1/2,1/2) with MIXNORM=2 reproduces the current env byte for byte.  The
# hardened env strictly CONTAINS the old one.  (Asserted in pact/test_pact1.py.)
#
# NOTE (honest, and a paper point): re-aiming does NOT change Phi -- a unit that
# pre-shifts its target is still engaged, still firing.  So unlike Ant, SMAC has NO
# COMPENSATION LOOP: compensating here does not feed the medium it compensates
# against.  That makes this env the CONTRAST CASE for the loop/commons result --
# it should show no over-compensation and no Pigouvian CTDE lift.  Set
# SMAC_SND_LOOP=1 to close the loop deliberately (emitting a stronger corrected
# targeting solution loads the bus harder) and recover Ant's structure.
_PACT1_MIX = os.environ.get("SMAC_SND_MIX", "0").lower() not in ("0", "false", "", "no")
_PACT1_SEED = int(os.environ.get("SMAC_SND_MIX_SEED", "0"))
#                 *** set this to the RUN SEED *** so theta rides the seed axis you
#                 already sweep and hardening costs no extra runs.
_PACT1_PERIOD = int(float(os.environ.get("SMAC_SND_MIX_PERIOD", "8000")))
#                 how long the loadout takes to slide between its anchors and back.
#                 Slower than the driver cycle (_PERIOD=5000); a second timescale.
_PACT1_RADIUS = float(os.environ.get("SMAC_SND_MIX_RADIUS", "0.35"))
#                 how far the split may wander from the legacy point.  THE "a bit
#                 harder" DIAL.  The harm is NOT constant across the simplex, so an
#                 unbounded theta is a different task at a different effective
#                 severity -- outside the certified frontier, where nothing recovers.
_PACT1_CONC = float(os.environ.get("SMAC_SND_MIX_CONC", "0.9"))
# ###########################################################################
# ###  RLS FORGETTING -- DERIVED FROM THE DRIVER PERIOD, NOT HAND-SET.     ##
# ###########################################################################
# The estimator must track beta* = A(t)*sigma*theta(t).  What decides whether it
# CAN is not the forgetting factor itself but the estimator's MEMORY AS A FRACTION
# OF THE DRIVER PERIOD -- and that depends on how often the sensor fires, which is
# where SMAC and Ant differ by an order of magnitude:
#
#   Ant   1 reading/step,    mu=0.999 -> 1000 updates = 1000 steps of a 40000 period
#                                                     =  2.5%   -> tracks (beta_err 0.020)
#   SMAC  ~0.43 readings/step, mu=0.999 -> 1000 updates = 2330 steps of a 5000 period
#                                                     =   47%   -> AVERAGES THE CYCLE
#
# Measured on the 20M 3s5z run at mu=0.999: beta_hat0 = 0.3744 against a true mean of
# 0.3699 (the MEAN is right to 1.2%) while beta_err = 0.362 -- exactly the RMS swing
# of beta* about its mean.  The estimator had converged to the cycle average and was
# tracking nothing.  Downstream: innov 0.355 ~ ell itself, the resolvability gate open
# on 18% of shots, cancel 0.054.  PACT-1 was blind with a handicap.
#
# So mu is DERIVED to hold the memory at Ant's ratio.  Change SMAC_SND_PERIOD and mu
# follows; there is no way to set the two inconsistently.  env_args.pact1_forget still
# overrides for ablations.
_P1_MEM_FRAC = float(os.environ.get("SMAC_SND_MEMFRAC", "0.025"))
#                 estimator memory as a fraction of the driver period.  Ant's measured
#                 working value.  Smaller = tracks faster but amplifies the quantiser
#                 noise; the optimum of that trade is guide III.9's tracking floor.
_P1_READ_RATE = float(os.environ.get("SMAC_SND_READRATE", "0.43"))
#                 sensor readings per step per unit -- MEASURED (p1_obs_frac on the
#                 3s5z runs, 0.43-0.57).  Declared, not tuned; it only converts the
#                 memory from steps into RLS updates.
_P1_MU_AUTO = float(np.clip(
    1.0 - 1.0 / max(2.0, _P1_MEM_FRAC * _PERIOD * _P1_READ_RATE), 0.90, 0.9995
))

# ###########################################################################
# ###  DITHERED DEFLECTION -- the channel property that makes SMAC == ANT. ##
# ###########################################################################
# The delivered target is  floor(ell*(K-1) + u)  where u in [0,1) is the unit's own
# sub-target aim offset this step, instead of  round(ell*(K-1)).  See
# pact/pact1_core.shift_from_ell for the full argument; in one line: round() has a
# dead zone below ell = 0.5/(K-1) (so the sensor emits pure zeros through the whole
# curriculum ramp) and demands EXACT integer agreement to cancel (tolerance
# 0.5/(K-1), which is the sensor's own resolution -- zero headroom).  Dithering is
# unbiased at every severity, preserves exact cancellation when ell_hat == ell, and
# makes the miss probability LINEAR in |ell - ell_hat| instead of a step function.
#
# It also makes the channel BITE HARDER on a blind team at low ell: a deflection of
# ell = 0.1 on a 5-target list now displaces the shot 40% of the time instead of
# never.  The harm is graded across the whole driver cycle rather than switching on
# at a threshold.
#
# Set SMAC_SND_DITHER=0 to restore the deterministic quantiser (ablation).
_DITHER = os.environ.get("SMAC_SND_DITHER", "1").lower() not in ("0", "false", "", "no")

_P1KREF = float(os.environ.get("SMAC_SND_KREF", "5"))
#                 reference target-list size at which a deflection reading counts as
#                 variance 1.  Only ratios matter (see _pact1_observe); 5 is the
#                 typical number of attackable enemies on 3s5z.
_PACT1_LOOP = float(os.environ.get("SMAC_SND_LOOP", "0"))
#                 >0 closes the compensation loop: Phi_j <- Phi_j*(1 + LOOP*|s_hat_j|)
#                 so a unit that corrects harder loads the bus harder.  0 = the
#                 contrast case (no loop), which is the default.

_NS_BANNER_SHOWN = False  # print the resolved NS config once per process


def _driver(clock):
    """Engagement tempo A(t) in [0, 1]: a smooth raised-cosine over _PERIOD steps
    (collapse-and-recover once per cycle).  Hidden from the agents; only the global,
    episode-persisting clock drives it."""
    phase = (clock % _PERIOD) / _PERIOD
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class StarCraft2Env(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(
        self,
        args,
        step_mul=8,
        move_amount=2,
        difficulty="7",
        game_version=None,
        seed=None,
        continuing_episode=False,
        obs_all_health=True,
        obs_own_health=True,
        obs_last_action=True,
        obs_pathing_grid=False,
        obs_terrain_height=False,
        obs_instead_of_state=False,
        obs_timestep_number=False,
        obs_agent_id=True,
        state_pathing_grid=False,
        state_terrain_height=False,
        state_last_action=True,
        state_timestep_number=False,
        state_agent_id=True,
        reward_sparse=False,
        reward_only_positive=True,
        reward_death_value=10,
        reward_win=200,
        reward_defeat=0,
        reward_negative_scale=0.5,
        reward_scale=True,
        reward_scale_rate=20,
        replay_dir="",
        replay_prefix="",
        window_size_x=1920,
        window_size_y=1200,
        heuristic_ai=False,
        heuristic_rest=False,
        debug=False,
    ):
        """
        Create a StarCraftC2Env environment.

        Parameters
        ----------
        map_name : str, optional
            The name of the SC2 map to play (default is "8m"). The full list
            can be found by running bin/map_list.
        step_mul : int, optional
            How many game steps per agent step (default is 8). None
            indicates to use the default map step_mul.
        move_amount : float, optional
            How far away units are ordered to move per step (default is 2).
        difficulty : str, optional
            The difficulty of built-in computer AI bot (default is "7").
        game_version : str, optional
            StarCraft II game version (default is None). None indicates the
            latest version.
        seed : int, optional
            Random seed used during game initialisation. This allows to
        continuing_episode : bool, optional
            Whether to consider episodes continuing or finished after time
            limit is reached (default is False).
        obs_all_health : bool, optional
            Agents receive the health of all units (in the sight range) as part
            of observations (default is True).
        obs_own_health : bool, optional
            Agents receive their own health as a part of observations (default
            is False). This flag is ignored when obs_all_health == True.
        obs_last_action : bool, optional
            Agents receive the last actions of all units (in the sight range)
            as part of observations (default is False).
        obs_pathing_grid : bool, optional
            Whether observations include pathing values surrounding the agent
            (default is False).
        obs_terrain_height : bool, optional
            Whether observations include terrain height values surrounding the
            agent (default is False).
        obs_instead_of_state : bool, optional
            Use combination of all agents' observations as the global state
            (default is False).
        obs_timestep_number : bool, optional
            Whether observations include the current timestep of the episode
            (default is False).
        state_last_action : bool, optional
            Include the last actions of all agents as part of the global state
            (default is True).
        state_timestep_number : bool, optional
            Whether the state include the current timestep of the episode
            (default is False).
        reward_sparse : bool, optional
            Receive 1/-1 reward for winning/loosing an episode (default is
            False). Whe rest of reward parameters are ignored if True.
        reward_only_positive : bool, optional
            Reward is always positive (default is True).
        reward_death_value : float, optional
            The amount of reward received for killing an enemy unit (default
            is 10). This is also the negative penalty for having an allied unit
            killed if reward_only_positive == False.
        reward_win : float, optional
            The reward for winning in an episode (default is 200).
        reward_defeat : float, optional
            The reward for loosing in an episode (default is 0). This value
            should be nonpositive.
        reward_negative_scale : float, optional
            Scaling factor for negative rewards (default is 0.5). This
            parameter is ignored when reward_only_positive == True.
        reward_scale : bool, optional
            Whether or not to scale the reward (default is True).
        reward_scale_rate : float, optional
            Reward scale rate (default is 20). When reward_scale == True, the
            reward received by the agents is divided by (max_reward /
            reward_scale_rate), where max_reward is the maximum possible
            reward per episode without considering the shield regeneration
            of Protoss units.
        replay_dir : str, optional
            The directory to save replays (default is None). If None, the
            replay will be saved in Replays directory where StarCraft II is
            installed.
        replay_prefix : str, optional
            The prefix of the replay to be saved (default is None). If None,
            the name of the map will be used.
        window_size_x : int, optional
            The length of StarCraft II window size (default is 1920).
        window_size_y: int, optional
            The height of StarCraft II window size (default is 1200).
        heuristic_ai: bool, optional
            Whether or not to use a non-learning heuristic AI (default False).
        heuristic_rest: bool, optional
            At any moment, restrict the actions of the heuristic AI to be
            chosen from actions available to RL agents (default is False).
            Ignored if heuristic_ai == False.
        debug: bool, optional
            Log messages about observations, state, actions and rewards for
            debugging purposes (default is False).
        """
        # Map arguments
        state_config = self.load_state_config(args["state_type"])
        self.map_name = args["map_name"]
        self.add_local_obs = state_config["add_local_obs"]
        self.add_move_state = state_config["add_move_state"]
        self.add_visible_state = state_config["add_visible_state"]
        self.add_distance_state = state_config["add_distance_state"]
        self.add_xy_state = state_config["add_xy_state"]
        self.add_enemy_action_state = state_config["add_enemy_action_state"]
        self.add_agent_id = state_config["add_agent_id"]
        self.use_state_agent = state_config["use_state_agent"]
        self.use_mustalive = state_config["use_mustalive"]
        self.add_center_xy = state_config["add_center_xy"]
        self.use_stacked_frames = state_config["use_stacked_frames"]
        self.stacked_frames = state_config["stacked_frames"]

        map_params = get_map_params(self.map_name)
        self.n_agents = map_params["n_agents"]
        self.n_enemies = map_params["n_enemies"]
        self.episode_limit = map_params["limit"]

        # --- Spoof-Coupled Navigation Drift (SND) state ---
        # clock persists across episodes (the spoofing campaign keeps running);
        # the per-unit drift d_i is reset to zero at the start of every episode.
        # Knobs are resolved PER INSTANCE here (before the obs/state spaces are
        # sized below), so the Phase-1 sweep can drive different severities and the
        # PACT/oracle obs-augmentation grows the declared spaces correctly.
        self._snd_resolve_knobs(args)
        self._snd_clock = self._snd_phase0                          # engagement-tempo clock
        #   (de-phased per rank so parallel envs tile the driver cycle -- see _DEPHASE)
        self._snd_age = 0                    # steps since construction; drives the
        #   CURRICULUM only, so de-phasing can never shift the warmup boundary.
        self._cwo_x2 = np.zeros(self.n_agents, dtype=np.float32)   # shared load (PACT waveform)
        self._cwo_x3 = np.zeros(self.n_agents, dtype=np.float32)   # own shots jammed lately
        self._cwo_x3try = np.zeros(self.n_agents, dtype=np.float32)  # own shots attempted
        self._cwo_ell = np.zeros(self.n_agents, dtype=np.float32)  # per-unit drop probability
        self._cwo_dropped = np.zeros(self.n_agents, dtype=np.float32)  # shots jammed this step
        self._cwo_can_fire = np.zeros(self.n_agents, dtype=np.float32)  # attack was AVAILABLE
        self._cwo_diag = {}                                        # per-step debug telemetry
        self._cwo_regen_pay = 0.0    # this step's shield-regeneration pay (diagnostic)
        self._cwo_reward_raw = 0.0   # delta_enemy + delta_deaths BEFORE abs()
        self._snd_sigma_applied = 0.0                             # curriculum severity this step
        self._cwo_rng = np.random.RandomState(0)  # weapon-jam RNG; re-seeded in seed()
        # The unit's own sub-target aim offset, redrawn every step.  Dedicated RNG so
        # it never perturbs the game stream (same discipline as Ant's _sensor_rng),
        # and it exists for EVERY arm -- the channel is the same for blind and PACT-1.
        self._aim_rng = np.random.RandomState(4000 + int(_PACT1_SEED))
        self._cwo_dither = np.zeros(self.n_agents)
        self._pact1_init_state()
        self._snd_payload = 0.0                                    # A(t)
        self._snd_load_mean = 0.0
        self._snd_load_max = 0.0
        self._move_amount = move_amount
        self._snd_banner()
        self._step_mul = step_mul
        self.difficulty = difficulty

        # Observations and state
        self.obs_own_health = obs_own_health
        self.obs_all_health = obs_all_health
        self.obs_instead_of_state = state_config["use_obs_instead_of_state"]
        self.obs_last_action = obs_last_action
        self.use_global_state = state_config["use_global_state"]
        self.global_state_include_info = state_config["global_state_include_info"]

        self.obs_pathing_grid = obs_pathing_grid
        self.obs_terrain_height = obs_terrain_height
        self.obs_timestep_number = obs_timestep_number
        self.obs_agent_id = obs_agent_id
        self.state_pathing_grid = state_config["state_pathing_grid"]
        self.state_terrain_height = state_config["state_terrain_height"]
        self.state_last_action = state_config["state_last_action"]
        self.state_timestep_number = state_config["state_timestep_number"]
        self.state_agent_id = state_agent_id
        if self.obs_all_health:
            self.obs_own_health = True
        self.n_obs_pathing = 8
        self.n_obs_height = 9

        # Rewards args
        self.reward_sparse = reward_sparse
        self.reward_only_positive = reward_only_positive
        self.reward_negative_scale = reward_negative_scale
        self.reward_death_value = reward_death_value
        self.reward_win = reward_win
        self.reward_defeat = reward_defeat

        self.reward_scale = reward_scale
        self.reward_scale_rate = reward_scale_rate

        # Other
        self.game_version = game_version
        self.continuing_episode = continuing_episode
        self._seed = seed
        self.heuristic_ai = heuristic_ai
        self.heuristic_rest = heuristic_rest
        self.debug = debug
        self.window_size = (window_size_x, window_size_y)
        self.replay_dir = replay_dir
        self.replay_prefix = replay_prefix

        # Actions
        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        self.n_actions = self.n_actions_no_attack + self.n_enemies

        # Map info
        self._agent_race = map_params["a_race"]
        self._bot_race = map_params["b_race"]
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = map_params["unit_type_bits"]
        self.map_type = map_params["map_type"]

        self.max_reward = self.n_enemies * self.reward_death_value + self.reward_win

        self.agents = {}
        self.enemies = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None
        self.death_tracker_ally = np.zeros(self.n_agents, dtype=np.float32)
        self.death_tracker_enemy = np.zeros(self.n_enemies, dtype=np.float32)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.last_action = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        self._min_unit_type = 0
        self.marine_id = self.marauder_id = self.medivac_id = 0
        self.hydralisk_id = self.zergling_id = self.baneling_id = 0
        self.stalker_id = self.colossus_id = self.zealot_id = 0
        self.max_distance_x = 0
        self.max_distance_y = 0
        self.map_x = 0
        self.map_y = 0
        self.terrain_height = None
        self.pathing_grid = None
        self._run_config = None
        self._sc2_proc = None
        self._controller = None

        # Try to avoid leaking SC2 processes on shutdown
        atexit.register(lambda: self.close())

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for i in range(self.n_agents):
            self.action_space.append(Discrete(self.n_actions))
            self.observation_space.append(self.get_obs_size())
            self.share_observation_space.append(self.get_state_size())

        # SND: grow the declared obs/state sizes by the PACT/oracle append (the
        # actual concatenation is done in _snd_augment on the RETURNED obs/state, so
        # it works for any state_type -- EP included, unlike the old internal append).
        self._snd_grow_spaces()

        if self.use_stacked_frames:
            self.stacked_local_obs = np.zeros(
                (
                    self.n_agents,
                    self.stacked_frames,
                    int(self.get_obs_size()[0] / self.stacked_frames),
                ),
                dtype=np.float32,
            )
            self.stacked_global_state = np.zeros(
                (
                    self.n_agents,
                    self.stacked_frames,
                    int(self.get_state_size()[0] / self.stacked_frames),
                ),
                dtype=np.float32,
            )

    def _launch(self):
        """Launch the StarCraft II game."""
        self._run_config = run_configs.get(version=self.game_version)
        _map = maps.get(self.map_name)
        self._seed += 1

        # Setting up the interface
        interface_options = sc_pb.InterfaceOptions(raw=True, score=False)
        self._sc2_proc = self._run_config.start(
            window_size=self.window_size, want_rgb=False
        )
        self._controller = self._sc2_proc.controller

        # Request to create the game
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path, map_data=self._run_config.map_data(_map.path)
            ),
            realtime=False,
            random_seed=self._seed,
        )
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(
            type=sc_pb.Computer,
            race=races[self._bot_race],
            difficulty=difficulties[self.difficulty],
        )
        self._controller.create_game(create)

        join = sc_pb.RequestJoinGame(
            race=races[self._agent_race], options=interface_options
        )
        self._controller.join_game(join)

        game_info = self._controller.game_info()
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y

        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8)
            )
            self.pathing_grid = np.transpose(
                np.array(
                    [
                        [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                        for row in vals
                    ],
                    dtype=np.bool,
                )
            )
        else:
            self.pathing_grid = np.invert(
                np.flip(
                    np.transpose(
                        np.array(
                            list(map_info.pathing_grid.data), dtype=np.bool
                        ).reshape(self.map_x, self.map_y)
                    ),
                    axis=1,
                )
            )

        self.terrain_height = (
            np.flip(
                np.transpose(
                    np.array(list(map_info.terrain_height.data)).reshape(
                        self.map_x, self.map_y
                    )
                ),
                1,
            )
            / 255
        )

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        else:
            self._restart()

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents, dtype=np.float32)
        self.death_tracker_enemy = np.zeros(self.n_enemies, dtype=np.float32)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)

        # SND: the squad starts each episode with a cold bus; the load re-accumulates
        # as the team begins to engage.  The engagement-tempo clock is NOT reset.
        self._cwo_x2 = np.zeros(self.n_agents, dtype=np.float32)   # shared load (reset each ep)
        self._cwo_x3 = np.zeros(self.n_agents, dtype=np.float32)   # jams (reset each ep)
        self._cwo_x3try = np.zeros(self.n_agents, dtype=np.float32)  # attempts (reset each ep)
        self._cwo_ell = np.zeros(self.n_agents, dtype=np.float32)  # drop probability
        self._cwo_can_fire = np.zeros(self.n_agents, dtype=np.float32)
        if self.snd_pact1:
            # The bus is cold again, so the accumulators and this episode's sensor
            # readings are stale -- drop them.  The ESTIMATE (beta_hat and its
            # covariance) PERSISTS: the loadout split drifts on a timescale far longer
            # than an episode, so forgetting it every reset would throw away the only
            # thing that is genuinely learnable across the run.
            self._p1_x[:] = 0.0
            self._p1_psi[:] = 0.0
            self._p1_ellhat[:] = 0.0
            self._p1_ellmeas[:] = 0.0
            self._p1_sobs[:] = -1.0
            self._p1_kobs[:] = 0
            self._p1_shat[:] = 0

        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        try:
            self._obs = self._controller.observe()
            self.init_units()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

        available_actions = []
        for i in range(self.n_agents):
            available_actions.append(self.get_avail_agent_actions(i))

        if self.debug:
            logging.debug(
                "Started Episode {}".format(self._episode_count).center(60, "*")
            )

        if self.use_state_agent:
            global_state = [
                self.get_state_agent(agent_id) for agent_id in range(self.n_agents)
            ]
        elif self.use_global_state:
            global_state = [
                self.get_global_state() for agent_id in range(self.n_agents)
            ]
        else:
            global_state = [
                self.get_state(agent_id) for agent_id in range(self.n_agents)
            ]

        local_obs = self.get_obs()

        if self.use_stacked_frames:
            self.stacked_local_obs = np.roll(self.stacked_local_obs, 1, axis=1)
            self.stacked_global_state = np.roll(self.stacked_global_state, 1, axis=1)

            self.stacked_local_obs[:, -1, :] = np.array(local_obs).copy()
            self.stacked_global_state[:, -1, :] = np.array(global_state).copy()

            local_obs = self.stacked_local_obs.reshape(self.n_agents, -1)
            global_state = self.stacked_global_state.reshape(self.n_agents, -1)

        # SND: append the oracle (true d) / PACT (computed x2) block (no-op if off)
        local_obs, global_state = self._snd_augment(local_obs, global_state)

        return local_obs, global_state, available_actions

    def load_state_config(self, state_type):
        base_path = osp.split(osp.split(osp.dirname(osp.abspath(__file__)))[0])[0]
        state_config_path = (
            Path(base_path)
            / "configs"
            / "envs_cfgs"
            / "smac_state_config"
            / f"{state_type}.yaml"
        )
        with open(str(state_config_path), "r", encoding="utf-8") as file:
            state_config = yaml.load(file, Loader=yaml.FullLoader)
        return state_config

    def _restart(self):
        """Restart the environment by killing all units on the map.
        There is a trigger in the SC2Map file, which restarts the
        episode when there are no units left.
        """
        try:
            self._kill_all_units()
            self._controller.step(2)
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

    def full_restart(self):
        """Full restart. Closes the SC2 process and launches a new one."""
        self._sc2_proc.close()
        self._launch()
        self.force_restarts += 1

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        terminated = False
        bad_transition = False
        infos = [{} for i in range(self.n_agents)]
        dones = np.zeros((self.n_agents), dtype=bool)

        actions_int = [int(a) for a in actions]

        # SND: the drift / waveform THIS step's move commands will receive (set at
        # the end of the previous step / at reset), captured BEFORE get_agent_action
        # applies it (from the end of the previous step).  Reset this step's drop
        # record; get_agent_action sets _cwo_dropped[i]=1 when unit i's shot jams.
        self._cwo_dropped = np.zeros(self.n_agents, dtype=np.float32)
        # This step's sub-target aim offsets.  Drawn ONCE per step, BEFORE any agent
        # acts, and read by BOTH the compensator's pre-shift and the channel's
        # deflection inside get_agent_action -- they must see the same u or exact
        # cancellation is impossible.  Drawn for every arm; blind simply never reads
        # it for compensation.
        self._cwo_dither = (
            self._aim_rng.random_sample(self.n_agents) if _DITHER
            else np.zeros(self.n_agents)
        )
        # PACT-1 per-step sensor buffers. Cleared HERE, before get_agent_action fills
        # them, and read by _pact1_observe inside _snd_step -- which runs after all
        # the actions have been applied, so every unit's reading is present.
        if self.snd_pact1:
            self._p1_sobs[:] = -1.0        # -1 = "no shot fired / no choice of target"
            self._p1_kobs[:] = 0
            self._p1_shat[:] = 0

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for a_id, action in enumerate(actions_int):
            if not self.heuristic_ai:
                sc_action = self.get_agent_action(a_id, action)
            else:
                sc_action, action_num = self.get_agent_action_heuristic(a_id, action)
                actions[a_id] = action_num
            if sc_action:
                sc_actions.append(sc_action)

        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)
        try:
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self._controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            terminated = True
            available_actions = []
            for i in range(self.n_agents):
                available_actions.append(self.get_avail_agent_actions(i))
                infos[i] = {
                    "battles_won": self.battles_won,
                    "battles_game": self.battles_game,
                    "battles_draw": self.timeouts,
                    "restarts": self.force_restarts,
                    "bad_transition": bad_transition,
                    "won": self.win_counted,
                }
                if terminated:
                    dones[i] = True
                else:
                    if self.death_tracker_ally[i]:
                        dones[i] = True
                    else:
                        dones[i] = False

            if self.use_state_agent:
                global_state = [
                    self.get_state_agent(agent_id) for agent_id in range(self.n_agents)
                ]
            elif self.use_global_state:
                global_state = [
                    self.get_global_state() for agent_id in range(self.n_agents)
                ]
            else:
                global_state = [
                    self.get_state(agent_id) for agent_id in range(self.n_agents)
                ]

            local_obs = self.get_obs()

            if self.use_stacked_frames:
                self.stacked_local_obs = np.roll(self.stacked_local_obs, 1, axis=1)
                self.stacked_global_state = np.roll(
                    self.stacked_global_state, 1, axis=1
                )

                self.stacked_local_obs[:, -1, :] = np.array(local_obs).copy()
                self.stacked_global_state[:, -1, :] = np.array(global_state).copy()

                local_obs = self.stacked_local_obs.reshape(self.n_agents, -1)
                global_state = self.stacked_global_state.reshape(self.n_agents, -1)

            # SND: augment on the restart path too, so shapes match the grown spaces
            local_obs, global_state = self._snd_augment(local_obs, global_state)

            return (
                local_obs,
                global_state,
                [[0]] * self.n_agents,
                dones,
                infos,
                available_actions,
            )

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()

        reward = self.reward_battle()

        # SND: advance the spoofing campaign + the hidden drift for the NEXT step
        # (uses this step's commanded actions and the post-move unit positions).
        self._snd_step(actions_int)

        available_actions = []
        for i in range(self.n_agents):
            available_actions.append(self.get_avail_agent_actions(i))

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            bad_transition = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        p1_diag = self._pact1_diag() if self.snd_pact1 else {}   # computed once, not per agent
        for i in range(self.n_agents):
            infos[i] = {
                "battles_won": self.battles_won,
                "battles_game": self.battles_game,
                "battles_draw": self.timeouts,
                "restarts": self.force_restarts,
                "bad_transition": bad_transition,
                "won": self.win_counted,
                # --- CWO diagnostics (never touch the reward; for TensorBoard + the
                #     pact_debug.csv).  "snd_load" = mean drop prob (calibration target). ---
                "snd_payload": self._snd_payload,  # A(t), the exogenous engagement driver
                "snd_load": self._snd_load_mean,   # mean drop prob ell over live units
                "snd_loadmax": self._snd_load_max,  # max drop prob over units
                # --- pcr_* aliases so harl/common/ns_probe.py works on SMAC too. ---
                # The probe attaches to EVERY on-policy arm, blind included, which is
                # the point: a PACT arm writes pact_debug.csv and shows its own
                # telemetry, but a blind baseline writes nothing -- so a silently
                # inert NS is invisible in exactly the arm you most need to trust.
                # That hole is what hid an inert disturbance on Ant for a full run.
                "pcr_payload": self._snd_payload,
                "pcr_load": self._snd_load_mean,
                "pcr_loadmax": self._snd_load_max,
                # the severity ACTUALLY APPLIED (after the warmup curriculum), so the
                # probe can tell "sigma is 0 on purpose" from "the NS is broken"
                "pcr_severity": self._snd_sigma_applied,
                "pcr_sat_frac": (
                    self._cwo_diag.get("cwo_drop_frac", 0.0)  # frac of shots deflected
                ),
                "pcr_theta": (
                    self._p1_theta if self.snd_pact1 else _P1THETA_LEGACY
                ),
                # neutral PACT keys read by OnPolicyPactSmacRunner (the smacv2 env emits
                # the same names, so one runner serves both):
                "pact_payload": self._snd_payload,  # driver A (>0.3 = "engaged" filter)
                "pact_dload": self._snd_load_mean,  # mean drop prob
                "cwo_A": self._snd_payload,        # driver A, for the eval phase split
                "cwo_sigma": self._snd_sigma_applied,  # curriculum severity applied (ramps 0->SEVERITY)
                "pact_cos": 1.0,   # x2 is a scalar computed env-side => exact, no leak gate
                "pact_x2load": (
                    self._cwo_diag.get("cwo_x2_mean", 0.0)
                    if (self.snd_pact or self.snd_pact1) else 0.0
                ),
                # --- PACT-1 telemetry (logging only) ---
                **p1_diag,
                **self._cwo_diag,  # cwo_{ell_mean,ell_max,x2_mean,fire_frac,drop_frac,
                #                    fire_hi_load,fire_lo_load} -- see _cwo_fill_diag
            }

            if terminated:
                dones[i] = True
            else:
                if self.death_tracker_ally[i]:
                    dones[i] = True
                else:
                    dones[i] = False

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, "-"))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate

        rewards = [[reward]] * self.n_agents

        if self.use_state_agent:
            global_state = [
                self.get_state_agent(agent_id) for agent_id in range(self.n_agents)
            ]
        elif self.use_global_state:
            global_state = [
                self.get_global_state() for agent_id in range(self.n_agents)
            ]
        else:
            global_state = [
                self.get_state(agent_id) for agent_id in range(self.n_agents)
            ]

        local_obs = self.get_obs()

        if self.use_stacked_frames:
            self.stacked_local_obs = np.roll(self.stacked_local_obs, 1, axis=1)
            self.stacked_global_state = np.roll(self.stacked_global_state, 1, axis=1)

            self.stacked_local_obs[:, -1, :] = np.array(local_obs).copy()
            self.stacked_global_state[:, -1, :] = np.array(global_state).copy()

            local_obs = self.stacked_local_obs.reshape(self.n_agents, -1)
            global_state = self.stacked_global_state.reshape(self.n_agents, -1)

        # SND: append the oracle (true d) / PACT (computed x2) block (no-op if off)
        local_obs, global_state = self._snd_augment(local_obs, global_state)

        return local_obs, global_state, rewards, dones, infos, available_actions

    # ---------------------------------------------------------- CWO (the NS) --
    def _snd_resolve_knobs(self, args):
        """Resolve the per-instance CWO knobs.  Severity's default is the module
        constant ``SEVERITY`` (edit that, or set $SMAC_SND_SEVERITY, to choose a run's
        strength).  The mode flags (oracle / pact / ctde) come from env_args (the
        config); freeze from env_args (calibration only) or None (the live cycle)."""
        a = args or {}
        # severity: env_args (only the calibration script sets it) -> module SEVERITY.
        self.snd_severity = (
            float(a["snd_severity"]) if a.get("snd_severity", None) is not None
            else SEVERITY
        )
        self.snd_oracle = (   # 1 = append the TRUE liability ell_i to obs/state
            int(a["snd_oracle"]) if a.get("snd_oracle", None) is not None
            else int(bool(_ORACLE))
        )
        self.snd_pact = int(a.get("snd_pact", 0))            # 1 = append COMPUTED x2_i
        self.snd_pact_ctde = int(a.get("snd_pact_ctde", 0))  # + true A(t) in the critic
        self.snd_pact_feedback = int(a.get("snd_pact_feedback", 1))  # + own jam rate x3_i
        #   x3_i is the residual that makes the hidden driver locally estimable; without
        #   it the policy sees the load but has NO way to tell how hard the bus is being
        #   pushed this cycle, so beta/phase tracking is information-theoretically out of
        #   reach and only a constant compromise firing rate is learnable.  Set 0 to ablate.
        self.snd_freeze = (   # hold A(t) constant (calibration only); None = live cycle
            float(a["snd_freeze"]) if a.get("snd_freeze", None) is not None else None
        )
        self.snd_eval = int(a.get("snd_eval", 0))  # eval env -> skip the warmup curriculum
        #                                            (use the full severity to measure harm)
        # --- PACT-1: unknown interference split, estimated online -----------------
        self.snd_pact1 = int(a.get("snd_pact1", 0))
        # RLS forgetting: DERIVED from the driver period and the sensor's reading
        # rate so the estimator's memory is a fixed small fraction of the cycle it
        # has to track -- see _P1_MU_AUTO.  Copying Ant's mu directly is what made
        # the estimator converge to the cycle MEAN and track nothing.
        _mu = a.get("pact1_forget", None)
        self.pact1_forget = _P1_MU_AUTO if _mu is None else float(_mu)
        self.pact1_p0 = float(a.get("pact1_p0", 10.0))            # RLS prior looseness
        self.pact1_assist = int(a.get("pact1_assist", 1))
        #   1 = the env applies the pre-shift (a real compensator, Ant's structure);
        #   0 = obs-only, the policy must learn the re-aim itself from ell_hat.
        self.pact1_gpol = float(a.get("pact1_gpol", 1.0))         # trust multiplier
        self.pact1_conf_thresh = float(a.get("pact1_conf_thresh", 0.5))
        #   compensate only once the estimator's self-confidence clears this. The
        #   knob is a THRESHOLD because the channel is a permutation with an integer
        #   shift -- partial re-aim is useless or harmful (Phase 1: beta=0.5 scored
        #   BELOW blind). Cold prior is 1/(1+r) = 0.33, so 0.5 means "wait until the
        #   covariance has roughly halved", then compensate fully.
        # *** FREEZE THE ESTIMATOR WHILE THE APPLIED SEVERITY IS 0.  DEFAULT ON. ***
        # During a sigma=0 warmup every deflection reading is y == 0 with psi != 0.
        # That is NOT information about beta*; it is the channel being switched off.
        # Feeding it to RLS does two harmful things:
        #   (1) it SHRINKS P along psi, so `conf` climbs (measured 0.33 -> 0.75 by 80k
        #       steps on 3s5z) and the compensator ARMS itself on a beta_hat that was
        #       fitted to an absent channel -- then, once the ramp starts, it re-aims
        #       by a lagging integer shift, which on a PERMUTATION channel is measured
        #       to be WORSE than not compensating at all (Phase 1: beta=0.5 -> 12.5 vs
        #       13.0 for beta=0; guide III.5);
        #   (2) `conf` becomes the ONLY time-varying appended feature during the
        #       warmup, so the arm is no longer input-identical to blind on a task
        #       that is byte-identical to stock SMAC -- which is exactly the warmup
        #       confound guide II.6 says destroys the comparison.
        # Frozen, P stays at p0*I, so conf == 1/(1+r) exactly and the whole appended
        # block is a CONSTANT vector -- functionally equivalent to no append at all --
        # and the estimator enters the ramp genuinely cold (conf 0.33 < thresh 0.5),
        # which is the designed "do not compensate before you have data" behaviour.
        # Set 0 to ablate (reproduces the old, poisoned behaviour).
        self.pact1_warmup_freeze = int(a.get("pact1_warmup_freeze", 1))
        # *** WHICH CONFIDENCE GATES THE COMPENSATOR.  DEFAULT "pred". ***
        #   "pred"  -> 1/(1 + r*psi^T P psi/(p0*||psi||^2)): uncertainty in the
        #              PREDICTION beta_hat.psi, which is the only thing the re-aim
        #              uses.  Same range and same threshold semantics as "trace"
        #              (cold = 1/(1+r), converged -> 1).
        #   "trace" -> the original 1/(1 + tr(P)/p0).  Dominated by the LEAST excited
        #              direction, which RLS-with-forgetting inflates by 1/mu on every
        #              update, unbounded.  On SMAC the regressor is near-degenerate by
        #              construction (Phi=alive makes B_same and B_cross both track
        #              squad size -- guide III.6), so tr(P) grows over a run even
        #              while the prediction stays good, and the gate eventually
        #              DISARMS a working compensator.  Measured: conf 0.75 -> 0.44
        #              over 1.8M steps and still falling, against a 0.5 threshold.
        # Watch p1_psi_cond in pact_debug.csv: if it is large, "trace" is guaranteed
        # to decay to zero and "pred" is the only readout that means anything.
        self.pact1_conf_mode = str(a.get("pact1_conf_mode", "pred")).lower()
        assert self.pact1_conf_mode in ("pred", "trace"), (
            f"pact1_conf_mode must be 'pred' or 'trace' (got {self.pact1_conf_mode!r})"
        )
        # *** DIRECTIONAL FORGETTING.  DEFAULT ON.  See AgentRLS.update. ***  0 = the
        # original scalar forgetting, which winds the covariance up without bound on
        # a near-degenerate regressor and is what blew beta_hat to 14x truth at the
        # curriculum ramp on 3s5z.
        self.pact1_df = int(a.get("pact1_df", 1))
        # *** THE RESOLVABILITY GATE.  See AgentRLS.resolves. ***  Re-aim only when
        # the unit's own realized prediction error is below this fraction of one
        # quantum (1/(k-1)) on its current target list.  0.5 = "my typical error is
        # under half a place".  Set <=0 to disable (restores the old covariance-only
        # gate and with it the ability to do worse than blind).
        self.pact1_resolve = float(a.get("pact1_resolve", 0.5))
        self.pact1_ctde = int(a.get("pact1_ctde", 0))             # true A -> critic only
        # de-phasing: this env's starting point on the driver cycle, in [0,1).  Set by
        # make_train_env / make_eval_env from the worker rank so the ensemble tiles the
        # cycle (see the _DEPHASE note); 0 reproduces the old all-in-phase behaviour.
        phase = float(a.get("snd_phase", 0.0)) if _DEPHASE else 0.0
        self._snd_phase0 = int((phase % 1.0) * _PERIOD)
        assert sum([bool(self.snd_oracle), bool(self.snd_pact), bool(self.snd_pact1)]) <= 1, (
            "oracle (true ell), pact (given x2) and pact1 (estimated beta) all fill "
            "the same obs slot; enable at most one."
        )
        if self.snd_pact1:
            assert _KNEE == 0.0 and _LMAX >= 1.0, (
                f"PACT-1 needs the LINEAR channel ell = c*x2 for the estimator's "
                f"regression to be well posed, i.e. SMAC_SND_KNEE=0 and "
                f"SMAC_SND_LMAX>=1 (got knee={_KNEE}, lmax={_LMAX}). With a knee or a "
                f"cap the map from psi to ell is piecewise and beta* is not "
                f"identifiable from deflection readings alone."
            )

    def _snd_banner(self):
        global _NS_BANNER_SHOWN
        if _NS_BANNER_SHOWN:
            return
        _NS_BANNER_SHOWN = True
        mode = "blind"
        if self.snd_oracle:
            mode = "ORACLE (true ell in obs+state)"
        elif self.snd_pact1:
            mode = "PACT-1 (beta_hat estimated online%s)" % (
                ", env pre-shifts" if self.pact1_assist else ", obs only"
            ) + (" +CTDE" if self.pact1_ctde else "") + (
                ", RLS FROZEN at sigma=0" if self.pact1_warmup_freeze
                else ", RLS RUNS at sigma=0 (WARMUP CONFOUND -- see pact1_warmup_freeze)"
            ) + (
                f"; mu={self.pact1_forget} p0={self.pact1_p0} "
                f"df={self.pact1_df} resolve={self.pact1_resolve} kref={_P1KREF:g}"
            )
        elif self.snd_pact:
            mode = "PACT (computed x2%s in obs+state)" % (
                "+x3" if self.snd_pact_feedback else ""
            ) + (" +CTDE" if self.snd_pact_ctde else "")
        curr = (f"warmup {_WARMUP}+ramp {_RAMP} steps/env" if _WARMUP > 0 else "off")
        # How hard the channel bites a BLIND team at the driver peak.  Because the
        # deflection is a permutation, a compensating team pays nothing -- the ceiling
        # is B0 itself (T2 conjugacy) -- so the only quantity to size is the harm.
        ell_peak = min(_LMAX, self.snd_severity * max(0.0, _GREEDY_LOAD - _KNEE)
                       / max(1e-6, 1.0 - _KNEE))
        print(
            f"[NS] SMAC CTI  severity={self.snd_severity}  mode={mode}  "
            f"curriculum={curr}  eval={self.snd_eval}  dephase={int(_DEPHASE)}  "
            f"phi={_PHI}  knee={_KNEE} lmax={_LMAX}  period={_PERIOD}  "
            f"dither={int(_DITHER)}"
            f"   (severity 0 == stock SMAC)",
            flush=True,
        )
        if _PHI == "alive":
            print("[NS] phi=alive -> the load is UNCANCELLABLE: a unit loads the bus "
                  "just by being alive,\n     so the team cannot dodge the NS by "
                  "firing less or disengaging. The only\n     remaining mitigation is "
                  "to compensate the deflection.", flush=True)
        else:
            print(f"[NS] *** WARNING: phi={_PHI} leaves a behavioural ESCAPE open "
                  f"({'stop shooting' if _PHI == 'fire' else 'disengage'}), which a "
                  f"blind\n     baseline will find instead of learning to compensate. "
                  f"Use phi=alive unless you are\n     deliberately measuring the "
                  f"escape.", flush=True)
        if self.snd_severity > 0.0:
            print(
                f"[NS] @ driver peak, at a stationary team's measured load "
                f"{_GREEDY_LOAD:.2f}: deflection ell={ell_peak:.2f} -- a blind shot "
                f"lands ~{ell_peak:.0%} of the way around its target list. The channel "
                f"is a PERMUTATION, so re-aiming is free and the compensation ceiling "
                f"is B0 itself (T2 conjugacy).",
                flush=True,
            )
            if ell_peak < 0.25:
                print(f"[NS] *** WARNING: deflection {ell_peak:.2f} is small; a blind "
                      f"team may barely be hurt and the arms will not separate. Raise "
                      f"SEVERITY. Certify with `python -m harl.envs.smac.pact.phase1`.",
                      flush=True)
        if _PACT1_MIX:
            a_, b_ = _p1_theta_anchors(_PACT1_SEED, _PACT1_RADIUS, _PACT1_CONC, _P1R)
            print(
                f"[NS] MIX=ON seed={_PACT1_SEED} period={_PACT1_PERIOD} "
                f"radius={_PACT1_RADIUS} conc={_PACT1_CONC} loop={_PACT1_LOOP}\n"
                f"     interference split theta slides between "
                f"[{a_[0]:.3f} {a_[1]:.3f}] and [{b_[0]:.3f} {b_[1]:.3f}] "
                f"(same-type, cross-type).\n"
                f"     W(theta) is HIDDEN; the agent knows only that the two channels "
                f"exist.",
                flush=True,
            )
        else:
            print("[NS] MIX=OFF -> theta pinned at (0.5,0.5); the shared load is "
                  "byte-identical to the pre-PACT-1 env.", flush=True)

    # ------------------------------------------------------------- PACT-1 --
    def _pact1_init_state(self):
        """Per-instance PACT-1 state.  All of it is either shared-engagement
        arithmetic or the unit's own observation of its own shots -- nothing here
        reads self._cwo_ell (the true liability), which is the hygiene line."""
        n = self.n_agents
        self._p1_x = np.zeros((_P1R, n))        # per-basis leaky accumulators
        self._p1_psi = np.zeros((_P1R, n))      # regressor ALIGNED with the applied ell
        self._p1_types = np.full(n, -1, dtype=np.int64)
        self._p1_rls = [
            _P1RLS(_P1R, self.pact1_forget, self.pact1_p0,
                   directional=bool(self.pact1_df))
            for _ in range(n)
        ]
        self._p1_beta = np.zeros((n, _P1R))     # beta_hat per agent
        # cold-prior confidence, so the value in the obs is meaningful from step 1
        # rather than a spurious 0 until the first _snd_step refreshes it
        self._p1_conf = np.full(n, self._p1_rls[0].confidence())
        self._p1_ellhat = np.zeros(n)           # predicted deflection for NEXT step
        self._p1_shat = np.zeros(n, dtype=np.int64)     # pre-shift applied this step
        self._p1_sobs = np.full(n, -1.0)        # observed true shift this step (-1 = none)
        self._p1_kobs = np.zeros(n, dtype=np.int64)     # attackable enemies this step
        self._p1_ellmeas = np.zeros(n)          # last sensor reading
        self._p1_th_a, self._p1_th_b = _p1_theta_anchors(
            _PACT1_SEED, _PACT1_RADIUS, _PACT1_CONC, _P1R
        )
        self._p1_theta = _P1THETA_LEGACY.copy()
        # The severity that produced the ell currently sitting in self._cwo_ell, i.e.
        # the one the NEXT batch of sensor readings will be measuring.  The warmup
        # freeze gates on THIS, not on the current step's sigma, so the gate is exact
        # across the ramp boundary instead of off by one step.
        self._p1_sigma_ell = 0.0
        self._p1_frozen = 0.0        # 1 while the estimator is frozen (diagnostic)
        self._p1_n_upd = 0           # RLS updates actually applied, this instance
        # Running Gram of the regressor, for cond(E[psi psi^T]) -- guide III.6 says
        # report it before claiming theta is DECOMPOSED rather than merely predicted.
        # SMAC with near-constant engagement is the ill-conditioned case.
        self._p1_gram = np.zeros((_P1R, _P1R))
        self._p1_gram_n = 0

    def _pact1_theta(self):
        """theta(t): how the squad's emissions split across the two channels.
        Pinned at the legacy (1/2, 1/2) when mixing is off, which with MIXNORM=2
        reproduces the pre-PACT-1 load exactly."""
        if not _PACT1_MIX:
            return _P1THETA_LEGACY
        return _p1_theta_at(self._snd_clock, _PACT1_PERIOD,
                            self._p1_th_a, self._p1_th_b)

    def _pact1_refresh_types(self):
        """Cache each unit's type id (stalker vs zealot on 3s5z).  Squad composition
        is visible to the team, so this is not privileged -- it is which channel a
        teammate emits on, not how hard that channel loads the bus."""
        for j in range(self.n_agents):
            u = self.get_unit_by_id(j)
            if u is not None:
                self._p1_types[j] = int(u.unit_type)

    def _pact1_refresh_conf(self):
        """Copy beta_hat and the self-confidence out of the per-agent RLS objects.

        The confidence is evaluated along THIS unit's current regressor when
        pact1_conf_mode == "pred" -- see the knob for why tr(P) is the wrong gate on
        a near-degenerate regressor."""
        pred = self.pact1_conf_mode == "pred"
        for i in range(self.n_agents):
            self._p1_beta[i] = self._p1_rls[i].beta
            self._p1_conf[i] = (
                self._p1_rls[i].confidence_pred(self._p1_psi[:, i]) if pred
                else self._p1_rls[i].confidence()
            )

    def _pact1_observe(self):
        """RLS update from THIS step's deflection readings.

        A unit that fired at one of K>1 attackable enemies sees where its shot
        actually landed, and knows the pre-shift s_hat it applied itself, so it can
        reconstruct the true displacement s and read off  ell_meas = s/(K-1).  The
        regressor is self._p1_psi, which is the psi that PRODUCED the ell applied
        this step -- so the pair is time-aligned.  Quantisation error <= 0.5/(K-1).

        FROZEN while the ell being measured was produced at severity 0: y == 0 there
        for every unit whatever psi does, which is the absence of the channel rather
        than evidence about beta*.  See pact1_warmup_freeze in _snd_resolve_knobs for
        the two measured harms of not freezing."""
        frozen = bool(self.pact1_warmup_freeze) and self._p1_sigma_ell <= 0.0
        self._p1_frozen = 1.0 if frozen else 0.0
        if frozen:
            # Still refresh the readouts so the obs block is well defined, but leave
            # beta_hat and P untouched.  P stays at p0*I, so BOTH confidence forms
            # return exactly 1/(1+r) for any psi and the whole appended block is a
            # constant vector -- input-equivalent to blind, which is the point.
            self._pact1_refresh_conf()
            return
        for i in range(self.n_agents):
            k = int(self._p1_kobs[i])
            if k <= 1 or self._p1_sobs[i] < 0.0:
                continue                        # no shot, or no choice of target
            y = _p1_ell_from_shift(self._p1_sobs[i], k)
            if y is None:
                continue
            self._p1_ellmeas[i] = y
            # Measurement variance of THIS reading, relative to a reference target
            # list of _P1KREF enemies.  Rounding to the nearest of (k-1) places is a
            # uniform quantiser of step 1/(k-1), so var ~ (1/(k-1))^2/12; only the
            # RATIO matters to RLS, so it is normalised at k = _P1KREF (var = 1).
            # A 2-target reading is then 16x less trusted than a 5-target one and
            # 64x less than a 9-target one -- which is the truth, and ignoring it is
            # what let the coarse near-zero readings of the early ramp dominate.
            var = ((_P1KREF - 1.0) / float(k - 1)) ** 2
            self._p1_rls[i].update(self._p1_psi[:, i], y, var)
            self._p1_n_upd += 1
        self._pact1_refresh_conf()

    def _pact1_advance(self, exert, denom, alive):
        """Run the per-basis leak, rebuild psi, and return the TRUE shared load x2.

        Returns x2 (n,).  Also refreshes self._p1_psi (aligned with the ell that this
        x2 will produce) and self._p1_ellhat (what each unit PREDICTS it will suffer
        next step, from its own beta_hat -- the quantity the compensator uses)."""
        same, cross = _p1_type_split(exert, self._p1_types, denom)
        B = np.stack([same, cross], axis=0)                  # (r, n)
        self._p1_x = _RHO * self._p1_x + (1.0 - _RHO) * B
        psi = _P1MIXNORM * self._p1_x                        # (r, n)
        th = self._pact1_theta()
        self._p1_theta = th
        x2 = (th[:, None] * psi).sum(axis=0)
        self._p1_psi = psi
        for i in range(self.n_agents):
            self._p1_ellhat[i] = max(0.0, _p1_predict_ell(self._p1_beta[i], psi[:, i]))
        self._p1_ellhat[~alive] = 0.0
        # re-evaluate the confidence against the psi the compensator will actually
        # use next step (no-op under conf_mode="trace"; exactly 1/(1+r) while frozen,
        # for any psi, so the appended block stays constant through the warmup)
        self._pact1_refresh_conf()
        return x2.astype(np.float32)

    def _pact1_diag(self):
        """PACT-1 telemetry for pact_debug.csv.  Logging only -- beta_true and the
        tracking error are the same class of privileged column as pcr_beta_true on
        Ant and must never reach a policy input."""
        beta_true = float(self._snd_payload) * self._snd_sigma_applied * self._p1_theta
        beta_hat = self._p1_beta.mean(axis=0)
        fired = self._p1_kobs > 1
        # cond(E[psi psi^T]) -- guide III.6.  On SMAC the two channels are driven by
        # attrition of two unit types, so this says whether the SPLIT is identifiable
        # at all or only the projection beta*.psi is.  Report it before claiming to
        # have identified theta.  Cheap: r=2.
        psi_cond = float("nan")
        psi_lmin = float("nan")
        if self._p1_gram_n > 0:
            # de-bias the EMA so the first few thousand steps are not scaled down
            G = self._p1_gram / max(1e-12, 1.0 - 0.999 ** self._p1_gram_n)
            w = np.linalg.eigvalsh(0.5 * (G + G.T))
            lo, hi = float(w[0]), float(w[-1])
            psi_lmin = lo
            psi_cond = (hi / lo) if lo > 1e-12 else float("inf")
        # the appended block exactly as _snd_augment builds it: (n_agents, 1+r+1+1)
        aug = np.stack(
            [self._p1_ellhat]
            + [self._p1_beta[:, m] for m in range(_P1R)]
            + [self._p1_conf, self._p1_ellmeas],
            axis=1,
        )
        return {
            "p1_ellhat": float(self._p1_ellhat.mean()),
            "p1_conf": float(self._p1_conf.mean()),
            "p1_conf_min": float(self._p1_conf.min()),
            "p1_conf_max": float(self._p1_conf.max()),
            "p1_beta_err": float(np.linalg.norm(beta_hat - beta_true)),
            "p1_beta_hat0": float(beta_hat[0]),
            "p1_beta_hat1": float(beta_hat[1]) if _P1R > 1 else float("nan"),
            "p1_beta_true0": float(beta_true[0]),
            "p1_beta_true1": float(beta_true[1]) if _P1R > 1 else float("nan"),
            # is the estimator frozen (curriculum sigma == 0), and has it ever run?
            "p1_frozen": float(self._p1_frozen),
            "p1_n_upd": float(self._p1_n_upd),
            # *** THE THREE COLUMNS THAT WOULD HAVE CAUGHT THE RAMP BLOW-UP. ***
            # p1_trP    tr(P): THE WINDUP DETECTOR.  Bounded by p0*r by construction
            #           now; if it ever pins at that bound the regressor has gone
            #           degenerate and the estimate is being held up by the cap.
            # p1_innov  the estimator's own realized |prediction error|, in units of
            #           ell.  This is what the arming gate reads.  It must fall below
            #           one quantum 1/(k-1) (~0.12-0.5 on 3s5z) or the compensator
            #           correctly refuses to act.
            # p1_ell_ratio  ell_hat / ell_true: the single number that screams.  It
            #           hit 9.0 at the ramp while p1_conf still read 0.92.  Healthy
            #           is ~1; > 2 means the compensator is inventing deflection.
            "p1_trP": float(np.mean([np.trace(r.P) for r in self._p1_rls])),
            "p1_innov": float(np.mean([r.innov_ema for r in self._p1_rls])),
            # GUARD HARD, NOT WITH AN EPSILON (guide II.7).  At the driver trough
            # ell is genuinely ~0, so the ratio is meaningless there -- a 1e-6 floor
            # let trough steps dominate the column average and it read 300-900x on a
            # run whose estimator was merely mistracking by 2x.  NaN is dropped by
            # the runner's _m(); 0.05 is ~10% of the peak deflection.
            "p1_ell_ratio": (
                float(self._p1_ellhat.mean() / self._cwo_ell.mean())
                if float(self._cwo_ell.mean()) > 0.05 else float("nan")
            ),
            # over the units that actually took a shot -- k is what sets the quantum,
            # so a unit with no target list has no gate to be open or shut
            "p1_armed": (float(np.mean([
                1.0 if (self.pact1_resolve <= 0.0
                        or self._p1_rls[i].resolves(int(self._p1_kobs[i]),
                                                    self.pact1_resolve))
                else 0.0
                for i in np.where(fired)[0]
            ])) if fired.any() else float("nan")),
            # the regressor itself: if psi is ~0 the sensor has nothing to regress on
            "p1_psi_norm": float(np.linalg.norm(self._p1_psi, axis=0).mean()),
            "p1_psi_cond": psi_cond,
            "p1_psi_lmin": psi_lmin,
            # *** WHAT THE POLICY ACTUALLY SEES -- THE AUGMENTATION CONTROL. ***
            # The append is the ONLY thing that differs from blind at severity 0, so
            # measure it instead of arguing about it.  aug_var is the variance ACROSS
            # UNITS of each appended component, averaged over components; combined
            # with p1_conf being pinned flat over time it certifies that the block is
            # a CONSTANT vector, which carries zero information (a LayerNorm + linear
            # first layer absorbs a constant input exactly).  Frozen, this must read
            # 0.0 and p1_conf must read 1/(1+r) forever.  Anything else means the arm
            # was never input-equivalent to blind during the warmup.
            "p1_aug_absmean": float(np.abs(aug).mean()),
            "p1_aug_var": float(aug.var(axis=0).mean()),
            "p1_shat": float(self._p1_shat.mean()),
            "p1_obs_frac": float(fired.mean()),   # how often the sensor fires at all
            # net displacement AFTER compensation: 0 == the shot landed where aimed
            "p1_net_shift": float(
                np.mean(np.abs(self._p1_sobs[fired] - self._p1_shat[fired]))
            ) if fired.any() else 0.0,
            "p1_raw_shift": float(
                np.mean(np.abs(self._p1_sobs[fired]))
            ) if fired.any() else 0.0,
            # THE headline: fraction of the deflection actually cancelled by the
            # re-aim.  Mirrors Ant's pact1_cancel_frac exactly, including the guard:
            # when the driver is at its trough the raw shift is genuinely ~0, so the
            # ratio is meaningless.  Return NaN, which the runner's _m() drops -- an
            # epsilon floor there produced a -1011 on Ant that poisoned the column
            # average until it was caught.
            "p1_cancel": self._pact1_cancel(fired),
        }

    def _pact1_cancel(self, fired):
        """1 - |s - s_hat| / |s|, over units that actually took a shot with K>1."""
        if not fired.any():
            return float("nan")
        raw = float(np.mean(np.abs(self._p1_sobs[fired])))
        if raw <= 1e-3:
            return float("nan")
        net = float(np.mean(np.abs(self._p1_sobs[fired] - self._p1_shat[fired])))
        return 1.0 - net / raw

    def _snd_grow_spaces(self):
        """Grow the declared obs/state sizes by the append: each agent's obs gets its
        own scalars (x2_i [+ x3_i under PACT feedback], or ell_i under the oracle) and
        the centralized state gets every unit's, plus +1 to the state under PACT+CTDE
        (the true driver A)."""
        if self.snd_oracle:
            add_obs, add_state = 1, self.n_agents
        elif self.snd_pact1:
            # [ell_hat_i, beta_hat_i (r), conf_i, last ell_meas_i] -- the estimator's
            # state, all of it computed from shared engagement + this unit's own shots
            add_obs = 1 + _P1R + 1 + 1
            # *** THE STATE GETS AGGREGATES, NOT PER-AGENT COPIES. ***
            # HARL's MLPBase applies nn.LayerNorm(obs_dim), which normalises ACROSS
            # the feature dimension -- so appended features do not merely add inputs,
            # they shift the mean/std that every OTHER feature is normalised by.
            # Appending add_obs*n_agents = 40 dims to the EP state was ~20-30% of the
            # critic's input, and during a severity-0 warmup nearly all of it is
            # identically zero: it shrinks every real feature and injects a constant
            # offset into 40 units. Measured cost on 3s5z: PACT-1 sat at win 0.000
            # through 1.1M while blind, on the byte-identical stationary task, reached
            # 0.24.
            # The per-agent values are near-redundant anyway (x2_i differs across
            # agents only by the excluded own term, <= (1-RHO)/(N-1) = 0.021), so the
            # aggregate carries essentially the same information at 1/8 the distortion.
            add_state = add_obs + (1 if self.pact1_ctde else 0)
        elif self.snd_pact:
            add_obs = 1 + (2 if self.snd_pact_feedback else 0)  # x2 [, x3_jam, x3_try]
            add_state = add_obs * self.n_agents + (1 if self.snd_pact_ctde else 0)
        else:
            return
        for sp in self.observation_space:
            sp[0] += add_obs
        for sp in self.share_observation_space:
            sp[0] += add_state

    def _snd_driver_value(self):
        """A(t): the live raised-cosine tempo, or a frozen constant if snd_freeze is
        set (used only by the calibration/diagnose scripts to hold the driver)."""
        if self.snd_freeze is not None:
            return float(self.snd_freeze)
        return _driver(self._snd_clock)

    def _curr_severity(self):
        """The severity applied THIS step (the CURRICULUM).  EVAL envs (and severity 0)
        use the full target so eval measures the harmed win rate.  TRAIN envs ramp from
        0 to SEVERITY -- a warmup that lets the policy learn to WIN before the NS bites
        -- driven by this env's own AGE (self._snd_age), deliberately NOT the driver
        clock, so per-rank de-phasing cannot shift the warmup boundary between parallel
        envs.  _WARMUP=0 disables it."""
        if self.snd_eval or self.snd_severity == 0.0 or _WARMUP <= 0:
            return self.snd_severity
        frac = (self._snd_age - _WARMUP) / float(max(1, _RAMP))
        return self.snd_severity * float(min(1.0, max(0.0, frac)))

    def _snd_augment(self, local_obs, global_state):
        """Append the CWO block to the RETURNED obs/state (no-op if neither oracle nor
        pact is on).  Each agent's OBS gets ONLY its own scalars (own x2_i [, x3_i] or
        ell_i) -- decentralized; the centralized STATE gets all units' (plus, for
        PACT+CTDE, the true driver A).  Sizes were grown to match in
        _snd_grow_spaces()."""
        if self.snd_oracle:
            blocks = [self._cwo_ell]           # the TRUE liability (privileged)
        elif self.snd_pact1:
            # the ESTIMATOR's state -- never the true liability.  ell_hat is what the
            # unit predicts from its own beta_hat; conf is the RLS covariance readout;
            # ell_meas is its last observation of its own deflected shot.
            blocks = [
                self._p1_ellhat.astype(np.float32),
                *[self._p1_beta[:, m].astype(np.float32) for m in range(_P1R)],
                self._p1_conf.astype(np.float32),
                self._p1_ellmeas.astype(np.float32),
            ]
        elif self.snd_pact:
            blocks = [self._cwo_x2]            # the COMPUTED shared load (the method)
            if self.snd_pact_feedback:
                blocks.append(self._cwo_x3)      # own shots jammed lately   } the local
                blocks.append(self._cwo_x3try)   # own shots attempted lately} residual
        else:
            return local_obs, global_state
        per_agent = np.stack(blocks, axis=1).astype(np.float32)   # (n_agents, k)
        local_obs = [
            np.append(np.asarray(o, dtype=np.float32), per_agent[i]).astype(np.float32)
            for i, o in enumerate(local_obs)
        ]
        if self.snd_pact1:
            # aggregate, not per-agent copies -- see _snd_grow_spaces for why
            g = per_agent.mean(axis=0).astype(np.float32)
        else:
            g = per_agent.flatten()
        if (self.snd_pact and self.snd_pact_ctde) or (self.snd_pact1 and self.pact1_ctde):
            g = np.append(g, np.float32(self._snd_payload))
        global_state = [
            np.append(np.asarray(s, dtype=np.float32), g).astype(np.float32)
            for s in global_state
        ]
        return local_obs, global_state

    def _cwo_fill_diag(self, alive, fire):
        """Fill self._cwo_diag with the per-step debug telemetry (copied into info).

        Uses the liability APPLIED this step (self._cwo_ell / _cwo_x2, before the
        advance below) + this step's commanded attacks (`fire`), this step's dropped
        shots (self._cwo_dropped) and whether an attack was even AVAILABLE
        (self._cwo_can_fire, recorded in get_agent_action).

        READ `cwo_hold_frac` AND THE PHASE SPLIT, NOT `cwo_fire_hi_load`.

        * cwo_hold_frac = 1 - (units that attacked)/(units that COULD attack) is the
          real decision variable.  Raw fire_frac is diluted by units with no enemy in
          range, which has nothing to do with coordination.
        * cwo_throughput = fire_frac * (1 - drop_frac) is the quantity the team is
          actually trying to maximize -- the fraction of units landing a shot.  A team
          that is coordinating correctly RAISES this while LOWERING fire_frac.
        * the honest coordination statistic is hold_frac at the driver PEAK minus at
          the TROUGH (the runner's phase split): the team must hold fire more when the
          bus is hot.  It is 0 by construction at severity 0.

        `cwo_fire_hi_load` / `cwo_fire_lo_load` are kept for continuity but are
        BIASED and must not be read as coordination: x2_i excludes agent i's own fire,
        so ranking agents by x2_i is very nearly REVERSE-ranking them by their own
        recent firing, and firing is strongly autocorrelated -- so fire_lo > fire_hi
        comes out positive with zero coordination.  Measured on the 20M-step 3s5z run,
        `stagger_gap` was +0.16 throughout the severity-0 warmup, where no NS exists at
        all.  cwo_x2_spread (max-min over live units) shows why the split is degenerate
        in the first place."""
        n = self.n_agents
        av = np.where(alive)[0]
        na = int(av.size)
        ell, x2 = self._cwo_ell, self._cwo_x2
        commanded = fire > 0.5
        n_cmd = int(commanded[alive].sum()) if na else 0
        can = self._cwo_can_fire > 0.5
        n_can = int((can & alive).sum())
        fire_hi = fire_lo = float("nan")
        if na >= 2:
            thr = float(np.median(x2[av]))
            hi, lo = av[x2[av] > thr], av[x2[av] <= thr]
            if hi.size:
                fire_hi = float(fire[hi].mean())
            if lo.size:
                fire_lo = float(fire[lo].mean())
        fire_frac = float(commanded[alive].mean()) if na else 0.0
        drop_frac = float(self._cwo_dropped[commanded].mean()) if n_cmd else 0.0
        self._cwo_diag = {
            "cwo_ell_mean": float(ell[av].mean()) if na else 0.0,   # mean drop prob felt
            "cwo_ell_max": float(ell.max()) if n else 0.0,
            "cwo_x2_mean": float(x2[av].mean()) if na else 0.0,     # mean shared load
            "cwo_x2_spread": (                       # how much x2 differs ACROSS agents
                float(x2[av].max() - x2[av].min()) if na else 0.0
            ),
            "cwo_x3_mean": float(self._cwo_x3[av].mean()) if na else 0.0,   # own jams
            "cwo_x3try_mean": float(self._cwo_x3try[av].mean()) if na else 0.0,  # attempts
            "cwo_fire_frac": fire_frac,                             # frac attacking
            "cwo_fire_avail": (float(n_can) / na) if na else 0.0,   # frac that COULD attack
            "cwo_hold_frac": (                # frac of ABLE units that chose to hold fire
                1.0 - float((commanded & can & alive).sum()) / n_can if n_can else float("nan")
            ),
            "cwo_drop_frac": drop_frac,                              # frac of shots jammed
            "cwo_throughput": fire_frac * (1.0 - drop_frac),         # shots actually LANDED
            "cwo_fire_hi_load": fire_hi,   # BIASED -- see the docstring, do not read as
            "cwo_fire_lo_load": fire_lo,   # coordination; use cwo_hold_frac by phase
            # --- IS THE TEAM FIGHTING OR FARMING?  (the sigma=0 basin, see below) ---
            # ally_dead / enemy_dead are the honest read on whether a long episode is
            # a hard-fought battle or a standoff: in the timeout basin BOTH stay low
            # while ep_len pins at the limit and reward keeps accruing.
            "cwo_ally_dead": 1.0 - (float(na) / float(n)) if n else 0.0,
            "cwo_enemy_dead": (
                float(self.death_tracker_enemy.mean())
                if getattr(self, "death_tracker_enemy", None) is not None
                and np.size(self.death_tracker_enemy) else 0.0
            ),
            # shield-regeneration pay, in the SAME units as the logged reward, so it
            # can be read directly as a fraction of r_step_mean (see reward_battle)
            "cwo_regen_pay": (
                float(getattr(self, "_cwo_regen_pay", 0.0))
                / max(1e-9, self.max_reward / self.reward_scale_rate)
                if self.reward_scale else float(getattr(self, "_cwo_regen_pay", 0.0))
            ),
        }

    def _snd_step(self, actions_int):
        """Advance the engagement tempo A(t), the shared load x2_i and the drop
        probability ell_i, using THIS step's commanded attacks.  The values set here
        are the ones the NEXT step applies, so an oracle/PACT policy that reads them
        can pre-empt them.  Also fills the debug telemetry.  The clock persists."""
        self._snd_clock += 1
        self._snd_age += 1     # curriculum clock: never de-phased (see _snd_phase0)
        A = self._snd_driver_value()
        self._snd_payload = A
        n = self.n_agents
        n_no_attack = int(getattr(self, "n_actions_no_attack", 6))

        alive = np.zeros(n, dtype=bool)
        fire = np.zeros(n, dtype=np.float32)   # who COMMANDED an attack this step
        for j in range(n):
            unit = self.get_unit_by_id(j)
            if unit is not None and unit.health > 0:
                alive[j] = True
                if int(actions_int[j]) >= n_no_attack:
                    fire[j] = 1.0  # load is generated by ATTEMPTING to fire (even if the
                    #                shot is later deflected) -> x2 is exact from actions

        # --- WHAT FEEDS THE BUS: engagement, not trigger-pulls ---------------------
        # Exertion Phi_j = "unit j is alive and has an enemy in weapons range", i.e. it
        # is COMMITTED to the fight, rather than "unit j pulled the trigger this step".
        #
        # Trigger-pulls give the policy a cheap escape that has nothing to do with the
        # intended solution: firing less lowers x2, which lowers everyone's deflection,
        # so the NS partly SWITCHES ITSELF OFF when the team sulks.  That is exactly
        # what was measured -- at full severity the team went fire_frac 0.82 -> 0.50,
        # fire_avail 0.90 -> 0.55 and ep_len 50 -> 86, which drove ell down to 0.13 but
        # cost far more damage than the deflection ever did (win 0.44 against a
        # stationary ~0.85).  It learned to stop fighting instead of to re-aim.
        #
        # Keying the load to engagement closes that hatch: the bus stays hot while the
        # squad is in contact whatever its trigger discipline, so the ONLY mitigation
        # left is the intended one -- compensate the deflection.  Still category-C:
        # a sum over j != i that is empty at N=1, and a unit never loads its own bus.
        # (Set SMAC_SND_PHI=fire to restore the old trigger-pull exertion.)
        if _PHI == "alive":
            # UNCANCELLABLE: a powered unit loads the bus whatever it does.  The only
            # way down is to lose units, which costs the battle -- no behavioural
            # dodge remains.  (Ant's analogue: key the load to |tau|, not to signed
            # tau, so an anti-symmetric gait cannot null it.)
            exert = alive.astype(np.float32)
        elif _PHI_ENGAGE:
            exert = np.where(alive, self._cwo_can_fire, 0.0).astype(np.float32)
        else:
            exert = fire

        # debug telemetry from the liability APPLIED this step (pre-advance):
        self._cwo_fill_diag(alive, fire)

        # advance the shared load and the drop probability for the NEXT step:
        sigma = self._curr_severity()          # the CURRICULUM severity applied this step
        self._snd_sigma_applied = sigma
        denom = float(max(1, n - 1))
        if self.snd_pact1:
            # PACT-1.  ORDER MATTERS.  Estimate FIRST, from this step's deflection
            # readings against self._p1_psi (the regressor that produced the ell just
            # applied) -- only then advance the leak, or the pair goes out of
            # alignment by one step and the regression fits the wrong thing.
            self._pact1_refresh_types()
            self._pact1_observe()
            if _PACT1_LOOP > 0.0:
                # close the loop: correcting harder emits harder, and emitting harder
                # loads the bus (off by default -- see the _PACT1_LOOP note).
                exert = exert * (1.0 + _PACT1_LOOP * np.abs(self._p1_shat))
            self._cwo_x2 = self._pact1_advance(exert, denom, alive)
        else:
            S = (float(exert.sum()) - exert) / denom  # (sum_{j!=i} Phi_j)/(N-1), in [0,1]
            self._cwo_x2 = _RHO * self._cwo_x2 + (1.0 - _RHO) * S        # PACT waveform
        # The LOCAL residual that reveals the hidden driver, as TWO raw leaky counters
        # rather than one derived rate:  x3_jam = "shots of mine that jammed lately",
        # x3_try = "shots I attempted lately".  The policy forms the ratio itself, and
        # x3_try ~ 0 correctly means "I have no evidence" instead of being confused with
        # "I fired and nothing jammed".  Both decay every step on the same leak as x2.
        #
        # A single held rate does not work: holding the value when the unit does not
        # fire freezes a stale estimate forever once it stops shooting, and starting an
        # episode at 0 reads as "no jams => the driver is low" for the first ~10 of only
        # ~50 steps.  That is a large part of why the policy stayed phase-blind
        # (hold_gap ~0, and negative during the ramp).
        self._cwo_x3 = (_RHO * self._cwo_x3
                        + (1.0 - _RHO) * self._cwo_dropped).astype(np.float32)
        self._cwo_x3try = (_RHO * self._cwo_x3try
                           + (1.0 - _RHO) * fire).astype(np.float32)
        # drop prob: free below the KNEE, then linear in the excess, capped at _LMAX.
        excess = np.maximum(0.0, self._cwo_x2 - _KNEE) / max(1e-6, 1.0 - _KNEE)
        self._cwo_ell = np.clip(A * sigma * excess, 0.0, _LMAX).astype(np.float32)
        if self.snd_pact1:
            # remember what produced THIS ell: the readings it generates are consumed
            # by _pact1_observe on the NEXT step, and the warmup freeze gates on it.
            # *** Gate on the CURRICULUM sigma only, never on A. ***  A y == 0 reading
            # at the driver TROUGH is real data -- beta* = A*sigma*theta genuinely is
            # ~0 there, and the estimator has to track it down and back up.  Freezing
            # at the trough would hold a stale high beta_hat into the rise and
            # over-compensate.  Only sigma == 0 means "the channel does not exist".
            self._p1_sigma_ell = float(sigma)
            if alive.any():
                Psi = self._p1_psi[:, alive]                    # (r, n_alive)
                # EMA, not a lifetime sum: conditioning is a property of the CURRENT
                # behaviour (a policy that stops fighting degenerates the regressor),
                # and a lifetime average would hide that behind early data.
                w = 0.999
                self._p1_gram = w * self._p1_gram + (1.0 - w) * (
                    (Psi @ Psi.T) / float(Psi.shape[1])
                )
                self._p1_gram_n += 1
        self._cwo_x2[~alive] = 0.0
        self._cwo_x3[~alive] = 0.0
        self._cwo_x3try[~alive] = 0.0
        self._cwo_ell[~alive] = 0.0
        if self.snd_pact1:
            self._p1_x[:, ~alive] = 0.0
            self._p1_psi[:, ~alive] = 0.0
            self._p1_ellhat[~alive] = 0.0
            self._p1_ellmeas[~alive] = 0.0
        self._snd_load_mean = float(self._cwo_ell.mean()) if n else 0.0
        self._snd_load_max = float(self._cwo_ell.max()) if n else 0.0

    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        assert avail_actions[action] == 1, "Agent {} cannot perform action {}".format(
            a_id, action
        )
        # CWO telemetry: could this unit have attacked at all this step?  Holding fire
        # is only a DECISION for units that had a target in range -- fire_frac alone is
        # diluted by units with nothing to shoot at (see _cwo_fill_diag).
        self._cwo_can_fire[a_id] = float(
            np.any(np.asarray(avail_actions[self.n_actions_no_attack:]) > 0)
        )

        # --- CTI: an overheated targeting bus MIS-AIMS the shot.  The commanded attack
        # is DEFLECTED onto a different enemy: the delivered target is shifted `s` places
        # along this unit's own list of currently-attackable enemies, where
        #     s = round(ell_i * (K-1)),  K = number of attackable enemies
        # and ell_i is set by the OTHERS' firing in _snd_step.  Reward untouched.
        #
        # *** THIS IS AN INVERTIBLE CHANNEL, AND THAT IS THE WHOLE POINT. ***  It is a
        # deterministic PERMUTATION of the attack action, so an agent that knows s can
        # command `desired - s` and land exactly on the enemy it wanted -- at zero cost.
        # Compensation restores the stationary game byte for byte (pipeline T2
        # conjugacy), so the 0.9*B0 frontier is reachable and sigma* is well defined.
        #
        # The predecessor (CWO) dropped the shot instead, and that is why it failed
        # Phase 1 at every severity: the only response to a dropped shot is to fire
        # less, which buys damage with damage.  Measured, compensation did help
        # (sigma=1.0: return 7.9 -> 11.3, throughput 0.489 -> 0.631) but the coordinated
        # ceiling was ~0.68 against a stationary 0.88, so 0.9*B0 = 16.4 was unreachable
        # from a best of 11.3 at EVERY severity from 0.5 to 3.2.  A non-invertible harm
        # channel has no conjugacy and therefore no Phase-1 frontier.
        #
        # Blind agents are hurt badly because SMAC rewards FOCUS FIRE: a mis-aimed squad
        # spreads damage over many enemies, kills nothing, and dies to a full-strength
        # enemy line.  Nothing is physically removed -- the damage still lands.
        tgts = np.where(np.asarray(avail_actions[self.n_actions_no_attack:]) > 0)[0]
        k = int(tgts.size)

        # --- PACT-1 COMPENSATION: pre-shift the commanded target BACKWARD by the
        # deflection this unit predicts it is about to suffer, so the channel's
        # forward shift lands the shot where the policy actually aimed.  This is the
        # real channel inverse (a permutation), not an obs hint -- when s_hat == s it
        # cancels exactly and at zero cost.
        #
        # The gain is the estimator's own confidence times a fixed multiplier: a cold
        # RLS compensates little, a converged one compensates fully, with no hand-set
        # warmup.  ell_hat comes from THIS unit's beta_hat, never from self._cwo_ell.
        # *** TRUST IS A THRESHOLD HERE, NOT A SCALE -- and that is a MEASURED
        # property of the channel, not a preference. ***  The shift is an INTEGER, so
        # partial compensation does not buy partial recovery: it lands the shot on a
        # DIFFERENT wrong enemy.  Phase 1 measured it directly -- at sigma=0.3,
        # beta=0.5 returned 12.5 against 13.0 for no compensation at all, while
        # re-aiming 62% of shots; beta=1.0 returned 17.5.  Scaling the shift by a
        # ramping confidence (the natural choice on Ant's additive channel, where
        # partial cancellation IS partially useful) would spend the whole warmup in
        # exactly that harmful regime.  So: compensate FULLY once the estimator is
        # confident enough, and not at all before.
        #
        # *** AND IT IS GATED ON RESOLVABILITY, NOT ONLY ON CONFIDENCE. ***  The
        # covariance says how uncertain the parameter is; it cannot say whether the
        # INTEGER shift will come out right, which is the only thing that matters on
        # a permutation channel.  On the 3s5z run `conf` read 0.92 while the
        # estimator was predicting 9x the true deflection, and the compensator
        # re-aimed by 1-3 places when the true shift was 0 -- net_shift 0.70 against
        # raw_shift 0.07.  AgentRLS.resolves() gates on the unit's own realized
        # residual against one quantum of its current target list, which is the
        # question actually being asked, and it restores the floor property: gate
        # shut => the executed action is the policy's own => never worse than blind.
        s_hat = 0
        if (
            self.snd_pact1 and self.pact1_assist
            and action >= self.n_actions_no_attack and k > 1
            and float(self._p1_conf[a_id]) >= self.pact1_conf_thresh
            and (
                self.pact1_resolve <= 0.0
                or self._p1_rls[a_id].resolves(k, self.pact1_resolve)
            )
        ):
            s_hat = _p1_shift_from_ell(
                self.pact1_gpol * float(self._p1_ellhat[a_id]), k,
                float(self._cwo_dither[a_id]) if _DITHER else None,
            )
            if s_hat > 0:
                cur0 = int(action - self.n_actions_no_attack)
                w0 = np.where(tgts == cur0)[0]
                if w0.size:
                    action = self.n_actions_no_attack + int(
                        tgts[(int(w0[0]) - s_hat) % k]
                    )
        if self.snd_pact1:
            self._p1_shat[a_id] = s_hat

        if (
            self.snd_severity != 0.0
            and action >= self.n_actions_no_attack
            and float(self._cwo_ell[a_id]) > 0.0
        ):
            if k > 1:
                # SAME u as the pre-shift above: with ell_hat == ell the two shifts
                # are identical and the channel cancels exactly (T2 conjugacy).
                s = _p1_shift_from_ell(
                    float(self._cwo_ell[a_id]), k,
                    float(self._cwo_dither[a_id]) if _DITHER else None,
                )
                if s > 0:
                    cur = int(action - self.n_actions_no_attack)
                    pos = int(np.where(tgts == cur)[0][0])
                    action = self.n_actions_no_attack + int(tgts[(pos + s) % k])
                    self._cwo_dropped[a_id] = 1.0  # "my shot was deflected" (local feel)

        # --- PACT-1 SENSOR: record the displacement this unit actually suffered.
        # The unit sees where its shot landed and knows its own s_hat, so it can
        # reconstruct s -- unprivileged.  s == 0 is recorded too: "no deflection" is a
        # genuine low-liability reading, and dropping those would bias the estimator
        # upward.  k is needed because the quantisation depends on it.
        if self.snd_pact1 and action >= self.n_actions_no_attack and k > 1:
            ell_true = float(self._cwo_ell[a_id])
            self._p1_sobs[a_id] = float(
                _p1_shift_from_ell(
                    ell_true, k, float(self._cwo_dither[a_id]) if _DITHER else None
                ) if self.snd_severity != 0.0 else 0
            )
            self._p1_kobs[a_id] = k

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
            return None
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"], unit_tags=[tag], queue_command=False
            )
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(x=x, y=y + self._move_amount),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move North".format(a_id))

        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(x=x, y=y - self._move_amount),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move South".format(a_id))

        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(x=x + self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move East".format(a_id))

        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(x=x - self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move West".format(a_id))
        else:
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                target_unit = self.agents[target_id]
                action_name = "heal"
            else:
                target_unit = self.enemies[target_id]
                action_name = "attack"

            action_id = actions[action_name]
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False,
            )

            if self.debug:
                logging.debug(
                    "Agent {} {}s unit # {}".format(a_id, action_name, target_id)
                )

        # --- SND: add the hidden navigation drift to the DELIVERED move target ---
        # Only move commands carry a world-space target, so only maneuvering is
        # corrupted; a unit that plants and shoots (stop/attack) holds its ground.
        # This is the sole change to the dynamics; the reward is untouched.
        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    def get_agent_action_heuristic(self, a_id, action):
        unit = self.get_unit_by_id(a_id)
        tag = unit.tag

        target = self.heuristic_targets[a_id]
        if unit.unit_type == self.medivac_id:
            if (
                target is None
                or self.agents[target].health == 0
                or self.agents[target].health == self.agents[target].health_max
            ):
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for al_id, al_unit in self.agents.items():
                    if al_unit.unit_type == self.medivac_id:
                        continue
                    if al_unit.health != 0 and al_unit.health != al_unit.health_max:
                        dist = self.distance(
                            unit.pos.x, unit.pos.y, al_unit.pos.x, al_unit.pos.y
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_id = al_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions["heal"]
            target_tag = self.agents[self.heuristic_targets[a_id]].tag
        else:
            if target is None or self.enemies[target].health == 0:
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for e_id, e_unit in self.enemies.items():
                    if (
                        unit.unit_type == self.marauder_id
                        and e_unit.unit_type == self.medivac_id
                    ):
                        continue
                    if e_unit.health > 0:
                        dist = self.distance(
                            unit.pos.x, unit.pos.y, e_unit.pos.x, e_unit.pos.y
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_id = e_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions["attack"]
            target_tag = self.enemies[self.heuristic_targets[a_id]].tag

        action_num = self.heuristic_targets[a_id] + self.n_actions_no_attack

        # Check if the action is available
        if self.heuristic_rest and self.get_avail_agent_actions(a_id)[action_num] == 0:
            # Move towards the target rather than attacking/healing
            if unit.unit_type == self.medivac_id:
                target_unit = self.agents[self.heuristic_targets[a_id]]
            else:
                target_unit = self.enemies[self.heuristic_targets[a_id]]

            delta_x = target_unit.pos.x - unit.pos.x
            delta_y = target_unit.pos.y - unit.pos.y

            if abs(delta_x) > abs(delta_y):  # east or west
                if delta_x > 0:  # east
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x + self._move_amount, y=unit.pos.y
                    )
                    action_num = 4
                else:  # west
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x - self._move_amount, y=unit.pos.y
                    )
                    action_num = 5
            else:  # north or south
                if delta_y > 0:  # north
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y + self._move_amount
                    )
                    action_num = 2
                else:  # south
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y - self._move_amount
                    )
                    action_num = 3

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=target_pos,
                unit_tags=[tag],
                queue_command=False,
            )
        else:
            # Attack/heal the target
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False,
            )

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action, action_num

    def reward_battle(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        if self.reward_sparse:
            return 0

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                    self.previous_ally_units[al_id].health
                    + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += neg_scale * (
                        prev_health - al_unit.health - al_unit.shield
                    )

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                    self.previous_enemy_units[e_id].health
                    + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        if self.reward_only_positive:
            reward = abs(delta_enemy + delta_deaths)  # shield regeneration
        else:
            reward = delta_enemy + delta_deaths - delta_ally

        # --- DIAGNOSTIC ONLY: how much of this reward is SHIELD-REGENERATION PAY? ---
        # This is stock SMAC and is NOT modified here (guide I.2: the reward function
        # is untouched byte for byte).  But it is the quantitative explanation of the
        # "farm damage, never finish" timeout basin that has now eaten a run on BOTH
        # arms, so it must be MEASURED rather than assumed:
        #   delta_enemy sums (prev_health+prev_shield) - (health+shield) per enemy.
        #   Protoss shields regenerate out of combat, so on a step where the team is
        #   NOT dealing damage that sum goes NEGATIVE -- and abs() pays the team for
        #   it.  A squad that stands off and lets 8 enemies regenerate collects real
        #   positive reward for doing nothing, and collects MORE of it the longer the
        #   episode runs.  That is a gradient straight into the 150-step timeout.
        # cwo_regen_pay = the part of this step's reward that came from a negative
        # raw delta, i.e. pure regeneration payment.  Read it as a fraction of
        # r_step_mean: if it is a large share, the basin is a reward artifact and no
        # amount of method work will climb out of it.
        raw = float(delta_enemy + delta_deaths)
        self._cwo_regen_pay = float(-raw) if (self.reward_only_positive and raw < 0.0) else 0.0
        self._cwo_reward_raw = raw

        return reward

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Distance between two points."""
        return math.hypot(x2 - x1, y2 - y1)

    def unit_shoot_range(self, agent_id):
        """Returns the shooting range for an agent."""
        return 6

    def unit_sight_range(self, agent_id):
        """Returns the sight range for an agent."""
        return 9

    def unit_max_cooldown(self, unit):
        """Returns the maximal cooldown for a unit."""
        switcher = {
            self.marine_id: 15,
            self.marauder_id: 25,
            self.medivac_id: 200,  # max energy
            self.stalker_id: 35,
            self.zealot_id: 22,
            self.colossus_id: 24,
            self.hydralisk_id: 10,
            self.zergling_id: 11,
            self.baneling_id: 1,
        }
        return switcher.get(unit.unit_type, 15)

    def save_replay(self):
        """Save a replay."""
        prefix = self.replay_prefix or self.map_name
        replay_dir = self.replay_dir or ""
        replay_path = self._run_config.save_replay(
            self._controller.save_replay(), replay_dir=replay_dir, prefix=prefix
        )
        logging.info("Replay saved at: %s" % replay_path)

    def unit_max_shield(self, unit):
        """Returns maximal shield for a given unit."""
        if unit.unit_type == 74 or unit.unit_type == self.stalker_id:
            return 80  # Protoss's Stalker
        if unit.unit_type == 73 or unit.unit_type == self.zealot_id:
            return 50  # Protoss's Zaelot
        if unit.unit_type == 4 or unit.unit_type == self.colossus_id:
            return 150  # Protoss's Colossus

    def can_move(self, unit, direction):
        """Whether a unit can move in a given direction."""
        m = self._move_amount / 2

        if direction == Direction.NORTH:
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == Direction.SOUTH:
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == Direction.EAST:
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else:
            x, y = int(unit.pos.x - m), int(unit.pos.y)

        if self.check_bounds(x, y) and self.pathing_grid[x, y]:
            return True

        return False

    def get_surrounding_points(self, unit, include_self=False):
        """Returns the surrounding points of the unit in 8 directions."""
        x = int(unit.pos.x)
        y = int(unit.pos.y)

        ma = self._move_amount

        points = [
            (x, y + 2 * ma),
            (x, y - 2 * ma),
            (x + 2 * ma, y),
            (x - 2 * ma, y),
            (x + ma, y + ma),
            (x - ma, y - ma),
            (x + ma, y - ma),
            (x - ma, y + ma),
        ]

        if include_self:
            points.append((x, y))

        return points

    def check_bounds(self, x, y):
        """Whether a point is within the map bounds."""
        return 0 <= x < self.map_x and 0 <= y < self.map_y

    def get_surrounding_pathing(self, unit):
        """Returns pathing values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=False)
        vals = [
            self.pathing_grid[x, y] if self.check_bounds(x, y) else 1 for x, y in points
        ]
        return vals

    def get_surrounding_height(self, unit):
        """Returns height values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=True)
        vals = [
            self.terrain_height[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

        - agent movement features (where it can move to, height information and pathing grid)
        - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
        - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
        - agent unit features (health, shield, unit_type)

        All of this information is flattened and concatenated into a list,
        in the aforementioned order. To know the sizes of each of the
        features inside the final list of features, take a look at the
        functions ``get_obs_move_feats_size()``,
        ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
        ``get_obs_own_feats_size()``.

        The size of the observation vector may vary, depending on the
        environment configuration and type of units present in the map.
        For instance, non-Protoss units will not have shields, movement
        features may or may not include terrain height and pathing grid,
        unit_type is not included if there is only one type of unit in the
        map etc.).

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        unit = self.get_unit_by_id(agent_id)

        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[
                    ind : ind + self.n_obs_pathing
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if dist < sight_range and e_unit.health > 0:  # visible and alive
                    # Sight range > shoot range
                    # available
                    enemy_feats[e_id, 0] = avail_actions[
                        self.n_actions_no_attack + e_id
                    ]
                    enemy_feats[e_id, 1] = dist / sight_range  # distance
                    enemy_feats[e_id, 2] = (e_x - x) / sight_range  # relative X
                    enemy_feats[e_id, 3] = (e_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = (
                            e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[e_id, ind] = (
                                e_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[e_id, ind + type_id] = 1  # unit type

            # Ally features
            al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
            for i, al_id in enumerate(al_ids):
                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                if dist < sight_range and al_unit.health > 0:  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                            al_unit.health / al_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = al_unit.shield / max_shield  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            own_feats[0] = 1  # visible
            own_feats[1] = 0  # distance
            own_feats[2] = 0  # X
            own_feats[3] = 0  # Y
            ind = 4
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1
                ind += self.unit_type_bits

            if self.obs_last_action:
                own_feats[ind:] = self.last_action[agent_id]

        agent_obs = np.concatenate(
            (
                ally_feats.flatten(),
                enemy_feats.flatten(),
                move_feats.flatten(),
                own_feats.flatten(),
            )
        )

        # Agent id features
        if self.obs_agent_id:
            agent_id_feats[agent_id] = 1.0
            agent_obs = np.concatenate(
                (
                    ally_feats.flatten(),
                    enemy_feats.flatten(),
                    move_feats.flatten(),
                    own_feats.flatten(),
                    agent_id_feats.flatten(),
                )
            )

        if self.obs_timestep_number:
            agent_obs = np.append(agent_obs, self._episode_steps / self.episode_limit)

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug(
                "Avail. actions {}".format(self.get_avail_agent_actions(agent_id))
            )
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))

        # NOTE: the SND oracle/PACT obs block is appended in _snd_augment() on the
        # RETURNED obs (see step()/reset()), so it works for every state_type.
        return agent_obs

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self, agent_id=-1):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
            return obs_concat

        nf_al = 2 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 1 + self.shield_bits_enemy + self.unit_type_bits

        if self.add_center_xy:
            nf_al += 2
            nf_en += 2

        if self.add_distance_state:
            nf_al += 1
            nf_en += 1

        if self.add_xy_state:
            nf_al += 2
            nf_en += 2

        if self.add_visible_state:
            nf_al += 1
            nf_en += 1

        if self.state_last_action:
            nf_al += self.n_actions
            nf_en += self.n_actions

        if self.add_enemy_action_state:
            nf_en += 1

        nf_mv = self.get_state_move_feats_size()

        ally_state = np.zeros((self.n_agents, nf_al), dtype=np.float32)
        enemy_state = np.zeros((self.n_enemies, nf_en), dtype=np.float32)
        move_state = np.zeros((1, nf_mv), dtype=np.float32)
        agent_id_feats = np.zeros((self.n_agents, 1), dtype=np.float32)

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        unit = self.get_unit_by_id(agent_id)  # get the unit of some agent
        x = unit.pos.x
        y = unit.pos.y
        sight_range = self.unit_sight_range(agent_id)
        avail_actions = self.get_avail_agent_actions(agent_id)

        if (self.use_mustalive and unit.health > 0) or (
            not self.use_mustalive
        ):  # or else all zeros
            # Movement features
            for m in range(self.n_actions_move):
                move_state[0, m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.state_pathing_grid:
                move_state[
                    0, ind : ind + self.n_obs_pathing
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.state_terrain_height:
                move_state[0, ind:] = self.get_surrounding_height(unit)

            for al_id, al_unit in self.agents.items():
                if al_unit.health > 0:
                    al_x = al_unit.pos.x
                    al_y = al_unit.pos.y
                    max_cd = self.unit_max_cooldown(al_unit)
                    dist = self.distance(x, y, al_x, al_y)

                    ally_state[al_id, 0] = al_unit.health / al_unit.health_max  # health
                    if self.map_type == "MMM" and al_unit.unit_type == self.medivac_id:
                        ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                    else:
                        ally_state[al_id, 1] = (
                            al_unit.weapon_cooldown / max_cd
                        )  # cooldown

                    ind = 2

                    if self.add_center_xy:
                        ally_state[al_id, ind] = (
                            al_x - center_x
                        ) / self.max_distance_x  # center X
                        # center Y
                        ally_state[al_id, ind + 1] = (
                            al_y - center_y
                        ) / self.max_distance_y
                        ind += 2

                    if self.shield_bits_ally > 0:
                        max_shield = self.unit_max_shield(al_unit)
                        ally_state[al_id, ind] = al_unit.shield / max_shield  # shield
                        ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_state[al_id, ind + type_id] = 1

                    if unit.health > 0:
                        ind += self.unit_type_bits
                        if self.add_distance_state:
                            ally_state[al_id, ind] = dist / sight_range  # distance
                            ind += 1
                        if self.add_xy_state:
                            ally_state[al_id, ind] = (
                                al_x - x
                            ) / sight_range  # relative X
                            # relative Y
                            ally_state[al_id, ind + 1] = (al_y - y) / sight_range
                            ind += 2
                        if self.add_visible_state:
                            if dist < sight_range:
                                ally_state[al_id, ind] = 1  # visible
                            ind += 1
                        if self.state_last_action:
                            ally_state[al_id, ind:] = self.last_action[al_id]

            for e_id, e_unit in self.enemies.items():
                if e_unit.health > 0:
                    e_x = e_unit.pos.x
                    e_y = e_unit.pos.y
                    dist = self.distance(x, y, e_x, e_y)

                    enemy_state[e_id, 0] = e_unit.health / e_unit.health_max  # health

                    ind = 1
                    if self.add_center_xy:
                        enemy_state[e_id, ind] = (
                            e_x - center_x
                        ) / self.max_distance_x  # center X
                        # center Y
                        enemy_state[e_id, ind + 1] = (
                            e_y - center_y
                        ) / self.max_distance_y
                        ind += 2

                    if self.shield_bits_enemy > 0:
                        max_shield = self.unit_max_shield(e_unit)
                        enemy_state[e_id, ind] = e_unit.shield / max_shield  # shield
                        ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_state[e_id, ind + type_id] = 1

                    if unit.health > 0:
                        ind += self.unit_type_bits
                        if self.add_distance_state:
                            enemy_state[e_id, ind] = dist / sight_range  # distance
                            ind += 1
                        if self.add_xy_state:
                            enemy_state[e_id, ind] = (
                                e_x - x
                            ) / sight_range  # relative X
                            # relative Y
                            enemy_state[e_id, ind + 1] = (e_y - y) / sight_range
                            ind += 2
                        if self.add_visible_state:
                            if dist < sight_range:
                                enemy_state[e_id, ind] = 1  # visible
                            ind += 1
                        if self.add_enemy_action_state:
                            # available
                            enemy_state[e_id, ind] = avail_actions[
                                self.n_actions_no_attack + e_id
                            ]

        state = np.append(ally_state.flatten(), enemy_state.flatten())

        if self.add_move_state:
            state = np.append(state, move_state.flatten())

        if self.add_local_obs:
            state = np.append(state, self.get_obs_agent(agent_id).flatten())

        if self.state_timestep_number:
            state = np.append(state, self._episode_steps / self.episode_limit)

        if self.add_agent_id:
            agent_id_feats[agent_id] = 1.0
            state = np.append(state, agent_id_feats.flatten())

        state = state.astype(dtype=np.float32)

        if self.debug:
            logging.debug("STATE".center(60, "-"))
            logging.debug("Ally state {}".format(ally_state))
            logging.debug("Enemy state {}".format(enemy_state))
            logging.debug("Move state {}".format(move_state))
            if self.state_last_action:
                logging.debug("Last actions {}".format(self.last_action))

        return state

    def get_global_state(self):
        """Returns the agent-agnostic global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
            return obs_concat

        nf_al = 2 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 1 + self.shield_bits_enemy + self.unit_type_bits

        if self.add_center_xy:
            nf_al += 2
            nf_en += 2

        if self.state_last_action:
            nf_al += self.n_actions

        nf_mv_glb = self.get_state_move_feats_size_global()

        ally_state = np.zeros((self.n_agents, nf_al), dtype=np.float32)
        enemy_state = np.zeros((self.n_enemies, nf_en), dtype=np.float32)
        move_state = np.zeros((self.n_agents, nf_mv_glb), dtype=np.float32)
        info_state = np.zeros((1, 5), dtype=np.float32)

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        # move_state
        for agent_id in range(self.n_agents):
            unit = self.get_unit_by_id(agent_id)
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions):
                move_state[agent_id, m] = avail_actions[m]
            ind = self.n_actions
            if self.state_pathing_grid:
                move_state[
                    agent_id, ind : ind + self.n_obs_pathing
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing
            if self.state_terrain_height:
                move_state[agent_id, ind:] = self.get_surrounding_height(unit)

        # ally_state
        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_unit)

                ally_state[al_id, 0] = al_unit.health / al_unit.health_max  # health
                if self.map_type == "MMM" and al_unit.unit_type == self.medivac_id:
                    ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                else:
                    ally_state[al_id, 1] = al_unit.weapon_cooldown / max_cd  # cooldown

                ind = 2

                if self.add_center_xy:
                    ally_state[al_id, ind] = (
                        al_x - center_x
                    ) / self.max_distance_x  # center X
                    # center Y
                    ally_state[al_id, ind + 1] = (al_y - center_y) / self.max_distance_y
                    ind += 2

                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    ally_state[al_id, ind] = al_unit.shield / max_shield  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(al_unit, True)
                    ally_state[al_id, ind + type_id] = 1
                    ind += self.unit_type_bits

                if self.state_last_action:
                    ally_state[al_id, ind:] = self.last_action[al_id]

        # enemy_state
        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y

                enemy_state[e_id, 0] = e_unit.health / e_unit.health_max  # health

                ind = 1
                if self.add_center_xy:
                    enemy_state[e_id, ind] = (
                        e_x - center_x
                    ) / self.max_distance_x  # center X
                    # center Y
                    enemy_state[e_id, ind + 1] = (e_y - center_y) / self.max_distance_y
                    ind += 2

                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    enemy_state[e_id, ind] = e_unit.shield / max_shield  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_state[e_id, ind + type_id] = 1
                    ind += self.unit_type_bits

        # info_state
        info_state[0, 0] = self.map_x
        info_state[0, 1] = self.map_y
        info_state[0, 2] = self.max_distance_x
        info_state[0, 3] = self.max_distance_y
        info_state[0, 4] = self.unit_sight_range(0)

        state = np.append(ally_state.flatten(), enemy_state.flatten())

        if self.add_move_state:
            state = np.append(state, move_state.flatten())

        if self.state_timestep_number:
            state = np.append(state, self._episode_steps / self.episode_limit)

        if self.global_state_include_info:
            state = np.append(state, info_state.flatten())

        state = state.astype(dtype=np.float32)

        if self.debug:
            logging.debug("STATE".center(60, "-"))
            logging.debug("Ally state {}".format(ally_state))
            logging.debug("Enemy state {}".format(enemy_state))
            logging.debug("Move state {}".format(move_state))
            logging.debug("Info state {}".format(info_state))
            if self.state_last_action:
                logging.debug("Last actions {}".format(self.last_action))

        return state

    def get_state_agent(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

        - agent movement features (where it can move to, height information and pathing grid)
        - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
        - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
        - agent unit features (health, shield, unit_type)

        All of this information is flattened and concatenated into a list,
        in the aforementioned order. To know the sizes of each of the
        features inside the final list of features, take a look at the
        functions ``get_obs_move_feats_size()``,
        ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
        ``get_obs_own_feats_size()``.

        The size of the observation vector may vary, depending on the
        environment configuration and type of units present in the map.
        For instance, non-Protoss units will not have shields, movement
        features may or may not include terrain height and pathing grid,
        unit_type is not included if there is only one type of unit in the
        map etc.).

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
            return obs_concat

        unit = self.get_unit_by_id(agent_id)

        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_state_enemy_feats_size()
        ally_feats_dim = self.get_state_ally_feats_size()
        own_feats_dim = self.get_state_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        # otherwise dead, return all zeros
        if (self.use_mustalive and unit.health > 0) or (not self.use_mustalive):
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.state_pathing_grid:
                move_feats[
                    ind : ind + self.n_obs_pathing
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.state_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if e_unit.health > 0:  # visible and alive
                    # Sight range > shoot range
                    if unit.health > 0:
                        # available
                        enemy_feats[e_id, 0] = avail_actions[
                            self.n_actions_no_attack + e_id
                        ]
                        enemy_feats[e_id, 1] = dist / sight_range  # distance
                        enemy_feats[e_id, 2] = (e_x - x) / sight_range  # relative X
                        enemy_feats[e_id, 3] = (e_y - y) / sight_range  # relative Y
                        if dist < sight_range:
                            enemy_feats[e_id, 4] = 1  # visible

                    ind = 5
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = (
                            e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[e_id, ind] = (
                                e_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[e_id, ind + type_id] = 1  # unit type
                        ind += self.unit_type_bits

                    if self.add_center_xy:
                        enemy_feats[e_id, ind] = (
                            e_x - center_x
                        ) / self.max_distance_x  # center X
                        # center Y
                        enemy_feats[e_id, ind + 1] = (
                            e_y - center_y
                        ) / self.max_distance_y

            # Ally features
            al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
            for i, al_id in enumerate(al_ids):
                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)
                max_cd = self.unit_max_cooldown(al_unit)

                if al_unit.health > 0:  # visible and alive
                    if unit.health > 0:
                        if dist < sight_range:
                            ally_feats[i, 0] = 1  # visible
                        ally_feats[i, 1] = dist / sight_range  # distance
                        ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                        ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    if self.map_type == "MMM" and al_unit.unit_type == self.medivac_id:
                        ally_feats[i, 4] = al_unit.energy / max_cd  # energy
                    else:
                        ally_feats[i, 4] = al_unit.weapon_cooldown / max_cd  # cooldown

                    ind = 5
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                            al_unit.health / al_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = al_unit.shield / max_shield  # shield
                            ind += 1

                    if self.add_center_xy:
                        ally_feats[i, ind] = (
                            al_x - center_x
                        ) / self.max_distance_x  # center X
                        # center Y
                        ally_feats[i, ind + 1] = (al_y - center_y) / self.max_distance_y
                        ind += 2

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.state_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            own_feats[0] = 1  # visible
            own_feats[1] = 0  # distance
            own_feats[2] = 0  # X
            own_feats[3] = 0  # Y
            ind = 4
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.add_center_xy:
                own_feats[ind] = (x - center_x) / self.max_distance_x  # center X
                own_feats[ind + 1] = (y - center_y) / self.max_distance_y  # center Y
                ind += 2

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1
                ind += self.unit_type_bits

            if self.state_last_action:
                own_feats[ind:] = self.last_action[agent_id]

        state = np.concatenate(
            (
                ally_feats.flatten(),
                enemy_feats.flatten(),
                move_feats.flatten(),
                own_feats.flatten(),
            )
        )

        # Agent id features
        if self.state_agent_id:
            agent_id_feats[agent_id] = 1.0
            state = np.append(state, agent_id_feats.flatten())

        if self.state_timestep_number:
            state = np.append(state, self._episode_steps / self.episode_limit)

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug(
                "Avail. actions {}".format(self.get_avail_agent_actions(agent_id))
            )
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))

        # NOTE: the SND oracle/PACT state block is appended in _snd_augment() on the
        # RETURNED state (see step()/reset()), so it works for every state_type.
        return state

    def get_obs_enemy_feats_size(self):
        """Returns the dimensions of the matrix containing enemy features.
        Size is n_enemies x n_features.
        """
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        return self.n_enemies, nf_en

    def get_state_enemy_feats_size(self):
        """Returns the dimensions of the matrix containing enemy features.
        Size is n_enemies x n_features.
        """
        nf_en = 5 + self.unit_type_bits

        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        if self.add_center_xy:
            nf_en += 2

        return self.n_enemies, nf_en

    def get_obs_ally_feats_size(self):
        """Returns the dimensions of the matrix containing ally features.
        Size is n_allies x n_features.
        """
        nf_al = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        if self.obs_last_action:
            nf_al += self.n_actions

        return self.n_agents - 1, nf_al

    def get_state_ally_feats_size(self):
        """Returns the dimensions of the matrix containing ally features.
        Size is n_allies x n_features.
        """
        nf_al = 5 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        if self.obs_last_action:
            nf_al += self.n_actions

        if self.add_center_xy:
            nf_al += 2

        return self.n_agents - 1, nf_al

    def get_obs_own_feats_size(self):
        """Returns the size of the vector containing the agents' own features."""
        own_feats = 4 + self.unit_type_bits
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally

        if self.obs_last_action:
            own_feats += self.n_actions

        return own_feats

    def get_state_own_feats_size(self):
        """Returns the size of the vector containing the agents' own features."""
        own_feats = 4 + self.unit_type_bits
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally

        if self.obs_last_action:
            own_feats += self.n_actions

        if self.add_center_xy:
            own_feats += 2

        return own_feats

    def get_obs_move_feats_size(self):
        """Returns the size of the vector containing the agents's movement-related features."""
        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height

        return move_feats

    def get_state_move_feats_size(self):
        """Returns the size of the vector containing the agents's movement-related features."""
        move_feats = self.n_actions_move
        if self.state_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.state_terrain_height:
            move_feats += self.n_obs_height

        return move_feats

    def get_state_move_feats_size_global(self):
        """Returns the size of the vector containing the agents's movement-related features. global"""
        move_feats = self.n_actions
        if self.state_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.state_terrain_height:
            move_feats += self.n_obs_height

        return move_feats

    def get_obs_size(self):
        """Returns the size of the observation."""
        own_feats = self.get_obs_own_feats_size()
        move_feats = self.get_obs_move_feats_size()

        n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()
        n_allies, n_ally_feats = self.get_obs_ally_feats_size()

        enemy_feats = n_enemies * n_enemy_feats
        ally_feats = n_allies * n_ally_feats

        all_feats = move_feats + enemy_feats + ally_feats + own_feats

        agent_id_feats = 0
        timestep_feats = 0

        if self.obs_agent_id:
            agent_id_feats = self.n_agents
            all_feats += agent_id_feats

        if self.obs_timestep_number:
            timestep_feats = 1
            all_feats += timestep_feats

        # (SND oracle/PACT append is added to observation_space in _snd_grow_spaces)
        return [
            all_feats * self.stacked_frames if self.use_stacked_frames else all_feats,
            [n_allies, n_ally_feats],
            [n_enemies, n_enemy_feats],
            [1, move_feats],
            [1, own_feats + agent_id_feats + timestep_feats],
        ]

    def get_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return [
                self.get_obs_size()[0] * self.n_agents,
                [self.n_agents, self.get_obs_size()[0]],
            ]

        if self.use_state_agent:
            own_feats = self.get_state_own_feats_size()
            move_feats = self.get_obs_move_feats_size()

            n_enemies, n_enemy_feats = self.get_state_enemy_feats_size()
            n_allies, n_ally_feats = self.get_state_ally_feats_size()

            enemy_feats = n_enemies * n_enemy_feats
            ally_feats = n_allies * n_ally_feats

            all_feats = move_feats + enemy_feats + ally_feats + own_feats

            agent_id_feats = 0
            timestep_feats = 0

            if self.state_agent_id:
                agent_id_feats = self.n_agents
                all_feats += agent_id_feats

            if self.state_timestep_number:
                timestep_feats = 1
                all_feats += timestep_feats

            # (SND oracle/PACT append is added to share_observation_space in
            # _snd_grow_spaces)
            return [
                all_feats * self.stacked_frames
                if self.use_stacked_frames
                else all_feats,
                [n_allies, n_ally_feats],
                [n_enemies, n_enemy_feats],
                [1, move_feats],
                [1, own_feats + agent_id_feats + timestep_feats],
            ]

        if self.use_global_state:
            nf_al = 2 + self.shield_bits_ally + self.unit_type_bits
            nf_en = 1 + self.shield_bits_enemy + self.unit_type_bits

            if self.add_center_xy:
                nf_al += 2
                nf_en += 2

            if self.state_last_action:
                nf_al += self.n_actions

            nf_mv_glb = self.get_state_move_feats_size_global()

            enemy_state = self.n_enemies * nf_en
            ally_state = self.n_agents * nf_al

            size = enemy_state + ally_state

            move_state = 0
            timestep_state = 0
            info_state = 0

            if self.add_move_state:
                move_state = self.n_agents * nf_mv_glb
                size += move_state

            if self.state_timestep_number:
                timestep_state = 1
                size += timestep_state

            if self.global_state_include_info:
                info_state = 5
                size += info_state

            return [
                size * self.stacked_frames if self.use_stacked_frames else size,
                [self.n_agents, nf_al],
                [self.n_enemies, nf_en],
                [self.n_agents, nf_mv_glb if self.add_move_state else 0],
                [1, timestep_state],
                [1, info_state],
            ]

        nf_al = 2 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 1 + self.shield_bits_enemy + self.unit_type_bits
        nf_mv = self.get_state_move_feats_size()

        if self.add_center_xy:
            nf_al += 2
            nf_en += 2

        if self.state_last_action:
            nf_al += self.n_actions
            nf_en += self.n_actions

        if self.add_visible_state:
            nf_al += 1
            nf_en += 1

        if self.add_distance_state:
            nf_al += 1
            nf_en += 1

        if self.add_xy_state:
            nf_al += 2
            nf_en += 2

        if self.add_enemy_action_state:
            nf_en += 1

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        move_state = 0
        obs_agent_size = 0
        timestep_state = 0
        agent_id_feats = 0

        if self.add_move_state:
            move_state = nf_mv
            size += move_state

        if self.add_local_obs:
            obs_agent_size = self.get_obs_size()[0]
            size += obs_agent_size

        if self.state_timestep_number:
            timestep_state = 1
            size += timestep_state

        if self.add_agent_id:
            agent_id_feats = self.n_agents
            size += agent_id_feats

        return [
            size * self.stacked_frames if self.use_stacked_frames else size,
            [self.n_agents, nf_al],
            [self.n_enemies, nf_en],
            [1, move_state + obs_agent_size + timestep_state + agent_id_feats],
        ]

    def get_visibility_matrix(self):
        """Returns a boolean numpy array of dimensions
        (n_agents, n_agents + n_enemies) indicating which units
        are visible to each agent.
        """
        arr = np.zeros((self.n_agents, self.n_agents + self.n_enemies), dtype=np.bool)

        for agent_id in range(self.n_agents):
            current_agent = self.get_unit_by_id(agent_id)
            if current_agent.health > 0:  # it agent not dead
                x = current_agent.pos.x
                y = current_agent.pos.y
                sight_range = self.unit_sight_range(agent_id)

                # Enemies
                for e_id, e_unit in self.enemies.items():
                    e_x = e_unit.pos.x
                    e_y = e_unit.pos.y
                    dist = self.distance(x, y, e_x, e_y)

                    if dist < sight_range and e_unit.health > 0:
                        # visible and alive
                        arr[agent_id, self.n_agents + e_id] = 1

                # The matrix for allies is filled symmetrically
                al_ids = [al_id for al_id in range(self.n_agents) if al_id > agent_id]
                for i, al_id in enumerate(al_ids):
                    al_unit = self.get_unit_by_id(al_id)
                    al_x = al_unit.pos.x
                    al_y = al_unit.pos.y
                    dist = self.distance(x, y, al_x, al_y)

                    if dist < sight_range and al_unit.health > 0:
                        # visible and alive
                        arr[agent_id, al_id] = arr[al_id, agent_id] = 1

        return arr

    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            type_id = unit.unit_type - self._min_unit_type
        else:  # use default SC2 unit types
            if self.map_type == "stalkers_and_zealots":
                # id(Stalker) = 74, id(Zealot) = 73
                type_id = unit.unit_type - 73
            elif self.map_type == "colossi_stalkers_zealots":
                # id(Stalker) = 74, id(Zealot) = 73, id(Colossus) = 4
                if unit.unit_type == 4:
                    type_id = 0
                elif unit.unit_type == 74:
                    type_id = 1
                else:
                    type_id = 2
            elif self.map_type == "bane":
                if unit.unit_type == 9:
                    type_id = 0
                else:
                    type_id = 1
            elif self.map_type == "MMM":
                if unit.unit_type == 51:
                    type_id = 0
                elif unit.unit_type == 48:
                    type_id = 1
                else:
                    type_id = 2

        return type_id

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            target_items = self.enemies.items()
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                # Medivacs cannot heal themselves or other flying units
                target_items = [
                    (t_id, t_unit)
                    for (t_id, t_unit) in self.agents.items()
                    if t_unit.unit_type != self.medivac_id
                ]

            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    dist = self.distance(
                        unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                    )
                    if dist <= shoot_range:
                        avail_actions[t_id + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def close(self):
        """Close StarCraft II."""
        if self._sc2_proc:
            self._sc2_proc.close()

    def seed(self, seed):
        """Returns the random seed used by the environment."""
        self._seed = seed
        self._cwo_rng = np.random.RandomState(seed)  # CWO weapon-jam RNG

    def render(self):
        """Use save_replay instead"""
        pass

    def _kill_all_units(self):
        """Kill all units on the map."""
        units_alive = [unit.tag for unit in self.agents.values() if unit.health > 0] + [
            unit.tag for unit in self.enemies.values() if unit.health > 0
        ]
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
        ]
        self._controller.debug(debug_command)

    def init_units(self):
        """Initialise the units."""
        while True:
            # Sometimes not all units have yet been created by SC2
            self.agents = {}
            self.enemies = {}

            ally_units = [
                unit for unit in self._obs.observation.raw_data.units if unit.owner == 1
            ]
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                reverse=False,
            )

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug:
                    logging.debug(
                        "Unit {} is {}, x = {}, y = {}".format(
                            len(self.agents),
                            self.agents[i].unit_type,
                            self.agents[i].pos.x,
                            self.agents[i].pos.y,
                        )
                    )

            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 2:
                    self.enemies[len(self.enemies)] = unit
                    if self._episode_count == 0:
                        self.max_reward += unit.health_max + unit.shield_max

            if self._episode_count == 0:
                min_unit_type = min(unit.unit_type for unit in self.agents.values())
                self._init_ally_unit_types(min_unit_type)

            all_agents_created = len(self.agents) == self.n_agents
            all_enemies_created = len(self.enemies) == self.n_enemies

            if all_agents_created and all_enemies_created:  # all good
                return

            try:
                self._controller.step(1)
                self._obs = self._controller.observe()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
                self.reset()

    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_ally_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated:  # dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # dead
                e_unit.health = 0

        if n_ally_alive == 0 and n_enemy_alive > 0 or self.only_medivac_left(ally=True):
            return -1  # lost
        if (
            n_ally_alive > 0
            and n_enemy_alive == 0
            or self.only_medivac_left(ally=False)
        ):
            return 1  # won
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None

    def _init_ally_unit_types(self, min_unit_type):
        """Initialise ally unit types. Should be called once from the
        init_units function.
        """
        self._min_unit_type = min_unit_type
        if self.map_type == "marines":
            self.marine_id = min_unit_type
        elif self.map_type == "stalkers_and_zealots":
            self.stalker_id = min_unit_type
            self.zealot_id = min_unit_type + 1
        elif self.map_type == "colossi_stalkers_zealots":
            self.colossus_id = min_unit_type
            self.stalker_id = min_unit_type + 1
            self.zealot_id = min_unit_type + 2
        elif self.map_type == "MMM":
            self.marauder_id = min_unit_type
            self.marine_id = min_unit_type + 1
            self.medivac_id = min_unit_type + 2
        elif self.map_type == "zealots":
            self.zealot_id = min_unit_type
        elif self.map_type == "hydralisks":
            self.hydralisk_id = min_unit_type
        elif self.map_type == "stalkers":
            self.stalker_id = min_unit_type
        elif self.map_type == "colossus":
            self.colossus_id = min_unit_type
        elif self.map_type == "bane":
            self.baneling_id = min_unit_type
            self.zergling_id = min_unit_type + 1

    def only_medivac_left(self, ally):
        """Check if only Medivac units are left."""
        if self.map_type != "MMM":
            return False

        if ally:
            units_alive = [
                a
                for a in self.agents.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 0:
                return True
            return False
        else:
            units_alive = [
                a
                for a in self.enemies.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 1 and units_alive[0].unit_type == 54:
                return True
            return False

    def get_unit_by_id(self, a_id):
        """Get unit by ID."""
        return self.agents[a_id]

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "restarts": self.force_restarts,
        }
        return stats
