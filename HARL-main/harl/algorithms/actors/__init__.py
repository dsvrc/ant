"""Algorithm registry."""
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.hatrpo import HATRPO
from harl.algorithms.actors.haa2c import HAA2C
from harl.algorithms.actors.haddpg import HADDPG
from harl.algorithms.actors.hatd3 import HATD3
from harl.algorithms.actors.hasac import HASAC
from harl.algorithms.actors.had3qn import HAD3QN
from harl.algorithms.actors.maddpg import MADDPG
from harl.algorithms.actors.matd3 import MATD3
from harl.algorithms.actors.mappo import MAPPO
from harl.algorithms.actors.corep import COREP
from harl.algorithms.actors.lcpo import LCPO
from harl.algorithms.actors.fsac import FSAC
from harl.algorithms.actors.trio import TRIO
from harl.algorithms.actors.escp import ESCP
from harl.algorithms.actors.oracle import Oracle
from harl.algorithms.actors.ernie import ERNIE
from harl.algorithms.actors.advantage_alignment import AdvantageAlignment
from harl.algorithms.actors.wisdom import WISDOM
from harl.algorithms.actors.mamt import MAMT

ALGO_REGISTRY = {
    "happo": HAPPO,
    "corep": COREP,
    "lcpo": LCPO,
    "fsac": FSAC,
    "trio": TRIO,
    "escp": ESCP,
    "oracle": Oracle,
    "ernie": ERNIE,
    # DORAEMON keeps the policy/critic identical to HAPPO; it only adds an
    # environment-side domain-randomization curriculum (see OnPolicyDoraemonRunner).
    "doraemon": HAPPO,
    "advantage_alignment": AdvantageAlignment,
    # DRIVE keeps the HAPPO policy/critic; the peer-incentive reward shaping and
    # per-agent gating value nets live in OnPolicyDriveRunner.
    "drive": HAPPO,
    # COMARL keeps the HAPPO policy/critic; the distributionally-robust Bellman
    # operator (robust GAE + optional G-network) lives in OnPolicyComarlRunner.
    "comarl": HAPPO,
    # MBCD uses the HASAC actor; the per-context detection/library is in the runner.
    "mbcd": HASAC,
    # ECHO-R keeps the host policy/critic identical to HAPPO / HASAC; the entire
    # method (probe injection + demodulator + c_hat conditioning) lives in the
    # env wrapper EchoRMujocoMulti. "echor" = HAPPO backbone, "echor_hasac" = HASAC.
    "echor": HAPPO,
    "echor_hasac": HASAC,
    # ECL keeps the host policy/critic identical; the identifier/localizer/anchor
    # live in the runner+buffer, the envelope in the env wrapper. "ecl" = HASAC
    # (full method), "ecl_happo" = HAPPO (envelope-only degraded variant, Part 5).
    "ecl": HASAC,
    "ecl_happo": HAPPO,
    # RECON keeps the host policy/critic identical to HAPPO's — the whole method
    # is one regression ([ID]), one numpy pass ([RE]), one supervised loss with
    # its own optimizer ([DI]), one obs block ([CE]) and one action-interface
    # layer ([CP]), all of which live in OnPolicyReconRunner. The obs/share-obs
    # dims are declared by ReconMujocoMulti, so HAPPO sizes itself correctly.
    "recon": HAPPO,
    # O-MAX uses the stock HAPPO actor; the ladder's advantages are all env-side
    # (hardwired compensation, privileged obs) + a flag-gated std floor.
    "omax": HAPPO,
    # PACT uses the stock HAPPO actor; the whole method is env-side (the exact
    # peer-action waveform x2, the compensation, and the extra bounded beta-control
    # action dim are all in PactMujocoMulti) plus pact_debug logging in the runner.
    # HAPPO sizes itself from the (extended) action/obs/share spaces the wrapper
    # declares, so the extra action dim is learned by standard PPO.
    "pact": HAPPO,
    # PACT-1 is PACT with the coupling operator W UNKNOWN: the wrapper carries a
    # per-agent RLS that tracks the drifting r-vector beta* = c*theta from the leg's
    # own torque sensor, and the policy learns only TRUST in that estimate (one
    # extra bounded action dim). Still stock HAPPO -- nothing about the learner
    # changes, so every arm shares host hyperparameters.
    "pact_1": HAPPO,
    # SMAC PACT-1: same idea on the discrete channel. The interference split across
    # the same-type / cross-type fire-control channels is unknown and drifting; each
    # unit tracks it from where its OWN shots actually land, and the env applies the
    # exact permutation inverse (a target pre-shift) at the estimator's confidence.
    # Still stock HAPPO -- everything is env-side.
    "smac_pact_1": HAPPO,
    # PCR diagnosis campaign: the host policy/critic are HASAC's, untouched
    # (Prohibition 2 — host hyperparameters identical across every arm). The
    # telemetry lives entirely in OffPolicyDiagRunner.
    "hasac_diag": HASAC,
    "wisdom": WISDOM,
    "mamt": MAMT,
    "hatrpo": HATRPO,
    "haa2c": HAA2C,
    "haddpg": HADDPG,
    "hatd3": HATD3,
    "hasac": HASAC,
    "had3qn": HAD3QN,
    "maddpg": MADDPG,
    "matd3": MATD3,
    "mappo": MAPPO,
}
