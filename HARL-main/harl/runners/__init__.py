"""Runner registry."""
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.runners.off_policy_ha_runner import OffPolicyHARunner
from harl.runners.off_policy_ma_runner import OffPolicyMARunner
from harl.runners.on_policy_corep_runner import OnPolicyCorepRunner
from harl.runners.on_policy_lcpo_runner import OnPolicyLcpoRunner
from harl.runners.off_policy_fsac_runner import OffPolicyFsacRunner
from harl.runners.on_policy_trio_runner import OnPolicyTrioRunner
from harl.runners.on_policy_escp_runner import OnPolicyEscpRunner
from harl.runners.on_policy_oracle_runner import OnPolicyOracleRunner
from harl.runners.on_policy_doraemon_runner import OnPolicyDoraemonRunner
from harl.runners.on_policy_advantage_alignment_runner import (
    OnPolicyAdvAlignRunner,
)
from harl.runners.off_policy_mbcd_runner import OffPolicyMbcdRunner
from harl.runners.on_policy_wisdom_runner import OnPolicyWisdomRunner
from harl.runners.on_policy_mamt_runner import OnPolicyMamtRunner
from harl.runners.on_policy_drive_runner import OnPolicyDriveRunner
from harl.runners.on_policy_comarl_runner import OnPolicyComarlRunner
from harl.runners.off_policy_echor_runner import OffPolicyEchoRRunner
from harl.runners.off_policy_ecl_runner import OffPolicyEclRunner
from harl.runners.off_policy_diag_runner import OffPolicyDiagRunner
from harl.runners.on_policy_recon_runner import OnPolicyReconRunner
from harl.runners.on_policy_omax_runner import OnPolicyOmaxRunner
from harl.runners.on_policy_pact_runner import OnPolicyPactRunner
from harl.runners.on_policy_pact_smac_runner import OnPolicyPactSmacRunner


def _pact_runner(args, algo_args, env_args):
    """`--algo pact` dispatches by env: the MAMuJoCo runner (continuous, learned
    beta, reward decomposition) for mamujoco, the SMAC runner (discrete soft
    variant, cosine-gate logging only) for smac/smacv2.  Both subclass the HAPPO
    runner and train bit-identically; only the diagnostics differ."""
    if args["env"] in ("smac", "smacv2"):
        return OnPolicyPactSmacRunner(args, algo_args, env_args)
    return OnPolicyPactRunner(args, algo_args, env_args)


RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,
    "corep": OnPolicyCorepRunner,
    "lcpo": OnPolicyLcpoRunner,
    "fsac": OffPolicyFsacRunner,
    "trio": OnPolicyTrioRunner,
    "escp": OnPolicyEscpRunner,
    "oracle": OnPolicyOracleRunner,
    # ERNIE only modifies the actor update (adversarial regularizer computed from
    # the minibatch obs); no extra buffer/state is needed, so it reuses the
    # standard on-policy HA runner.
    "ernie": OnPolicyHARunner,
    "doraemon": OnPolicyDoraemonRunner,
    "advantage_alignment": OnPolicyAdvAlignRunner,
    "mbcd": OffPolicyMbcdRunner,
    "wisdom": OnPolicyWisdomRunner,
    "mamt": OnPolicyMamtRunner,
    "drive": OnPolicyDriveRunner,
    "comarl": OnPolicyComarlRunner,
    # ECHO-R: on-policy (HAPPO) backbone reuses the standard HA runner unchanged
    # (all ECHO-R logic is in the env wrapper); off-policy (HASAC) backbone uses
    # a thin subclass that selects the HASAC code paths.
    "echor": OnPolicyHARunner,
    "echor_hasac": OffPolicyEchoRRunner,
    # ECL: full method on HASAC (custom off-policy runner); envelope-only degraded
    # variant on HAPPO reuses the standard on-policy HA runner.
    "ecl": OffPolicyEclRunner,
    "ecl_happo": OnPolicyHARunner,
    # RECON: identify centrally, filter locally, act certainty-equivalently.
    # Reference host is HAPPO — hindsight relabeling is trivial on contiguous
    # on-policy rollouts and the filter always trains on-distribution.
    "recon": OnPolicyReconRunner,
    # O-MAX ceiling ladder: training is bit-identical to HAPPO; the runner only
    # adds omax_debug.csv (residual/timing/reward-decomposition diagnostics).
    "omax": OnPolicyOmaxRunner,
    # PACT: reach the O-MAX (O1) ceiling WITHOUT privileged info — compensate with
    # the peer-action-computed exact coupling waveform x2 and one learned gain beta.
    # Training is bit-identical to HAPPO (the mechanism is entirely env-side, with
    # beta learned as one extra action dim); the runner only adds pact_debug.csv +
    # the arithmetic exactness gate.  Dispatched by env: mamujoco (continuous, learned
    # beta) vs smac/smacv2 (discrete soft variant, obs-augmentation, gate-only).
    "pact": _pact_runner,
    # PACT-1: same dispatch. The wrapper emits the same pact_* info keys the runner
    # already reads, so pact_debug.csv works unchanged -- except the gate now
    # compares the PREDICTED load d_hat against the true pcr_d_next, which is a
    # stricter and more directly meaningful check than the old x2 waveform cosine.
    "pact_1": _pact_runner,
    # SMAC PACT-1: same dispatch -> OnPolicyPactSmacRunner (env is smac), which now
    # also logs the p1_* estimator columns.
    "smac_pact_1": _pact_runner,
    # PCR diagnosis campaign: HASAC + read-only, RNG-transparent telemetry.
    # Not a method — with telemetry off it IS hasac, and with it on the training
    # trajectory is still bit-identical (see OffPolicyDiagRunner._rng_frozen).
    "hasac_diag": OffPolicyDiagRunner,
    "hatrpo": OnPolicyHARunner,
    "haa2c": OnPolicyHARunner,
    "haddpg": OffPolicyHARunner,
    "hatd3": OffPolicyHARunner,
    "hasac": OffPolicyHARunner,
    "had3qn": OffPolicyHARunner,
    "maddpg": OffPolicyMARunner,
    "matd3": OffPolicyMARunner,
    "mappo": OnPolicyMARunner,
}
