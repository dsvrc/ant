"""Critic registry."""
from harl.algorithms.critics.v_critic import VCritic
from harl.algorithms.critics.continuous_q_critic import ContinuousQCritic
from harl.algorithms.critics.twin_continuous_q_critic import TwinContinuousQCritic
from harl.algorithms.critics.soft_twin_continuous_q_critic import (
    SoftTwinContinuousQCritic,
)
from harl.algorithms.critics.discrete_q_critic import DiscreteQCritic
from harl.algorithms.critics.fsac_critic import FSACCritic

CRITIC_REGISTRY = {
    "happo": VCritic,
    "fsac": FSACCritic,
    "hatrpo": VCritic,
    "haa2c": VCritic,
    "mappo": VCritic,
    "haddpg": ContinuousQCritic,
    "hatd3": TwinContinuousQCritic,
    "hasac": SoftTwinContinuousQCritic,
    "had3qn": DiscreteQCritic,
    "maddpg": ContinuousQCritic,
    "matd3": TwinContinuousQCritic,
    # MBCD uses the HASAC backbone (soft twin-Q centralized critic)
    "mbcd": SoftTwinContinuousQCritic,
    # ECHO-R (HASAC backbone) uses the same soft twin-Q centralized critic.
    "echor_hasac": SoftTwinContinuousQCritic,
    # ECL (HASAC backbone) uses the same soft twin-Q centralized critic.
    "ecl": SoftTwinContinuousQCritic,
    # PCR diagnosis campaign (HASAC backbone), unchanged.
    "hasac_diag": SoftTwinContinuousQCritic,
}
