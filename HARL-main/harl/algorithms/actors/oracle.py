"""Oracle algorithm for HARL.

An *oracle* run measures the performance ceiling for the non-stationary Ant: it
gives the policy privileged access to the hidden non-stationary context (the
exogenous day/night ``ambient`` phase, the per-motor ``heat``, and the resulting
``derate``) by appending it to the observation. There is no representation
learning and no inference -- the agent simply *sees everything*. It is plain
HAPPO conditioned on the true hidden state, and serves as the upper bound that the
context-handling baselines (COREP / LCPO / FSAC / TRIO / ESCP) are compared to.

The runner (``OnPolicyOracleRunner``) reads the context from the env ``info`` dict
and augments the observation (and the centralized critic state) at both training
and evaluation time.
"""

import numpy as np
import torch
from gym.spaces import Box

from harl.algorithms.actors.happo import HAPPO


class Oracle(HAPPO):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize Oracle (HAPPO on observations augmented with the true context).

        Args:
            args: (dict) merged ``{**model, **algo}`` arguments.
            obs_space: (gym.spaces.Box) *raw* observation space.
            act_space: (gym.spaces.Box) action space.
            device: (torch.device) device.
        """
        self.oracle_dim = int(args.get("oracle_dim", 3))
        assert obs_space.__class__.__name__ == "Box", "Oracle expects Box observations."
        self.raw_obs_dim = int(obs_space.shape[0])

        low = np.concatenate(
            [
                np.full(self.raw_obs_dim, -np.inf, dtype=np.float32),
                np.full(self.oracle_dim, -np.inf, dtype=np.float32),
            ]
        )
        augmented_obs_space = Box(low=low, high=-low, dtype=np.float32)
        super(Oracle, self).__init__(args, augmented_obs_space, act_space, device)
