"""FSAC (Forecaster-SAC) actor.

The FSAC actor is identical to the HASAC actor (a squashed-Gaussian / stochastic
MLP policy trained with the SAC objective). All of FSAC's novelty lives in the
critic (the forecasting head, ``FSACCritic``) and in how the off-policy runner
uses the *forecasted* Q for policy improvement. This subclass exists only so that
``fsac`` is a first-class entry in the algorithm registry.
"""

import torch

from harl.algorithms.actors.hasac import HASAC


class FSAC(HASAC):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(FSAC, self).__init__(args, obs_space, act_space, device)
