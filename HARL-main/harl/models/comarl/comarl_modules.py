"""Building blocks for COMARL (Distributionally Robust Cooperative MARL).

The single new network is the ``GNetwork`` -- the learned dual variable of the
TV (total-variation) distributionally-robust Bellman operator from the paper
"Distributionally Robust Cooperative Multi-agent RL with Value Factorization".

In the reference (``networks.py``/``vdn_g.py``/``qmix_g.py``) ``g`` is a function
of ``(state, joint_action)``; the robust next-state value is

    sigma_rho(V') = (1 - rho) * g - relu(g - V')

and ``g`` is trained to satisfy ``relu(g - V') = (1 - rho) * g`` (the G-loss).
The whole construction is mixer-agnostic -- it is byte-for-byte identical in the
VDN and QMIX variants -- so it transfers directly onto HAPPO's single
centralized V-critic, with ``V'`` the critic's next-state value and ``g`` a
function of the centralized state ``share_obs`` (there is no single greedy joint
action in the on-policy actor-critic setting).
"""

import numpy as np
import torch
import torch.nn as nn

from harl.models.base.mlp import MLPBase
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_init_method, init


class GNetwork(nn.Module):
    """Dual-variable network ``g(share_obs) -> scalar`` for the TV-robust operator."""

    def __init__(self, args, state_dim, device=torch.device("cpu")):
        super(GNetwork, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        state_shape = (state_dim,)
        self.base = MLPBase(args, state_shape)
        init_method = get_init_method(args["initialization_method"])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.g_out = init_(nn.Linear(args["hidden_sizes"][-1], 1))
        self.to(device)

    def forward(self, state):
        state = check(state).to(**self.tpdv)
        return self.g_out(self.base(state))

    @torch.no_grad()
    def values_np(self, state):
        """Forward a (possibly batched) numpy ``state`` -> numpy ``g`` of shape (..., 1)."""
        return self.forward(state).cpu().numpy()


def robust_next_value(v_next, rho, mode="contamination", g=None):
    """Distributionally-robust transform of a next-state value (numpy).

    Implements the exact COMARL robust operator (mixer-agnostic):

      * ``contamination``: ``sigma = (1 - rho) * V'``        (base VDN/QMIX/QTRAN)
      * ``tv``           : ``sigma = (1 - rho) * g - (g - V')_+``  (the G-network variants)

    ``rho = 0`` returns ``V'`` unchanged (recovers standard, non-robust returns).

    Args:
        v_next: (np.ndarray) next-state value V'(s') in *true* (denormalized) scale.
        rho: (float) robustness radius in [0, 1).
        mode: (str) "contamination" or "tv".
        g: (np.ndarray or None) G-network output g(s), same shape as v_next (tv only).
    """
    if mode == "tv" and g is not None:
        return (1.0 - rho) * g - np.maximum(g - v_next, 0.0)
    return (1.0 - rho) * v_next
