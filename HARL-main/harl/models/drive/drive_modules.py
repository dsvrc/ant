"""Building blocks for the DRIVE algorithm (HAPPO + dynamic reward-difference
incentive exchange).

Two pieces live here:

* ``DriveValueNet`` -- a small per-agent MLP value function ``V_i(obs_i)`` used
  *only* by DRIVE's TD-advantage gating.  It is deliberately kept separate from
  the HAPPO centralized critic: in HARL's mamujoco wrapper the centralized
  ``share_obs`` is *identical* across agents, so a centralized value would give
  the same gate for everybody (DRIVE would become a no-op).  Each agent's
  ``obs`` instead carries an agent-id one-hot, so per-agent obs nets produce
  genuinely different gates -- exactly the decentralized per-agent critics of
  the original DRIVE / MATE implementation.

* ``drive_shape_rewards`` -- a pure-numpy, fully-vectorized implementation of
  Algorithm 2 of the paper (TD-gated request/response with min-aggregated
  reward-difference exchange).  It maps the raw per-agent rewards of one
  environment step to the DRIVE-shaped rewards.
"""

import numpy as np
import torch
import torch.nn as nn

from harl.models.base.mlp import MLPBase
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_init_method, init


class DriveValueNet(nn.Module):
    """Per-agent value function ``V_i(obs_i)`` used for DRIVE's TD gating."""

    def __init__(self, args, obs_dim, device=torch.device("cpu")):
        super(DriveValueNet, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        obs_shape = (obs_dim,)
        self.base = MLPBase(args, obs_shape)
        init_method = get_init_method(args["initialization_method"])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(args["hidden_sizes"][-1], 1))
        self.to(device)

    def forward(self, obs):
        obs = check(obs).to(**self.tpdv)
        return self.v_out(self.base(obs))

    @torch.no_grad()
    def values_np(self, obs):
        """Convenience: forward a numpy ``obs`` batch -> ``(B,)`` numpy values."""
        return self.forward(obs).squeeze(-1).cpu().numpy()


def drive_shape_rewards(rewards, ubar, winners, coef=1.0, _inf=1e9):
    """Vectorized DRIVE reward shaping (paper Algorithm 2 / Eq. 4).

    All arrays are batched over the rollout threads ``B`` and agents ``N`` and
    the neighborhood is taken to be *fully connected* (every other agent), which
    matches the small, tightly-coupled set of Ant legs.

    Args:
        rewards: (B, N) raw per-agent reward ``u_t,i`` of this step.
        ubar:    (N,) or (B, N) running epoch-average reward ``\\bar u_i``.
        winners: (B, N) bool, ``True`` where ``TD_i >= 0`` (agent issues a request).
        coef:    (float) scales the net incentive relative to the raw reward.

    Returns:
        shaped: (B, N) float32 ``u^DRIVE_t,i = u_t,i - u_req_i + u_res_i``.
        u_req:  (B, N) float32 amount paid out responding to neighbors' requests.
        u_res:  (B, N) float32 amount received back from neighbors' responses.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    winners = np.asarray(winners, dtype=bool)
    B, N = rewards.shape
    ubar = np.asarray(ubar, dtype=np.float32)
    if ubar.ndim == 1:
        ubar_b = np.broadcast_to(ubar, (B, N))
    else:
        ubar_b = ubar

    eye = np.eye(N, dtype=bool)[None, :, :]  # (1, N, N): i == j (self)

    # --- responses i sends to requesting neighbors j: Delta_{j,i} = ubar_i - r_j ---
    # u_req_i = min over winning neighbors j of (ubar_i - r_j); subtracted from i.
    req_pair = ubar_b[:, :, None] - rewards[:, None, :]  # (B, N, N): [b, i, j]
    win_j = winners[:, None, :]  # (B, 1, N): is neighbor j a winner (did j request)?
    mask_req = (~win_j) | eye  # ignore non-winning neighbors and self
    u_req = np.where(mask_req, _inf, req_pair).min(axis=2)  # (B, N)
    u_req = np.where(u_req >= 0.5 * _inf, 0.0, u_req)  # no winning neighbor -> 0

    # --- responses i receives from neighbors j: Delta_{i,j} = ubar_j - r_i ---
    # u_res_i = min over neighbors j of (ubar_j - r_i); only if i issued a request.
    res_pair = ubar_b[:, None, :] - rewards[:, :, None]  # (B, N, N): [b, i, j]
    u_res = np.where(eye, _inf, res_pair).min(axis=2)  # (B, N)
    u_res = np.where(u_res >= 0.5 * _inf, 0.0, u_res)  # N == 1 safety
    u_res = np.where(winners, u_res, 0.0)  # gated by i's own request

    shaped = rewards + coef * (u_res - u_req)
    return (
        shaped.astype(np.float32),
        u_req.astype(np.float32),
        u_res.astype(np.float32),
    )
