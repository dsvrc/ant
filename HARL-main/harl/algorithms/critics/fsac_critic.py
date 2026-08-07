"""Forecaster Soft Twin Continuous Q Critic (FSAC).

Implements the "Forecaster" of Forecaster-SAC (https://arxiv.org/pdf/2405.16053)
on top of HARL's soft twin Q critic (the critic used by HASAC).

Core idea: in a non-stationary environment the optimal Q-function drifts over
time. FSAC keeps a rolling *history* of past critic snapshots, and for the policy
improvement step it does **not** use the current Q. Instead it evaluates every
snapshot at (s, pi(s)), obtaining a short time-series of Q-values, fits a
least-squares line through them, and **extrapolates that line to a future time
index** ``past_length + future_length``. The policy is then improved against this
*forecasted future Q*, so it anticipates where the optimum is heading rather than
chasing the (already stale) present Q.

The least-squares forecasting matrix is exactly the one from the reference code
(``utils.compute_weight_X``): with design matrix ``X = [[1,1],[2,1],...,[L,1]]``
(time index + intercept), ``W = (X^T X)^{-1} X^T`` maps the L past Q-values to
``[slope, intercept]``; the forecast is ``[future_idx, 1] @ [slope; intercept]``.

Difference from the released reference: the reference appends the *live* critic by
reference (so all "snapshots" alias one network and the linear fit is flat). Here
we store genuinely frozen deep copies, so the forecast reflects the real temporal
trend of the Q-function -- which is the method's actual intent.

Only the *policy improvement* uses the forecast; the critic itself is trained
exactly like HASAC's soft twin critic (inherited unchanged).
"""

from collections import deque
from copy import deepcopy

import numpy as np
import torch

from harl.algorithms.critics.soft_twin_continuous_q_critic import (
    SoftTwinContinuousQCritic,
)
from harl.utils.envs_tools import check


def compute_weight_X(past_length):
    """Least-squares weight matrix W = (X^T X)^{-1} X^T, shape (2, past_length).

    X has rows [i+1, 1] for i in 0..past_length-1 (time index, intercept).
    W @ y -> [slope, intercept] for a vector y of ``past_length`` Q-values.
    """
    X = np.array([[i + 1, 1] for i in range(past_length)], dtype=np.float64)
    XtX = np.matmul(np.transpose(X), X)
    XtX_inv = np.linalg.inv(XtX)
    W = np.matmul(XtX_inv, np.transpose(X))
    return torch.FloatTensor(W)


class FSACCritic(SoftTwinContinuousQCritic):
    """Soft twin Q critic with a forecasting head for the policy improvement step."""

    def __init__(
        self,
        args,
        share_obs_space,
        act_space,
        num_agents,
        state_type,
        device=torch.device("cpu"),
    ):
        super(FSACCritic, self).__init__(
            args, share_obs_space, act_space, num_agents, state_type, device
        )
        self.fsac_past_length = int(args.get("fsac_past_length", 10))
        self.fsac_future_length = int(args.get("fsac_future_length", 10))
        self.fsac_snapshot_interval = int(args.get("fsac_snapshot_interval", 1000))

        # rolling history of frozen (critic, critic2) snapshots, oldest -> newest
        self.snapshots = deque(maxlen=self.fsac_past_length)
        # least-squares regression matrix and the future evaluation point
        self.fw_x = compute_weight_X(self.fsac_past_length).to(device=device)
        self._future_idx = float(self.fsac_past_length + self.fsac_future_length)
        self.future_vec = torch.tensor(
            [[self._future_idx, 1.0]], dtype=torch.float32, device=device
        )

    def maybe_snapshot(self, it):
        """Capture a frozen snapshot of the current critics every snapshot_interval."""
        if self.fsac_snapshot_interval <= 0:
            return
        if it % self.fsac_snapshot_interval == 0:
            snap1 = deepcopy(self.critic)
            snap2 = deepcopy(self.critic2)
            for p in snap1.parameters():
                p.requires_grad = False
            for p in snap2.parameters():
                p.requires_grad = False
            snap1.eval()
            snap2.eval()
            self.snapshots.append((snap1, snap2))

    def forecaster_ready(self):
        """Whether enough snapshots have accumulated to forecast."""
        return len(self.snapshots) >= self.fsac_past_length

    def get_forecasted_values(self, share_obs, actions):
        """Forecasted future Q for policy improvement (twin min).

        Falls back to the current soft-Q (``get_values``) until the snapshot
        history is full. Gradients flow through ``actions`` (the policy output);
        the snapshot parameters are frozen.
        """
        if not self.forecaster_ready():
            return self.get_values(share_obs, actions)

        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)

        q1_cols = []
        q2_cols = []
        for snap1, snap2 in self.snapshots:  # oldest -> newest (time index 1..L)
            q1_cols.append(snap1(share_obs, actions))  # (batch, 1)
            q2_cols.append(snap2(share_obs, actions))  # (batch, 1)
        q1_concat = torch.cat(q1_cols, dim=1)  # (batch, past_length)
        q2_concat = torch.cat(q2_cols, dim=1)  # (batch, past_length)

        # [slope; intercept] per batch sample -> (2, batch)
        w1 = torch.matmul(self.fw_x, torch.transpose(q1_concat, 0, 1))
        w2 = torch.matmul(self.fw_x, torch.transpose(q2_concat, 0, 1))

        # extrapolate the fitted line to the future time index -> (batch, 1)
        q1_fore = torch.transpose(torch.matmul(self.future_vec, w1), 0, 1)
        q2_fore = torch.transpose(torch.matmul(self.future_vec, w2), 0, 1)

        return torch.min(q1_fore, q2_fore)
