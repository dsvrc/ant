"""ESCP algorithm for HARL.

ESCP ("Adapt to Environment Sudden Changes by Learning a Context Sensitive
Policy", AAAI 2022) learns an environment-sensitive context embedding with an
Environment Probe (EP) trained by the RMDM loss (variance-minimization +
relational-matrix-determinant-maximization), and conditions the policy on it.

This class adapts ESCP onto HARL's HAPPO (consistent with the other context-based
baselines; the on-policy rollout makes the history-RNN natural):

* Each agent owns an EP (history-truncated GRU) that produces a low-dim
  environment embedding from its (obs, last_action) context.
* The policy/critic are conditioned on the EP by **augmenting the observation**
  with the (detached) embedding -- ESCP's ``stop_pg_for_ep``: the policy gradient
  never trains the EP; the EP is trained *only* by RMDM.
* RMDM groups embeddings by task id. The TCC Ant has no discrete tasks, so the
  runner supplies **discretized ``ambient`` bins** as pseudo-task ids (privileged
  at training, like TRIO). EMA per-task means persist across iterations so the
  determinant-diversity term is well-defined as the phase drifts through bins.

The runner (``OnPolicyEscpRunner``) orchestrates the EP rollout, the ambient-bin
labels, and the RMDM update.
"""

import numpy as np
import torch
from gym.spaces import Box

from harl.algorithms.actors.happo import HAPPO
from harl.models.escp.escp_modules import EnvProbe, RMDMLoss
from harl.utils.discrete_util import get_encoded_act_dim, encode_actions_torch


class ESCP(HAPPO):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        # ---- ESCP hyper-parameters ----
        self.ep_dim = int(args.get("escp_ep_dim", 2))
        self.ep_hidden = int(args.get("escp_ep_hidden", 64))
        self.history_len = int(args.get("escp_history_len", 16))
        self.bottleneck = bool(args.get("escp_bottleneck", True))
        self.ep_lr = float(args.get("escp_ep_lr", 3e-4))
        self.consistency_w = float(args.get("escp_consistency_weight", 1.0))
        self.diversity_w = float(args.get("escp_diversity_weight", 1.0))
        self.repre_loss_factor = float(args.get("escp_repre_loss_factor", 1.0))
        self.rmdm_tau = float(args.get("escp_rmdm_tau", 0.995))
        self.rbf_radius = float(args.get("escp_rbf_radius", 16.0))
        self.kernel_type = args.get("escp_kernel_type", "rbf")
        self.num_rmdm_updates = int(args.get("escp_num_rmdm_updates", 1))
        self.ep_max_grad_norm = float(args.get("escp_ep_max_grad_norm", 1.0))

        assert obs_space.__class__.__name__ == "Box", "ESCP expects Box observations."
        self.raw_obs_dim = int(obs_space.shape[0])
        # Encoded action dim: shape[0] for continuous Box, n (one-hot) for Discrete
        # (SMAC/SMACv2). The environment probe consumes the (obs, last_action)
        # context; discrete last-actions are one-hot encoded by the runner via
        # ``encode_last_action`` before being handed to ``step_ep``.
        self.act_space = act_space
        self.act_dim = get_encoded_act_dim(act_space)

        # ---- actor on AUGMENTED observation (raw_obs + ep) ----
        low = np.concatenate(
            [
                np.full(self.raw_obs_dim, -np.inf, dtype=np.float32),
                np.full(self.ep_dim, -np.inf, dtype=np.float32),
            ]
        )
        augmented_obs_space = Box(low=low, high=-low, dtype=np.float32)
        super(ESCP, self).__init__(args, augmented_obs_space, act_space, device)

        # ---- environment probe + RMDM ----
        self.ep = EnvProbe(
            obs_dim=self.raw_obs_dim,
            act_dim=self.act_dim,
            ep_dim=self.ep_dim,
            hidden=self.ep_hidden,
            bottleneck=self.bottleneck,
        ).to(device)
        self.ep_optimizer = torch.optim.Adam(self.ep.parameters(), lr=self.ep_lr)
        self.rmdm = RMDMLoss(
            ep_dim=self.ep_dim,
            tau=self.rmdm_tau,
            rbf_radius=self.rbf_radius,
            kernel=self.kernel_type,
        )

    # ======================================================================
    # online environment-probe (rollout / eval)
    # ======================================================================
    def init_ep(self, n_threads):
        """Return the initial (zero) embedding and a None hidden state."""
        return np.zeros((n_threads, self.ep_dim), dtype=np.float32), None

    @torch.no_grad()
    def encode_last_action(self, action):
        """One-hot encode a raw stored action into the EP input vector.

        For continuous ``Box`` actions this is a no-op; for ``Discrete``
        (SMAC/SMACv2) the integer index ``(n, 1)`` becomes a one-hot ``(n, act_dim)``.
        Called by the runner before ``step_ep`` so the probe always receives a
        real-valued action vector of width ``self.act_dim``.
        """
        a = torch.as_tensor(np.asarray(action, dtype=np.float32), device=self.device)
        return encode_actions_torch(a, self.act_space).cpu().numpy()

    @torch.no_grad()
    def step_ep(self, obs, last_action, hidden, deterministic=False):
        """Advance the environment probe one step.

        Args:
            obs: (np.ndarray) (n, raw_obs_dim) current observation.
            last_action: (np.ndarray) (n, act_dim) action taken at the previous step.
            hidden: (torch.Tensor) GRU state or None.
            deterministic: (bool) deterministic embedding (eval).
        Returns:
            ep: (np.ndarray) (n, ep_dim) detached embedding.
            hidden: (torch.Tensor) updated GRU state.
        """
        device = self.device
        o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(1)
        a = torch.as_tensor(last_action, dtype=torch.float32, device=device).unsqueeze(1)
        ep, _, _, hidden = self.ep(o, a, hidden=hidden, deterministic=deterministic)
        return ep.squeeze(1).cpu().numpy(), hidden.detach()

    # ======================================================================
    # RMDM representation update (history-truncated windows)
    # ======================================================================
    def update_rmdm(self, obs_seq, act_seq, task_bins):
        """One RMDM update of the environment probe.

        Args:
            obs_seq: (np.ndarray) (T+1, B, raw_obs_dim) raw observations.
            act_seq: (np.ndarray) (T, B, act_dim) actions.
            task_bins: (np.ndarray) (T, B) integer task ids (0 = invalid).
        Returns:
            info: (dict) loss components (or zeros if not enough tasks).
        """
        device = self.device
        T = act_seq.shape[0]
        B = act_seq.shape[1]
        H = self.history_len
        num_win = T // H
        if num_win < 1:
            return {"rmdm_loss": 0.0, "consistency": 0.0, "diversity": 0.0}

        states = torch.as_tensor(obs_seq[:T], dtype=torch.float32, device=device)
        act = torch.as_tensor(act_seq, dtype=torch.float32, device=device)
        # one-hot encode discrete actions (no-op for continuous Box) -> (T, B, act_dim)
        act = encode_actions_torch(act, self.act_space)
        # last_action[t] = action taken at t-1 (0 at t=0)
        last_act = torch.zeros_like(act)
        last_act[1:] = act[:-1]
        labels = torch.as_tensor(task_bins, dtype=torch.long, device=device)

        ep_points = []
        label_points = []
        for w in range(num_win):
            sl = slice(w * H, (w + 1) * H)
            s_win = states[sl].transpose(0, 1)  # (B, H, raw)
            a_win = last_act[sl].transpose(0, 1)  # (B, H, act)
            ep_w, _, _, _ = self.ep(s_win, a_win, hidden=None, deterministic=False)
            ep_points.append(ep_w[:, -1, :])  # (B, ep_dim) last-step embedding
            label_points.append(labels[(w + 1) * H - 1])  # (B,)
        ep_all = torch.cat(ep_points, dim=0)  # (num_win*B, ep_dim)
        lab_all = torch.cat(label_points, dim=0)  # (num_win*B,)

        loss, consistency, diversity = self.rmdm(
            ep_all, lab_all, self.consistency_w, self.diversity_w
        )
        if loss is None:
            return {"rmdm_loss": 0.0, "consistency": 0.0, "diversity": 0.0}

        # the loss can be a grad-free constant (consistency already converged and
        # only one task seen so far) -- skip the optimizer step in that case.
        info = {
            "rmdm_loss": float(loss.item()),
            "consistency": float(consistency.item()),
            "diversity": float(diversity.item()),
        }
        if not loss.requires_grad or torch.isnan(loss).any():
            return info

        self.ep_optimizer.zero_grad()
        (loss * self.repre_loss_factor).backward()
        torch.nn.utils.clip_grad_norm_(self.ep.parameters(), self.ep_max_grad_norm)
        self.ep_optimizer.step()
        return info

    # ======================================================================
    # mode toggles
    # ======================================================================
    def prep_training(self):
        self.actor.train()
        self.ep.train()

    def prep_rollout(self):
        self.actor.eval()
        self.ep.eval()
