"""Runner for COMARL (Distributionally Robust Cooperative MARL) on HAPPO.

COMARL ("Distributionally Robust Cooperative Multi-agent RL with Value
Factorization") makes value learning robust to environmental uncertainty (sim-to-
real gap, model mismatch, system noise) by replacing the Bellman bootstrap with
its worst-case value over an uncertainty set of radius ``rho``.  The reference
implements this on discrete-action value-factorization Q-learning (VDN/QMIX/QTRAN);
the robust operator itself is mixer-agnostic (byte-for-byte identical in the VDN
and QMIX variants), so it ports directly onto HAPPO's centralized V-critic.

Adaptation (documented deviations from the discrete-Q original):

* Backbone is HAPPO/EP, identical to the other baselines.  COMARL's robust
  Bellman operator is injected into the critic's return computation: every
  bootstrap ``gamma * V(s')`` becomes ``gamma * sigma_rho(V(s'))`` (a *robust
  GAE*).  The critic and every per-agent advantage become distributionally
  robust, so the per-agent HAPPO policies are trained toward the robust team
  value -- the on-policy analog of the paper's DrIGM (decentralized robust
  execution from a centralized robust value).  ``rho = 0`` exactly recovers HAPPO.
* Two uncertainty models, exactly as in the reference:
    - ``contamination``: ``sigma = (1 - rho) * V'``           (base VDN/QMIX/QTRAN)
    - ``tv``           : ``sigma = (1 - rho) * g - (g - V')_+`` (the G-network "_g" variants)
  with the G-network ``g(share_obs)`` trained by the exact COMARL G-loss.
* The discrete VDN/QMIX/QTRAN mixer is not applicable to HAPPO's single
  continuous V-critic and is intentionally omitted; the robust operator (the
  paper's contribution) is mixer-agnostic and ported exactly.
"""

import numpy as np
import torch
import torch.nn.functional as F

from harl.common.buffers.robust_critic_buffer_ep import RobustCriticBufferEP
from harl.common.buffers.robust_critic_buffer_fp import RobustCriticBufferFP
from harl.models.comarl.comarl_modules import GNetwork
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.utils.trans_tools import _t2n


class OnPolicyComarlRunner(OnPolicyHARunner):
    """Runner for the COMARL algorithm."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyComarlRunner, self).__init__(args, algo_args, env_args)

        algo_cfg = algo_args["algo"]
        self.comarl_rho = float(algo_cfg.get("comarl_rho", 0.05))
        self.comarl_mode = str(algo_cfg.get("comarl_robust_type", "contamination"))
        assert self.comarl_mode in ("contamination", "tv"), (
            "comarl_robust_type must be 'contamination' or 'tv', got "
            f"{self.comarl_mode}"
        )
        self.comarl_g_lr = float(algo_cfg.get("comarl_g_lr", 1e-3))
        self.comarl_g_epochs = int(algo_cfg.get("comarl_g_epochs", 5))
        self.comarl_g_hidden = list(
            algo_cfg.get("comarl_g_hidden_sizes", algo_args["model"]["hidden_sizes"])
        )
        self.comarl_steps = 0

        if self.algo_args["render"]["use_render"]:
            return

        # ---- rebuild the critic buffer as the robust (COMARL) variant --------
        # EP (mamujoco Ant) and FP (SMAC/SMACv2) are both supported; the robust
        # operator is elementwise so it applies to per-agent FP values as well.
        share_observation_space = self.envs.share_observation_space[0]
        if self.state_type == "EP":
            self.critic_buffer = RobustCriticBufferEP(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                share_observation_space,
            )
        elif self.state_type == "FP":
            self.critic_buffer = RobustCriticBufferFP(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                share_observation_space,
                self.num_agents,
            )
        else:
            raise NotImplementedError

        # ---- G-network (TV-robust dual variable); only used in 'tv' mode -----
        self.state_dim = share_observation_space.shape[0]
        g_args = {**algo_args["model"]}
        g_args["hidden_sizes"] = self.comarl_g_hidden
        self.g_net = GNetwork(g_args, self.state_dim, device=self.device)
        self.g_optimizer = torch.optim.Adam(
            self.g_net.parameters(), lr=self.comarl_g_lr
        )

        if self.algo_args["train"]["model_dir"] is not None:
            self._restore_g_net()

    # ======================================================================
    # compute: robust returns (robust GAE bootstrap)
    # ======================================================================
    @torch.no_grad()
    def compute(self):
        """Compute robust returns and advantages for the centralized critic."""
        if self.state_type == "EP":
            next_value, _ = self.critic.get_values(
                self.critic_buffer.share_obs[-1],
                self.critic_buffer.rnn_states_critic[-1],
                self.critic_buffer.masks[-1],
            )
            next_value = _t2n(next_value)
        elif self.state_type == "FP":
            next_value, _ = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[-1]),
                np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                np.concatenate(self.critic_buffer.masks[-1]),
            )
            next_value = np.array(
                np.split(_t2n(next_value), self.algo_args["train"]["n_rollout_threads"])
            )

        g_values = self._compute_g_values() if self.comarl_mode == "tv" else None

        self.critic_buffer.compute_returns(
            next_value,
            self.value_normalizer,
            rho=self.comarl_rho,
            mode=self.comarl_mode,
            g_values=g_values,
        )

    def _compute_g_values(self):
        """G-network output g(s) for every stored state, in true (denorm) value scale.

        Returns a numpy array of the stored-state leading shape with a trailing
        singleton: EP -> (T+1, B, 1); FP -> (T+1, B, num_agents, 1). Shape-agnostic
        so the same code serves both state types.
        """
        share = self.critic_buffer.share_obs  # EP: (T+1,B,dim); FP: (T+1,B,N,dim)
        lead = share.shape[:-1]
        flat = share.reshape(-1, share.shape[-1])
        g = self.g_net.values_np(flat)  # (prod(lead), 1)
        return g.reshape(*lead, 1)

    # ======================================================================
    # train: HAPPO update on robust returns + G-network update (tv) + logging
    # ======================================================================
    def train(self):
        actor_train_infos, critic_train_info = super(
            OnPolicyComarlRunner, self
        ).train()

        g_loss = 0.0
        if self.comarl_mode == "tv":
            g_loss = self._update_g_network()

        self.comarl_steps += (
            self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        if self.writter is not None:
            self.writter.add_scalar("comarl/rho", self.comarl_rho, self.comarl_steps)
            self.writter.add_scalar(
                "comarl/robust_shrink", 1.0 - self.comarl_rho, self.comarl_steps
            )
            self.writter.add_scalar(
                "comarl/mean_return",
                float(np.mean(self.critic_buffer.returns[:-1])),
                self.comarl_steps,
            )
            if self.comarl_mode == "tv":
                self.writter.add_scalar("comarl/g_loss", g_loss, self.comarl_steps)
                g_arr = self._compute_g_values()
                self.writter.add_scalar(
                    "comarl/g_mean", float(np.mean(g_arr)), self.comarl_steps
                )

        return actor_train_infos, critic_train_info

    def _update_g_network(self):
        """Train g(s_t) against V(s_{t+1}) with the exact COMARL G-loss."""
        share = self.critic_buffer.share_obs  # (T+1, B, share_dim)
        value_preds = self.critic_buffer.value_preds  # (T+1, B, 1), normalized
        if self.value_normalizer is not None:
            V = self.value_normalizer.denormalize(value_preds)  # numpy, true scale
        else:
            V = value_preds

        share_t = share[:-1].reshape(-1, share.shape[-1])  # (T*B, share_dim): s_t
        v_next = V[1:].reshape(-1, 1)  # (T*B, 1): V(s_{t+1}), true scale
        v_next_t = torch.as_tensor(v_next, dtype=torch.float32, device=self.device)

        total = 0.0
        for _ in range(self.comarl_g_epochs):
            g = self.g_net(share_t)  # (T*B, 1), grad
            relu_term = torch.relu(g - v_next_t)
            loss = ((relu_term - (1.0 - self.comarl_rho) * g) ** 2).mean()
            self.g_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.g_net.parameters(), 10.0)
            self.g_optimizer.step()
            total += float(loss.item())
        return total / max(1, self.comarl_g_epochs)

    # ======================================================================
    # save / restore (also persist the G-network)
    # ======================================================================
    def save(self):
        super(OnPolicyComarlRunner, self).save()
        torch.save(self.g_net.state_dict(), str(self.save_dir) + "/comarl_g_net.pt")

    def restore(self):
        super(OnPolicyComarlRunner, self).restore()
        if self.algo_args["render"]["use_render"] or not hasattr(self, "g_net"):
            return
        self._restore_g_net()

    def _restore_g_net(self):
        state_dict = torch.load(
            str(self.algo_args["train"]["model_dir"]) + "/comarl_g_net.pt"
        )
        self.g_net.load_state_dict(state_dict)
