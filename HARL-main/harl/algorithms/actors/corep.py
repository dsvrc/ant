"""COREP algorithm for HARL.

COREP (Causal-Origin REPresentation, ICML 2024) tackles non-stationarity by
learning a stable graph representation of the state with a dual-GAT and a guided
(TD-error gated) updating mechanism, wrapped in a VariBAD-style variational
belief encoder. This class adapts COREP to HARL's heterogeneous-agent setting:

* The underlying multi-agent RL algorithm is **HAPPO** (sequential policy update
  with the importance-correction ``factor`` and a centralized critic) -- COREP is
  added *on top* of it, so this is genuinely "HAPPO + the COREP representation".
* Each agent owns a COREP module (encoder + reward/state decoders). The agent's
  policy (and, optionally, the centralized critic) is conditioned on the
  causal-origin latent by **augmenting the observation** with ``(mu, logvar)``.
* The VAE is trained with the COREP objective: reward + state reconstruction, KL,
  and the three graph-regularization terms (graph consistency / sparsity /
  magnitude-symmetry). The stable GAT is frozen when the guided-updating test
  (handled by the runner, which owns the shared TD buffer) says so.

All the rollout-time latent bookkeeping (carrying GRU hidden states across steps
and episodes, resetting on ``done``) is orchestrated by
``OnPolicyCorepRunner``; this class only provides the per-call primitives.
"""

import numpy as np
import torch
from gym.spaces import Box

from harl.algorithms.actors.happo import HAPPO
from harl.models.corep.corep_modules import (
    CorepEncoder,
    RewardDecoder,
    StateTransitionDecoder,
    sigmoid_adjacency_matrix,
)
from harl.utils.discrete_util import get_encoded_act_dim, encode_actions_torch


class COREP(HAPPO):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize COREP algorithm.

        Args:
            args: (dict) merged ``{**model, **algo}`` arguments.
            obs_space: (gym.spaces.Box) *raw* observation space of the agent.
            act_space: (gym.spaces.Box) action space of the agent.
            device: (torch.device) device for tensor operations.
        """
        # ---- COREP / VAE hyper-parameters (read before building the actor) ----
        self.latent_dim = int(args.get("corep_latent_dim", 8))
        self.num_nodes = int(args.get("corep_num_nodes", 8))
        self.num_heads = int(args.get("corep_num_heads", 4))
        self.node_feature_dim = int(args.get("corep_node_feature_dim", 16))
        self.gat_hidden_dim = int(args.get("corep_gat_hidden_dim", 32))
        self.state_embed_dim = int(args.get("corep_state_embed_dim", 16))
        self.action_embed_dim = int(args.get("corep_action_embed_dim", 16))
        self.reward_embed_dim = int(args.get("corep_reward_embed_dim", 16))
        self.gru_hidden_size = int(args.get("corep_gru_hidden_size", 128))
        self.gat_dropout = float(args.get("corep_gat_dropout", 0.6))
        self.graph_threshold = float(args.get("corep_graph_threshold", 0.8))
        self.graph_temperature = float(args.get("corep_graph_temperature", 0.8))
        self.similarity_threshold = float(args.get("corep_similarity_threshold", 0.7))

        self.lr_vae = float(args.get("lr_vae", 5e-4))
        self.kl_weight = float(args.get("corep_kl_weight", 0.1))
        self.rew_loss_coeff = float(args.get("corep_rew_loss_coeff", 1.0))
        self.state_loss_coeff = float(args.get("corep_state_loss_coeff", 1.0))
        self.graph_loss_coeff = float(args.get("corep_graph_loss_coeff", 0.1))
        self.sparsity_loss_coeff = float(args.get("corep_sparsity_loss_coeff", 0.001))
        self.mag_loss_coeff = float(args.get("corep_mag_loss_coeff", 0.001))
        self.decode_reward = bool(args.get("corep_decode_reward", True))
        self.decode_state = bool(args.get("corep_decode_state", True))
        self.kl_to_gauss_prior = bool(args.get("corep_kl_to_gauss_prior", False))
        self.num_vae_updates = int(args.get("corep_num_vae_updates", 1))
        self.tbptt_stepsize = args.get("corep_tbptt_stepsize", 50)
        if self.tbptt_stepsize is not None:
            self.tbptt_stepsize = int(self.tbptt_stepsize)
        self.encoder_max_grad_norm = args.get("corep_encoder_max_grad_norm", 1.0)
        decoder_layers = args.get("corep_decoder_layers", [64, 32])
        self.decoder_layers = [int(x) for x in decoder_layers]

        # ---- raw obs / action dims --------------------------------------------
        assert obs_space.__class__.__name__ == "Box", "COREP expects Box observations."
        self.raw_obs_dim = int(obs_space.shape[0])
        # Encoded action dim: shape[0] for continuous Box, n (one-hot) for Discrete
        # (SMAC/SMACv2). Discrete actions are one-hot encoded before being fed to
        # the encoder/decoders; ``self.act_space`` drives that encoding.
        self.act_space = act_space
        self.act_dim = get_encoded_act_dim(act_space)
        self.latent_concat_dim = 2 * self.latent_dim  # (mu, logvar) fed to the policy

        # ---- build the actor on the AUGMENTED observation space ---------------
        augmented_low = np.concatenate(
            [
                np.full(self.raw_obs_dim, -np.inf, dtype=np.float32),
                np.full(self.latent_concat_dim, -np.inf, dtype=np.float32),
            ]
        )
        augmented_high = -augmented_low
        augmented_obs_space = Box(low=augmented_low, high=augmented_high, dtype=np.float32)
        super(COREP, self).__init__(args, augmented_obs_space, act_space, device)

        # ---- build the COREP encoder + decoders -------------------------------
        self.encoder = CorepEncoder(
            state_dim=self.raw_obs_dim,
            action_dim=self.act_dim,
            latent_dim=self.latent_dim,
            num_nodes=self.num_nodes,
            num_heads=self.num_heads,
            node_feature_dim=self.node_feature_dim,
            gat_hidden_dim=self.gat_hidden_dim,
            state_embed_dim=self.state_embed_dim,
            action_embed_dim=self.action_embed_dim,
            reward_embed_dim=self.reward_embed_dim,
            gru_hidden_size=self.gru_hidden_size,
            gat_dropout=self.gat_dropout,
            graph_threshold=self.graph_threshold,
            graph_temperature=self.graph_temperature,
        ).to(device)

        vae_params = list(self.encoder.parameters())
        self.reward_decoder = None
        self.state_decoder = None
        if self.decode_reward:
            self.reward_decoder = RewardDecoder(
                self.latent_dim, self.raw_obs_dim, self.act_dim, self.decoder_layers
            ).to(device)
            vae_params += list(self.reward_decoder.parameters())
        if self.decode_state:
            self.state_decoder = StateTransitionDecoder(
                self.latent_dim, self.raw_obs_dim, self.act_dim, self.decoder_layers
            ).to(device)
            vae_params += list(self.state_decoder.parameters())

        self.vae_optimizer = torch.optim.Adam(vae_params, lr=self.lr_vae)

    # ======================================================================
    # Rollout-time latent primitives (called by the runner)
    # ======================================================================
    @torch.no_grad()
    def init_latent(self, n_threads):
        """Return the prior latent ``(mu, logvar)`` and GRU hidden state.

        Returns:
            latent_concat: (np.ndarray) shape (n_threads, 2 * latent_dim).
            hidden: (torch.Tensor) shape (1, n_threads, gru_hidden_size).
        """
        _, mean, logvar, hidden = self.encoder.prior(n_threads)
        latent_concat = torch.cat((mean, logvar), dim=-1).squeeze(0)
        return latent_concat.cpu().numpy(), hidden.detach()

    @torch.no_grad()
    def step_latent(self, action, reward, next_obs, done, hidden):
        """Advance the belief by one environment step.

        Args:
            action: (np.ndarray) (n_threads, act_dim) action just taken.
            reward: (np.ndarray) (n_threads, 1) reward just received.
            next_obs: (np.ndarray) (n_threads, raw_obs_dim) raw next observation.
            done: (np.ndarray) (n_threads,) or (n_threads, 1) episode-done flags.
            hidden: (torch.Tensor) (1, n_threads, gru_hidden_size) current GRU state.
        Returns:
            latent_concat: (np.ndarray) (n_threads, 2 * latent_dim).
            hidden: (torch.Tensor) (1, n_threads, gru_hidden_size) updated GRU state.
        """
        device = self.device
        action_raw = torch.as_tensor(action, dtype=torch.float32, device=device)
        # one-hot encode discrete actions (no-op for continuous Box)
        action_t = encode_actions_torch(action_raw, self.act_space).unsqueeze(0)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=device).reshape(1, -1, 1)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        done_t = torch.as_tensor(np.asarray(done), dtype=torch.float32, device=device).reshape(1, -1, 1)

        hidden = self.encoder.reset_hidden(hidden, done_t)
        _, mean, logvar, hidden, _, _ = self.encoder(
            actions=action_t,
            states=next_obs_t,
            rewards=reward_t,
            hidden_state=hidden,
            return_prior=False,
            sample=False,
        )
        latent_concat = torch.cat((mean, logvar), dim=-1).squeeze(0)
        return latent_concat.cpu().numpy(), hidden.detach()

    # ======================================================================
    # VAE / COREP representation update
    # ======================================================================
    def _compute_kl(self, latent_mean, latent_logvar):
        """KL term of the ELBO. Returns (T, B) tensor (prior dropped)."""
        if self.kl_to_gauss_prior:
            kl = -0.5 * (
                1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()
            ).sum(dim=-1)
            return kl[1:]
        gauss_dim = latent_mean.shape[-1]
        mu = latent_mean[1:]
        m = latent_mean[:-1]
        logE = latent_logvar[1:]
        logS = latent_logvar[:-1]
        kl = 0.5 * (
            torch.sum(logS, dim=-1)
            - torch.sum(logE, dim=-1)
            - gauss_dim
            + torch.sum(torch.exp(logE) / torch.exp(logS), dim=-1)
            + torch.sum((m - mu).pow(2) / torch.exp(logS), dim=-1)
        )
        return kl

    def update_vae(self, obs_seq, act_seq, rew_seq, mask_seq, freeze):
        """One COREP VAE update over a batch of trajectories.

        Args:
            obs_seq: (np.ndarray) (T+1, B, raw_obs_dim) raw observations.
            act_seq: (np.ndarray) (T, B, act_dim) actions.
            rew_seq: (np.ndarray) (T, B, 1) rewards.
            mask_seq: (np.ndarray) (T, B, 1) 1 if the step is valid (not just-reset).
            freeze: (bool) if True, the stable GAT is not updated this step.
        Returns:
            info: (dict) scalar loss components for logging.
        """
        device = self.device
        obs = torch.as_tensor(obs_seq, dtype=torch.float32, device=device)
        act = torch.as_tensor(act_seq, dtype=torch.float32, device=device)
        # one-hot encode discrete actions (no-op for continuous Box) -> (T, B, act_dim)
        act = encode_actions_torch(act, self.act_space)
        rew = torch.as_tensor(rew_seq, dtype=torch.float32, device=device)
        mask = torch.as_tensor(mask_seq, dtype=torch.float32, device=device)

        prev_obs = obs[:-1]  # (T, B, obs)  s_t
        next_obs = obs[1:]  # (T, B, obs)   s_{t+1}

        (
            latent_sample,
            latent_mean,
            latent_logvar,
            _,
            stable_nodes,
            flexible_nodes,
        ) = self.encoder(
            actions=act,
            states=next_obs,
            rewards=rew,
            hidden_state=None,
            return_prior=True,
            sample=True,
            detach_every=self.tbptt_stepsize,
            freeze=freeze,
        )

        # posterior latent aligned to each transition (drop the prior at index 0)
        latent_post = latent_sample[1:]  # (T, B, latent_dim)

        mask_sum = mask.sum().clamp(min=1.0)

        # --- reconstruction losses (masked mean over valid steps) -------------
        rew_loss = torch.zeros((), device=device)
        if self.decode_reward:
            rew_pred = self.reward_decoder(latent_post, prev_obs, act, next_obs)
            rew_err = (rew_pred - rew).pow(2)  # (T, B, 1)
            rew_loss = (rew_err * mask).sum() / mask_sum

        state_loss = torch.zeros((), device=device)
        if self.decode_state:
            state_pred = self.state_decoder(latent_post, prev_obs, act)
            state_err = (state_pred - next_obs).pow(2).mean(dim=-1, keepdim=True)
            state_loss = (state_err * mask).sum() / mask_sum

        # --- KL term (masked mean) -------------------------------------------
        kl = self._compute_kl(latent_mean, latent_logvar)  # (T, B)
        kl_loss = (kl * mask.squeeze(-1)).sum() / mask_sum

        # --- graph-regularization losses -------------------------------------
        stable_adj = sigmoid_adjacency_matrix(stable_nodes, self.similarity_threshold)
        flexible_adj = sigmoid_adjacency_matrix(flexible_nodes, self.similarity_threshold)

        graph_loss = torch.abs(stable_adj - flexible_adj).mean()
        sparsity_loss = stable_adj.abs().mean() + flexible_adj.abs().mean()
        mag_loss = (
            torch.abs(stable_adj - stable_adj.transpose(-2, -1)).mean()
            + torch.abs(flexible_adj - flexible_adj.transpose(-2, -1)).mean()
        )

        loss = (
            self.rew_loss_coeff * rew_loss
            + self.state_loss_coeff * state_loss
            + self.kl_weight * kl_loss
            + self.graph_loss_coeff * graph_loss
            + self.sparsity_loss_coeff * sparsity_loss
            + self.mag_loss_coeff * mag_loss
        )

        self.vae_optimizer.zero_grad()
        loss.backward()
        if self.encoder_max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), self.encoder_max_grad_norm
            )
        self.vae_optimizer.step()

        return {
            "vae_loss": loss.item(),
            "vae_rew_recon": float(rew_loss.item()),
            "vae_state_recon": float(state_loss.item()),
            "vae_kl": float(kl_loss.item()),
            "vae_graph_loss": float(graph_loss.item()),
            "vae_sparsity_loss": float(sparsity_loss.item()),
            "vae_mag_loss": float(mag_loss.item()),
        }

    # ======================================================================
    # mode toggles (also toggle the COREP modules, not just the actor)
    # ======================================================================
    def prep_training(self):
        self.actor.train()
        self.encoder.train()
        if self.reward_decoder is not None:
            self.reward_decoder.train()
        if self.state_decoder is not None:
            self.state_decoder.train()

    def prep_rollout(self):
        self.actor.eval()
        self.encoder.eval()
        if self.reward_decoder is not None:
            self.reward_decoder.eval()
        if self.state_decoder is not None:
            self.state_decoder.eval()
