"""WISDOM algorithm for HARL.

WISDOM (Wavelet Predictive Representations for Non-Stationary RL,
arXiv:2510.04507) tackles non-stationarity by learning a *wavelet predictive
representation* of the task: a task-belief encoder maps recent transitions to a
low-dimensional latent ``z``, and a learnable multi-scale wavelet network refines
``z`` into ``pred_z`` while a **wavelet temporal-difference operator** makes the
low-frequency approximation track the discounted evolution of the task. The policy
conditions on ``(obs, pred_z)``.

This class adapts WISDOM to HARL's heterogeneous-agent setting following the same
recipe as the other representation-based non-stationary baselines (COREP / ESCP):

* The multi-agent RL backbone is **HAPPO**; WISDOM is added on top by conditioning
  each agent's policy on the wavelet representation, i.e. by **augmenting the
  observation** with ``pred_z`` (dim ``latent_dim``).
* Each agent owns an encoder + wavelet ``z_model`` (+ a soft-updated target wavelet
  net) with their own optimizers, trained by the WISDOM objective:
  ``kl_loss`` (encoder, KL-to-prior) and ``z_loss = pred_loss + td_coef * td_loss``
  (wavelet prediction + wavelet TD), exactly as in the reference
  ``reconstruction_trainer`` (encoder trained by KL only; the wavelet trained on the
  detached latent with the prediction and TD losses).

Rollout-time latent bookkeeping (carrying the previous raw observation across
steps, augmenting obs/state) is orchestrated by ``OnPolicyWisdomRunner``; this
class only provides the per-call primitives.
"""

import numpy as np
import torch
from gym.spaces import Box

from harl.algorithms.actors.happo import HAPPO
from harl.models.wisdom.wisdom_modules import WaveletYNetwork, WisdomEncoder
from harl.utils.discrete_util import get_encoded_act_dim, encode_actions_torch


class WISDOM(HAPPO):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize WISDOM algorithm.

        Args:
            args: (dict) merged ``{**model, **algo}`` arguments.
            obs_space: (gym.spaces.Box) *raw* observation space of the agent.
            act_space: (gym.spaces.Box) action space of the agent.
            device: (torch.device) device for tensor operations.
        """
        # ---- WISDOM hyper-parameters (read before building the actor) ----
        self.latent_dim = int(args.get("wisdom_latent_dim", 5))
        self.lr_encoder = float(args.get("wisdom_lr_encoder", 3e-4))
        self.td_loss_coefficient = float(args.get("wisdom_td_loss_coefficient", 0.1))
        self.kl_weight = float(args.get("wisdom_kl_weight", 0.1))
        self.wavelet_dimension = int(args.get("wisdom_wavelet_dimension", 1))
        self.wavelet_filter_size = int(args.get("wisdom_wavelet_filter_size", 2))
        self.wavelet_depth = int(args.get("wisdom_wavelet_depth", 2))
        self.wavelet_dropout = float(args.get("wisdom_wavelet_dropout", 0.2))
        self.gamma_z = float(args.get("wisdom_gamma_z", 0.99))
        self.soft_target_tau = float(args.get("wisdom_soft_target_tau", 5e-3))
        self.num_repr_updates = int(args.get("wisdom_num_repr_updates", 1))
        self.encoder_max_grad_norm = args.get("wisdom_encoder_max_grad_norm", 10.0)
        enc_hidden = args.get("wisdom_encoder_hidden", [200, 200, 200])
        self.encoder_hidden = [int(x) for x in enc_hidden]

        # ---- raw obs / action dims --------------------------------------------
        assert obs_space.__class__.__name__ == "Box", "WISDOM expects Box observations."
        self.raw_obs_dim = int(obs_space.shape[0])
        # Encoded action dim: shape[0] for continuous Box, n (one-hot) for Discrete
        # (SMAC/SMACv2). Discrete actions are one-hot encoded inside the transition
        # (obs, action, reward, next_obs) fed to the encoder.
        self.act_space = act_space
        self.act_dim = get_encoded_act_dim(act_space)
        # encoder input is a transition (obs, action, reward, next_obs)
        self.encoder_input_dim = 2 * self.raw_obs_dim + self.act_dim + 1
        self.latent_concat_dim = self.latent_dim  # pred_z augments the observation

        # ---- build the actor on the AUGMENTED observation space ---------------
        augmented_low = np.concatenate(
            [
                np.full(self.raw_obs_dim, -np.inf, dtype=np.float32),
                np.full(self.latent_concat_dim, -np.inf, dtype=np.float32),
            ]
        )
        augmented_high = -augmented_low
        augmented_obs_space = Box(low=augmented_low, high=augmented_high, dtype=np.float32)
        super(WISDOM, self).__init__(args, augmented_obs_space, act_space, device)

        # ---- build the WISDOM encoder + wavelet networks ----------------------
        self.encoder = WisdomEncoder(
            self.encoder_input_dim, self.latent_dim, self.encoder_hidden
        ).to(device)
        self.z_model = WaveletYNetwork(
            d_model=self.wavelet_dimension,
            kernel_size=self.wavelet_filter_size,
            depth=self.wavelet_depth,
            dropout=self.wavelet_dropout,
        ).to(device)
        self.target_z_model = WaveletYNetwork(
            d_model=self.wavelet_dimension,
            kernel_size=self.wavelet_filter_size,
            depth=self.wavelet_depth,
            dropout=self.wavelet_dropout,
        ).to(device)
        self.target_z_model.load_state_dict(self.z_model.state_dict())
        for p in self.target_z_model.parameters():
            p.requires_grad = False

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=self.lr_encoder
        )
        self.z_optimizer = torch.optim.Adam(
            self.z_model.parameters(), lr=self.lr_encoder
        )

    # ======================================================================
    # Rollout-time latent primitives (called by the runner)
    # ======================================================================
    @torch.no_grad()
    def init_latent(self, n_threads):
        """Return the prior wavelet representation (from a zero latent)."""
        z = torch.zeros(n_threads, self.latent_dim, device=self.device)
        pred_z, _ = self.z_model(z)
        return pred_z.cpu().numpy()

    @torch.no_grad()
    def step_latent(self, prev_obs, action, reward, next_obs):
        """Encode the latest transition and produce the wavelet representation.

        Args:
            prev_obs: (np.ndarray) (n_threads, raw_obs_dim) obs before the action.
            action: (np.ndarray) (n_threads, act_dim) action taken.
            reward: (np.ndarray) (n_threads, 1) reward received.
            next_obs: (np.ndarray) (n_threads, raw_obs_dim) resulting raw obs.
        Returns:
            pred_z: (np.ndarray) (n_threads, latent_dim) wavelet representation.
        """
        device = self.device
        prev_obs_t = torch.as_tensor(
            np.asarray(prev_obs, dtype=np.float32), device=device
        )
        # one-hot encode discrete actions (no-op for continuous Box)
        action_t = encode_actions_torch(
            torch.as_tensor(np.asarray(action, dtype=np.float32), device=device),
            self.act_space,
        )
        reward_t = torch.as_tensor(
            np.asarray(reward, dtype=np.float32).reshape(-1, 1), device=device
        )
        next_obs_t = torch.as_tensor(
            np.asarray(next_obs, dtype=np.float32), device=device
        )
        enc_in = torch.cat([prev_obs_t, action_t, reward_t, next_obs_t], dim=-1)
        z, _, _ = self.encoder.infer_posterior(enc_in, deterministic=True)
        pred_z, _ = self.z_model(z)
        return pred_z.cpu().numpy()

    # ======================================================================
    # WISDOM representation update (encoder KL + wavelet prediction + TD)
    # ======================================================================
    def update_representation(self, obs_seq, act_seq, rew_seq, mask_seq):
        """One WISDOM representation update over a batch of rollout trajectories.

        Args:
            obs_seq: (np.ndarray) (T+1, B, raw_obs_dim) raw observations.
            act_seq: (np.ndarray) (T, B, act_dim) actions.
            rew_seq: (np.ndarray) (T, B, 1) rewards.
            mask_seq: (np.ndarray) (T, B, 1) 1 if the transition is valid.
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

        prev_obs = obs[:-1]  # (T, B, obs)
        next_obs = obs[1:]  # (T, B, obs)
        enc_in = torch.cat((prev_obs, act, rew, next_obs), dim=-1)  # (T, B, in)
        T, B = enc_in.shape[0], enc_in.shape[1]
        enc_in_flat = enc_in.reshape(T * B, -1)
        mask_flat = mask.reshape(T * B, 1)
        mask_sum = mask_flat.sum().clamp(min=1.0)

        last_info = {}
        for _ in range(self.num_repr_updates):
            # ---- encoder: KL-to-prior (the only encoder signal, as in WISDOM) --
            # deterministic (mean) latent to match the rollout augmentation
            z, mu, var = self.encoder.infer_posterior(enc_in_flat, deterministic=True)
            kl_loss = self.encoder.kl_to_prior(mu, var)
            self.encoder_optimizer.zero_grad()
            (self.kl_weight * kl_loss).backward()
            if self.encoder_max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.encoder_max_grad_norm
                )
            self.encoder_optimizer.step()

            # ---- wavelet: prediction loss + wavelet TD on the detached latent --
            z_det = z.detach()
            pred_z, res_lo = self.z_model(z_det)  # (T*B, latent)
            pred_err = (pred_z - z_det).pow(2).mean(dim=-1, keepdim=True)
            pred_loss = (pred_err * mask_flat).sum() / mask_sum

            # wavelet TD: res_lo(z_t) <- z_t + gamma * target_res_lo(z_{t+1})
            z_seq = z_det.reshape(T, B, self.latent_dim)
            res_lo_seq = res_lo.reshape(T, B, self.latent_dim)
            with torch.no_grad():
                _, next_res_lo = self.target_z_model(
                    z_seq[1:].reshape((T - 1) * B, self.latent_dim)
                )
                next_res_lo = next_res_lo.reshape(T - 1, B, self.latent_dim)
                target_res_lo = z_seq[:-1] + self.gamma_z * next_res_lo
            td_err = (res_lo_seq[:-1] - target_res_lo).pow(2).mean(dim=-1, keepdim=True)
            td_mask = mask[:-1]
            td_loss = torch.sqrt(
                (td_err * td_mask).sum() / td_mask.sum().clamp(min=1.0) + 1e-12
            )

            z_loss = pred_loss + self.td_loss_coefficient * td_loss
            self.z_optimizer.zero_grad()
            z_loss.backward()
            self.z_optimizer.step()

            # soft-update the target wavelet network
            with torch.no_grad():
                for tp, p in zip(
                    self.target_z_model.parameters(), self.z_model.parameters()
                ):
                    tp.data.mul_(1.0 - self.soft_target_tau)
                    tp.data.add_(self.soft_target_tau * p.data)

            last_info = {
                "wisdom_kl_loss": float(kl_loss.item()),
                "wisdom_pred_loss": float(pred_loss.item()),
                "wisdom_td_loss": float(td_loss.item()),
                "wisdom_z_loss": float(z_loss.item()),
            }
        return last_info

    # ======================================================================
    # mode toggles (also toggle the WISDOM modules, not just the actor)
    # ======================================================================
    def prep_training(self):
        self.actor.train()
        self.encoder.train()
        self.z_model.train()

    def prep_rollout(self):
        self.actor.eval()
        self.encoder.eval()
        self.z_model.eval()
