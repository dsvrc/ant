"""TRIO algorithm for HARL.

TRIO ("Meta-Reinforcement Learning by Tracking task non-stationarity", IJCAI 2021)
infers a task latent with a variational module and *tracks its evolution* over
time, conditioning the policy on the inferred (or, at training, the true) task.

This class adapts TRIO onto HARL's HAPPO:

* Each agent owns a variational inference (VI) network that infers a task latent
  from its (action, reward, next_obs) context, taking the previous posterior as
  its prior (Bayesian filtering -> online tracking).
* The policy/critic are conditioned on the task by **augmenting the observation**
  with the task latent (``latent_dim`` extra features).
* Following TRIO, the policy is trained with the **oracle** task (the env's
  exposed non-stationary context, ``info["ambient"]`` for the TCC Ant) appended,
  while the VI is trained *supervised* to predict that task
  (``loss_inference_closed_form`` = MSE + posterior-variance + KL-to-prior).
* At evaluation the oracle is replaced by a **Thompson sample** from the inferred
  posterior, produced by the online filter (so the policy is deployable without
  privileged information). This is the standard TRIO oracle-train / inferred-test
  protocol; the EVAL curve is TRIO's true (inference-only) performance.

The runner (``OnPolicyTrioRunner``) orchestrates the oracle augmentation, label
collection, VI training, and the eval-time online filter.
"""

import numpy as np
import torch
from gym.spaces import Box

from harl.algorithms.actors.happo import HAPPO
from harl.models.trio.trio_modules import InferenceNetwork, loss_inference_closed_form
from harl.utils.discrete_util import get_encoded_act_dim, encode_actions_torch


class TRIO(HAPPO):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize TRIO.

        Args:
            args: (dict) merged ``{**model, **algo}`` arguments.
            obs_space: (gym.spaces.Box) *raw* observation space.
            act_space: (gym.spaces.Box) action space.
            device: (torch.device) device.
        """
        # ---- TRIO hyper-parameters ----
        self.latent_dim = int(args.get("trio_latent_dim", 1))
        self.vi_gru_hidden = int(args.get("trio_vi_gru_hidden", 64))
        self.vi_hidden2 = int(args.get("trio_vi_hidden2", 16))
        self.vae_lr = float(args.get("trio_vae_lr", 1e-3))
        self.prior_mu = float(args.get("trio_prior_mu", 0.5))
        self.prior_var = float(args.get("trio_prior_var", 0.25))
        self.use_decay_kld = bool(args.get("trio_use_decay_kld", True))
        self.decay_kld_rate = float(args.get("trio_decay_kld_rate", 0.1))
        self.vae_max_window = args.get("trio_vae_max_window", None)
        if self.vae_max_window is not None:
            self.vae_max_window = int(self.vae_max_window)
        self.num_vi_updates = int(args.get("trio_num_vi_updates", 1))
        self.vi_max_grad_norm = float(args.get("trio_vi_max_grad_norm", 1.0))

        assert obs_space.__class__.__name__ == "Box", "TRIO expects Box observations."
        self.raw_obs_dim = int(obs_space.shape[0])
        # Encoded action dim: shape[0] for continuous Box, n (one-hot) for Discrete
        # (SMAC/SMACv2). Discrete actions are one-hot encoded inside the VI context.
        self.act_space = act_space
        self.act_dim = get_encoded_act_dim(act_space)
        self.ctx_dim = self.act_dim + 1 + self.raw_obs_dim  # action + reward + next_obs

        # ---- actor on AUGMENTED observation (raw_obs + task latent) ----
        low = np.concatenate(
            [
                np.full(self.raw_obs_dim, -np.inf, dtype=np.float32),
                np.full(self.latent_dim, -np.inf, dtype=np.float32),
            ]
        )
        augmented_obs_space = Box(low=low, high=-low, dtype=np.float32)
        super(TRIO, self).__init__(args, augmented_obs_space, act_space, device)

        # ---- variational inference network ----
        self.vi = InferenceNetwork(
            ctx_dim=self.ctx_dim,
            z_dim=self.latent_dim,
            gru_hidden=self.vi_gru_hidden,
            hidden2=self.vi_hidden2,
        ).to(device)
        self.vi_optimizer = torch.optim.Adam(self.vi.parameters(), lr=self.vae_lr)

        # fixed broad prior over the task latent
        self._prior_mu_vec = torch.full(
            (1, self.latent_dim), self.prior_mu, dtype=torch.float32, device=device
        )
        self._prior_logvar_vec = torch.full(
            (1, self.latent_dim),
            float(np.log(self.prior_var)),
            dtype=torch.float32,
            device=device,
        )

    # ======================================================================
    # belief priors / online filtering (eval)
    # ======================================================================
    def init_belief(self, n_threads):
        """Return the prior belief (mu, logvar) as numpy (n_threads, 2*latent_dim)."""
        prior = torch.cat(
            [self._prior_mu_vec, self._prior_logvar_vec], dim=-1
        ).repeat(n_threads, 1)
        return prior.cpu().numpy()

    @torch.no_grad()
    def filter_step(self, action, reward, next_obs, prior, hidden, seq_len):
        """One online Bayesian-filter step (eval).

        Args:
            action: (np.ndarray) (n, act_dim)
            reward: (np.ndarray) (n, 1)
            next_obs: (np.ndarray) (n, raw_obs_dim)
            prior: (np.ndarray) (n, 2*latent_dim) current belief [mu, logvar]
            hidden: (torch.Tensor) GRU state or None
            seq_len: (int) samples seen so far
        Returns:
            posterior: (np.ndarray) (n, 2*latent_dim) [mu, logvar]
            z_sample: (np.ndarray) (n, latent_dim) Thompson sample
            hidden: (torch.Tensor) updated GRU state
            total_len: (int)
        """
        device = self.device
        a = torch.as_tensor(action, dtype=torch.float32, device=device)
        # one-hot encode discrete actions (no-op for continuous Box) -> (n, act_dim)
        a = encode_actions_torch(a, self.act_space)
        r = torch.as_tensor(reward, dtype=torch.float32, device=device).reshape(-1, 1)
        s = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        context = torch.cat([a, r, s], dim=-1).unsqueeze(1)  # (n, 1, ctx_dim)
        prior_t = torch.as_tensor(prior, dtype=torch.float32, device=device)

        mu, logvar, hidden, total_len = self.vi(
            context, prior_t, hidden=hidden, seq_len_so_far=seq_len
        )
        z = self.vi.reparameterize(mu, logvar)
        posterior = torch.cat([mu, logvar], dim=-1)
        return posterior.cpu().numpy(), z.cpu().numpy(), hidden.detach(), total_len

    # ======================================================================
    # supervised VI training (window-based, on the collected rollout)
    # ======================================================================
    def update_vi(self, obs_seq, act_seq, rew_seq, task_labels, mask_seq):
        """One supervised VI update on a random context window.

        Args:
            obs_seq: (np.ndarray) (T+1, B, raw_obs_dim) raw observations.
            act_seq: (np.ndarray) (T, B, act_dim).
            rew_seq: (np.ndarray) (T, B, 1).
            task_labels: (np.ndarray) (T, B, latent_dim) true task per transition.
            mask_seq: (np.ndarray) (T, B, 1) (unused for now; kept for parity).
        Returns:
            info: (dict) loss components.
        """
        device = self.device
        T = act_seq.shape[0]
        B = act_seq.shape[1]

        # random contiguous window, TRIO-style (varying context length)
        max_w = T if self.vae_max_window is None else min(T, self.vae_max_window)
        max_w = max(1, max_w)
        L = int(np.random.randint(1, max_w + 1))
        start = int(np.random.randint(0, T - L + 1))

        act = torch.as_tensor(act_seq[start : start + L], dtype=torch.float32, device=device)
        # one-hot encode discrete actions (no-op for continuous Box) -> (L, B, act_dim)
        act = encode_actions_torch(act, self.act_space)
        rew = torch.as_tensor(rew_seq[start : start + L], dtype=torch.float32, device=device)
        nobs = torch.as_tensor(
            obs_seq[start + 1 : start + L + 1], dtype=torch.float32, device=device
        )
        # (L, B, ctx_dim) -> (B, L, ctx_dim)
        context = torch.cat([act, rew, nobs], dim=-1).transpose(0, 1)

        # target task = the task at the end of the window
        z = torch.as_tensor(
            task_labels[start + L - 1], dtype=torch.float32, device=device
        )  # (B, latent_dim)

        mu_prior = self._prior_mu_vec.repeat(B, 1)
        logvar_prior = self._prior_logvar_vec.repeat(B, 1)
        prior = torch.cat([mu_prior, logvar_prior], dim=-1)

        mu_hat, logvar_hat, _, _ = self.vi(context, prior, hidden=None, seq_len_so_far=0)

        loss, kld, mse = loss_inference_closed_form(
            z=z,
            mu_hat=mu_hat,
            logvar_hat=logvar_hat,
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
            n_samples=L,
            use_decay=self.use_decay_kld,
            decay_param=self.decay_kld_rate,
        )

        self.vi_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vi.parameters(), self.vi_max_grad_norm)
        self.vi_optimizer.step()

        return {"vi_loss": float(loss.item()), "vi_kld": kld, "vi_mse": mse}

    # ======================================================================
    # mode toggles
    # ======================================================================
    def prep_training(self):
        self.actor.train()
        self.vi.train()

    def prep_rollout(self):
        self.actor.eval()
        self.vi.eval()
