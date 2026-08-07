"""ERNIE algorithm for HARL.

ERNIE (Bukharin et al., "Robust Multi-Agent Reinforcement Learning via
Adversarial Regularization: Theoretical Foundation and Stable Algorithms",
NeurIPS 2023; arXiv:2310.10810) makes MARL policies robust to noisy
observations, changing transition dynamics, and adversarial perturbations by
*controlling the Lipschitz constant of the policy with respect to its
observation*.  Robustness is obtained by an **adversarial regularizer** that
penalizes how much the policy output changes between the true observation ``o``
and a worst-case perturbed observation ``o + delta`` inside an epsilon-ball:

    R_pi(o; theta) = max_{||delta|| <= eps}  D( pi_theta(o + delta), pi_theta(o) )

and the overall objective for each agent n is (Eq. 5 of the paper)

    min_theta  L(theta)  +  lambda * E_{o ~ pi_n}[ R_pi(o_n; theta_n) ].

For *stochastic* policies (HAPPO / MAPPO) the discrepancy ``D`` is the KL
divergence between the two action distributions; for deterministic policies the
paper uses an l_p norm.  The inner maximization is solved by a few steps of
projected gradient ascent ("attack") on ``delta``.

This class adapts ERNIE onto HARL's HAPPO with a minimal, faithful design that
mirrors the LCPO integration (``harl/algorithms/actors/lcpo.py``):

* Only the **actor update** changes -- observations, the centralized critic, and
  HAPPO's sequential-update / ``factor`` machinery are all untouched.
* The clipped-surrogate policy loss is computed exactly as in HAPPO, then the
  ERNIE adversarial regularizer ``lambda * R_pi`` is added before ``backward``.
* The regularizer is computed entirely from the minibatch observations, so -
  unlike COREP / LCPO - **no extra buffer and no custom runner are required**;
  ``ernie`` is registered onto the standard ``OnPolicyHARunner``.

Two variants of the inner attack are provided, toggled by ``ernie_stackelberg``:

* ``False`` (default, "simplest version" from the paper's README): the perturbation
  ``delta`` is detached after the PGD attack, so the regularizer gradient flows
  only through ``theta`` (decoupled adversarial training).
* ``True``: the PGD attack is differentiated through (``create_graph=True``) and
  ``delta`` is *not* detached, so ``dR/dtheta`` includes the attacker's reaction
  ``d delta*/d theta``.  This is the unrolled-differentiation realization of the
  paper's **Stackelberg game** formulation (Eq. 6-7), in which the leader
  (defender/policy) anticipates the follower (attacker/perturbation) -- it gives
  the same leader-follower gradient as the paper's Hessian-vector approach but in
  a single extra backward pass.

When ``ernie_lambda == 0`` or ``ernie_perturb_steps == 0`` the update reduces
exactly to HAPPO (the built-in fallback).
"""

import numpy as np
import torch
import torch.nn as nn

from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.happo import HAPPO


class ERNIE(HAPPO):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize ERNIE algorithm.

        Args:
            args: (dict) merged ``{**model, **algo}`` arguments.
            obs_space: (gym.spaces) observation space of the agent.
            act_space: (gym.spaces) action space of the agent.
            device: (torch.device) device for tensor operations.
        """
        super(ERNIE, self).__init__(args, obs_space, act_space, device)

        # ----- ERNIE-specific hyperparameters -----
        # weight of the adversarial regularizer (lambda in Eq. 5)
        self.ernie_lambda = float(args.get("ernie_lambda", 0.05))
        # number of PGD attack steps used to approximate the worst-case delta (K)
        self.ernie_perturb_steps = int(args.get("ernie_perturb_steps", 1))
        # PGD step size (alpha)
        self.ernie_perturb_alpha = float(args.get("ernie_perturb_alpha", 0.05))
        # epsilon-ball radius for the perturbation (<= 0 disables the projection)
        self.ernie_epsilon = float(args.get("ernie_epsilon", 0.1))
        # if True, perturbation magnitudes are relative to |o| (the released-code
        # ``* torch.abs(obs)`` scaling); if False they are absolute in obs units.
        self.ernie_relative = bool(args.get("ernie_relative", True))
        # gradient-ascent direction normalization: "l2" | "sign" | "raw".
        # "raw" reproduces the exact released-code step (un-normalized gradient).
        self.ernie_perturb_norm = str(args.get("ernie_perturb_norm", "l2"))
        # std of the small Gaussian noise used to initialize delta
        self.ernie_init_std = float(args.get("ernie_init_std", 1e-3))
        # use the Stackelberg (unrolled / differentiate-through-the-attack) variant
        self.ernie_stackelberg = bool(args.get("ernie_stackelberg", False))

        self.action_type = act_space.__class__.__name__

        # diagnostics (filled in every update, averaged by ``train``)
        self.last_ernie_reg = 0.0
        self.last_ernie_adv_kl = 0.0
        self.last_ernie_delta_norm = 0.0

    # ======================================================================
    # policy-distribution helpers
    # ======================================================================
    def _policy_dist(self, obs):
        """Return the current policy's action distribution at ``obs``.

        ``obs`` must be a torch tensor on ``self.device``.  Gradients flow through
        the actor parameters (and through ``obs`` if it requires grad).
        """
        if self.actor.use_recurrent_policy or self.actor.use_naive_recurrent_policy:
            raise NotImplementedError(
                "ERNIE adversarial regularization is implemented for non-recurrent "
                "actors (use_recurrent_policy=False), matching the mamujoco Ant config."
            )
        if self.actor.act.multidiscrete_action:
            raise NotImplementedError(
                "ERNIE regularization supports Box / Discrete action spaces."
            )
        features = self.actor.base(obs)
        return self.actor.act.action_out(features)

    def _policy_kl(self, dist_p, dist_q):
        """KL( dist_p || dist_q ) per sample, summed over action dimensions.

        For a diagonal-Gaussian (Box) policy ``kl_divergence`` returns one value
        per action dimension, so we sum over the last axis; for a Categorical
        (Discrete) policy it already returns one value per sample.
        """
        kl = torch.distributions.kl.kl_divergence(dist_p, dist_q)
        if self.action_type == "Box":
            kl = kl.sum(-1)
        return kl  # shape [batch]

    # ======================================================================
    # ERNIE adversarial regularizer
    # ======================================================================
    def _ernie_regularizer(self, obs_batch):
        """Compute the ERNIE adversarial regularizer for a minibatch of obs.

        Solves the inner maximization ``max_{||delta||<=eps} KL(pi(o+delta), pi(o))``
        with projected gradient ascent, then returns
        ``R = mean_o KL( pi(o + delta*), pi(o) )`` and a diagnostics dict.

        Returns:
            reg: (torch.Tensor) scalar regularizer (carries grad w.r.t. theta).
            diag: (dict) scalar diagnostics for logging.
        """
        o = check(obs_batch).to(**self.tpdv).detach()

        # per-dimension scale for the (optionally relative) perturbation
        scale = o.abs() if self.ernie_relative else torch.ones_like(o)
        eps_box = self.ernie_epsilon * scale  # epsilon-ball radius per dimension

        # ---- initialize the perturbation with small Gaussian noise ----
        if self.ernie_init_std > 0:
            delta = torch.randn_like(o) * self.ernie_init_std * scale
        else:
            delta = torch.zeros_like(o)
        if self.ernie_epsilon > 0:
            delta = torch.min(torch.max(delta, -eps_box), eps_box)
        delta = delta.detach().requires_grad_(True)

        create_graph = self.ernie_stackelberg

        # ---- projected gradient ascent to find the worst-case delta ----
        for _ in range(self.ernie_perturb_steps):
            dist_pert = self._policy_dist(o + delta)
            dist_base = self._policy_dist(o)
            attack_obj = self._policy_kl(dist_pert, dist_base).mean()

            grad = torch.autograd.grad(
                attack_obj,
                delta,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]

            # ascent direction
            if self.ernie_perturb_norm == "sign":
                direction = grad.sign()
            elif self.ernie_perturb_norm == "l2":
                gnorm = (
                    grad.flatten(1)
                    .norm(dim=1)
                    .clamp_min(1e-12)
                    .view(-1, *([1] * (grad.dim() - 1)))
                )
                direction = grad / gnorm
            else:  # "raw" -- exact released-code step (un-normalized gradient)
                direction = grad

            delta = delta + self.ernie_perturb_alpha * direction * scale

            # project back into the epsilon-ball (l_inf, per dimension)
            if self.ernie_epsilon > 0:
                delta = torch.min(torch.max(delta, -eps_box), eps_box)

            if not create_graph:
                # decouple successive attack steps from theta (simple variant)
                delta = delta.detach().requires_grad_(True)

        if not create_graph:
            delta = delta.detach()

        # ---- regularizer: discrepancy at the worst-case perturbation ----
        dist_pert = self._policy_dist(o + delta)
        dist_base = self._policy_dist(o)
        adv_kl = self._policy_kl(dist_pert, dist_base)  # [batch]
        reg = adv_kl.mean()

        diag = {
            "ernie_adv_kl": float(adv_kl.mean().detach().item()),
            "ernie_delta_norm": float(delta.detach().abs().mean().item()),
        }
        return reg, diag

    # ======================================================================
    # actor update = HAPPO clipped surrogate + ERNIE adversarial regularizer
    # ======================================================================
    def update(self, sample):
        """Update actor network (HAPPO update + ERNIE adversarial regularizer).

        Args:
            sample: (Tuple) minibatch of training data.
        Returns:
            policy_loss, dist_entropy, actor_grad_norm, imp_weights, ernie_diag
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        # ---- standard HAPPO clipped surrogate ----
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss

        # ---- ERNIE adversarial regularizer ----
        if self.ernie_lambda > 0 and self.ernie_perturb_steps > 0:
            ernie_reg, ernie_diag = self._ernie_regularizer(obs_batch)
        else:  # fallback: pure HAPPO
            ernie_reg = torch.zeros((), device=self.device)
            ernie_diag = {"ernie_adv_kl": 0.0, "ernie_delta_norm": 0.0}

        self.last_ernie_reg = float(ernie_reg.detach().item())
        self.last_ernie_adv_kl = ernie_diag["ernie_adv_kl"]
        self.last_ernie_delta_norm = ernie_diag["ernie_delta_norm"]

        self.actor_optimizer.zero_grad()

        total_loss = (
            policy_loss
            - dist_entropy * self.entropy_coef
            + self.ernie_lambda * ernie_reg
        )
        total_loss.backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        ernie_diag["ernie_reg_loss"] = self.last_ernie_reg
        return policy_loss, dist_entropy, actor_grad_norm, imp_weights, ernie_diag

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD (HAPPO + ERNIE logging).

        Identical to HAPPO.train but unpacks the extra ERNIE diagnostics returned
        by ``update`` and averages them into ``train_info`` so they are logged.
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0
        train_info["ernie_reg_loss"] = 0
        train_info["ernie_adv_kl"] = 0
        train_info["ernie_delta_norm"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
                (
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                    ernie_diag,
                ) = self.update(sample)

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()
                train_info["ernie_reg_loss"] += ernie_diag["ernie_reg_loss"]
                train_info["ernie_adv_kl"] += ernie_diag["ernie_adv_kl"]
                train_info["ernie_delta_norm"] += ernie_diag["ernie_delta_norm"]

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
