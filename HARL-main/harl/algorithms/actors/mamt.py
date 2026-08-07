"""MAMT algorithm for HARL.

MAMT (Multi-Agent trust-region decomposition; "Dealing with Non-Stationarity in
MARL via Trust Region Decomposition", arXiv:2102.10616) handles non-stationarity by
constraining the divergence between consecutive joint policies (delta-stationarity).
It decomposes the joint trust region into **adaptive per-agent local trust regions**
whose sizes are learned end-to-end (via a message-passing TRD-Net that estimates the
joint policy divergence), and updates each agent's policy with a mirror-descent step
using a Tsallis-KL trust-region term relative to an old policy.

This class is the actor side of the HARL adaptation (HAPPO backbone, continuous
Gaussian policy):

* it keeps an ``old_actor`` snapshot (the trust-region reference, refreshed
  periodically by the runner) and a learnable scalar ``local_tr``;
* its ``update`` is HAPPO's clipped surrogate PLUS an adaptive Tsallis-KL penalty
  ``tr_scale * KL_q(pi || pi_old) / local_tr`` -- the per-agent local trust region.

The TRD-Net, coordination coefficients, teammate-modeling non-stationarity
measurement, and the adaptive ``local_tr`` update are orchestrated by
``OnPolicyMamtRunner``; this class exposes the per-call primitives it needs.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.happo import HAPPO
from harl.models.mamt.mamt_modules import tsallis_log_q


class MAMT(HAPPO):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(MAMT, self).__init__(args, obs_space, act_space, device)

        self.tsallis_q = float(args.get("mamt_tsallis_q", 0.5))
        self.tr_scale = float(args.get("mamt_tr_scale", 10.0))
        self.local_tr_init = float(args.get("mamt_local_tr_init", 0.01))
        self.local_tr_min = float(args.get("mamt_local_tr_min", 1e-2))
        self.local_tr_max = float(args.get("mamt_local_tr_max", 1e2))
        self.action_type = act_space.__class__.__name__

        # old policy snapshot = the trust-region reference (refreshed by the runner)
        self.old_actor = deepcopy(self.actor)
        for p in self.old_actor.parameters():
            p.requires_grad = False

        # learnable per-agent local trust region (the runner owns the optimizer)
        self.local_tr = torch.tensor(
            self.local_tr_init, dtype=torch.float32, device=device, requires_grad=True
        )

        self.last_tr_term = torch.zeros((), device=device)  # for the runner's f_loss

    # ======================================================================
    # old-policy snapshot
    # ======================================================================
    @torch.no_grad()
    def snapshot_old(self):
        """Refresh the trust-region reference (old policy)."""
        for tp, p in zip(self.old_actor.parameters(), self.actor.parameters()):
            tp.data.copy_(p.data)

    def clamp_local_tr(self):
        with torch.no_grad():
            self.local_tr.data.clamp_(self.local_tr_min, self.local_tr_max)

    # ======================================================================
    # actor update = HAPPO clipped surrogate + adaptive Tsallis-KL trust region
    # ======================================================================
    def update(self, sample):
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

        # ---- MAMT adaptive trust-region: divergence to the old policy ----
        # The consecutive-policy divergence MAMT bounds is KL(pi_new || pi_old):
        # closed-form Gaussian KL for a continuous Box policy, categorical KL for a
        # Discrete (SMAC/SMACv2) policy. With q!=1 the Tsallis q-logarithm reweights
        # it. (MAMT's raw q-log of probabilities is unbounded for Gaussian
        # densities, so the KL form is the faithful continuous analog.)
        new_repr = self._policy_repr(self.actor, obs_batch)
        with torch.no_grad():
            old_repr = self._policy_repr(self.old_actor, obs_batch)
        kl_new_old = self._policy_kl(new_repr, old_repr)  # (B,)
        if self.tsallis_q == 1.0:
            tr_term = kl_new_old.mean()
        else:
            tr_term = tsallis_log_q(1.0 + kl_new_old, self.tsallis_q).mean()
        # local_tr is updated separately (f_loss); detach it from the policy update
        tr_penalty = self.tr_scale * tr_term / self.local_tr.detach().clamp(
            min=self.local_tr_min
        )
        self.last_tr_term = tr_term.detach()

        self.actor_optimizer.zero_grad()
        total_loss = policy_action_loss - dist_entropy * self.entropy_coef + tr_penalty
        total_loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())
        self.actor_optimizer.step()

        return policy_action_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type):
        """HAPPO train loop with MAMT trust-region logging."""
        train_info = {
            "policy_loss": 0,
            "dist_entropy": 0,
            "actor_grad_norm": 0,
            "ratio": 0,
            "mamt_tr_term": 0,
            "mamt_local_tr": float(self.local_tr.detach().item()),
        }

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
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    sample
                )
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()
                train_info["mamt_tr_term"] += float(self.last_tr_term.item())

        num_updates = self.ppo_epoch * self.actor_num_mini_batch
        for k in ["policy_loss", "dist_entropy", "actor_grad_norm", "ratio", "mamt_tr_term"]:
            train_info[k] /= num_updates
        train_info["mamt_local_tr"] = float(self.local_tr.detach().item())
        return train_info

    # ======================================================================
    # policy-distribution helpers (used by the runner for ccs / d_ns / TRD-Net)
    # ======================================================================
    def _policy_repr(self, actor, obs):
        """Light representation of a (non-recurrent) actor's policy at ``obs``.

        Returns a tuple of distribution parameters: ``(mean, std)`` for a
        continuous Box policy, or ``(logprobs,)`` for a Discrete (SMAC/SMACv2)
        policy (raw, unmasked logits, so the divergence measures the change of the
        policy function itself -- consistent with LCPO's OOD-KL treatment).
        """
        if actor.use_recurrent_policy or actor.use_naive_recurrent_policy:
            raise NotImplementedError(
                "MAMT trust-region / teammate modeling is implemented for "
                "non-recurrent actors."
            )
        obs = check(obs).to(**self.tpdv)
        dist = actor.act.action_out(actor.base(obs))
        if self.action_type == "Box":
            return (dist.mean, dist.stddev.expand_as(dist.mean))
        return (torch.log_softmax(dist.logits, dim=-1),)

    def _policy_kl(self, repr_p, repr_q):
        """KL(p || q) between two policy representations (dispatch on action type)."""
        if self.action_type == "Box":
            return self._gaussian_kl(repr_p[0], repr_p[1], repr_q[0], repr_q[1])
        logp, logq = repr_p[0], repr_q[0]
        return (logp.exp() * (logp - logq)).sum(-1)

    @staticmethod
    def _gaussian_kl(mu_p, std_p, mu_q, std_q):
        """KL( N(mu_p, std_p) || N(mu_q, std_q) ), summed over action dims -> (B,)."""
        var_p, var_q = std_p.pow(2), std_q.pow(2)
        return (
            torch.log(std_q)
            - torch.log(std_p)
            + (var_p + (mu_p - mu_q).pow(2)) / (2.0 * var_q)
            - 0.5
        ).sum(-1)

    def teammate_repr(self, obs):
        """Policy representation of this actor (used by the runner's teammate KL)."""
        return self._policy_repr(self.actor, obs)
