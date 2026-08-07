"""LCPO algorithm for HARL.

LCPO (Locally Constrained Policy Optimization, ICLR 2025) is an online RL method
for non-stationary, context-driven environments. Its central idea is to optimize
the policy on the *current* data while constraining the policy to remain
*unchanged on out-of-distribution (past-context) states* -- this prevents
catastrophic forgetting when the environment drifts to a new regime and later
returns to an old one.

This class adapts the authors' PPO instantiation (``LCPPO``,
``core_lcppo.py``) onto HARL's HAPPO:

* The in-distribution trust region is provided by HAPPO's PPO clip (unchanged).
* The out-of-distribution constraint is added as a soft penalty
  ``kappa * mean_{s in OOD} KL(pi_old(.|s) || pi_new(.|s))`` to the actor loss,
  evaluated on anchor states drawn from a per-agent OOD buffer.
* When no OOD anchor states are available (early training), the penalty is zero
  and the update is exactly HAPPO (the LCPO fallback to plain PPO).

Everything except the actor update is untouched: observations, the centralized
critic, and the HAPPO sequential-update / ``factor`` machinery all stay as-is.
The runner (``OnPolicyLcpoRunner``) owns the OOD buffers and calls ``set_ood`` /
``clear_ood`` around training.
"""

import torch
import torch.nn as nn

from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.happo import HAPPO


class LCPO(HAPPO):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize LCPO algorithm.

        Args:
            args: (dict) merged ``{**model, **algo}`` arguments.
            obs_space: (gym.spaces) observation space of the agent.
            act_space: (gym.spaces) action space of the agent.
            device: (torch.device) device for tensor operations.
        """
        super(LCPO, self).__init__(args, obs_space, act_space, device)

        self.lcpo_kappa = float(args.get("lcpo_kappa", 1.0))
        # max number of OOD anchor states used per minibatch KL evaluation
        self.lcpo_ood_minibatch = int(args.get("lcpo_ood_minibatch", 1024))
        self.action_type = act_space.__class__.__name__

        # OOD anchor state buffers (set by the runner each training iteration)
        self.ood_obs = None
        self.ood_mean_old = None
        self.ood_std_old = None
        self.ood_logprob_old = None

        # ---- diagnostics (for tuning lcpo_kappa / lcpo_ood_thresh) ----
        self.last_ood_kl = 0.0        # KL on the most recent minibatch
        self.last_ood_n = 0           # number of OOD anchors currently set
        self.last_ood_kl_avg = 0.0    # KL averaged over the last train() call
        self.last_ood_pen_avg = 0.0   # kappa*KL averaged over the last train() call
        self.last_pol_loss_avg = 0.0  # |policy loss| averaged over the last train()
        self.last_ood_ratio = 0.0     # penalty / |policy loss| (how much LCPO bites)
        # per-train() accumulators (reset at the top of ``train``)
        self._ood_kl_sum = 0.0
        self._ood_pen_sum = 0.0
        self._pol_loss_sum = 0.0
        self._n_ood_updates = 0

    # ======================================================================
    # OOD anchor management
    # ======================================================================
    def _actor_features(self, obs):
        """Run the actor trunk on obs (assumes a non-recurrent MLP actor)."""
        if self.actor.use_naive_recurrent_policy or self.actor.use_recurrent_policy:
            raise NotImplementedError(
                "LCPO OOD anchoring is implemented for non-recurrent actors "
                "(use_recurrent_policy=False)."
            )
        return self.actor.base(obs)

    def _ood_distribution(self, obs):
        """Return the action distribution of the current policy on obs."""
        features = self._actor_features(obs)
        return self.actor.act.action_out(features)

    @torch.no_grad()
    def set_ood(self, ood_obs_np):
        """Store OOD anchor states and snapshot the current (old) policy on them.

        Args:
            ood_obs_np: (np.ndarray) (M, obs_dim) out-of-distribution anchor states.
        """
        if ood_obs_np is None or len(ood_obs_np) == 0:
            self.clear_ood()
            return
        self.ood_obs = torch.as_tensor(
            ood_obs_np, dtype=torch.float32, device=self.device
        )
        self.last_ood_n = int(self.ood_obs.shape[0])
        dist = self._ood_distribution(self.ood_obs)
        if self.action_type == "Box":
            self.ood_mean_old = dist.mean.detach()
            self.ood_std_old = dist.stddev.expand_as(dist.mean).detach()
        else:
            self.ood_logprob_old = torch.log_softmax(dist.logits, dim=-1).detach()

    def clear_ood(self):
        """Forget the current OOD anchors."""
        self.ood_obs = None
        self.ood_mean_old = None
        self.ood_std_old = None
        self.ood_logprob_old = None
        self.last_ood_n = 0

    def _ood_kl(self):
        """KL(pi_old || pi_new) averaged over a minibatch of OOD anchor states."""
        if self.ood_obs is None or self.ood_obs.shape[0] == 0:
            return torch.zeros((), device=self.device)

        n = self.ood_obs.shape[0]
        if n > self.lcpo_ood_minibatch:
            idx = torch.randint(
                0, n, (self.lcpo_ood_minibatch,), device=self.device
            )
            obs = self.ood_obs[idx]
        else:
            idx = None
            obs = self.ood_obs

        dist_new = self._ood_distribution(obs)

        if self.action_type == "Box":
            mu_old = self.ood_mean_old if idx is None else self.ood_mean_old[idx]
            std_old = self.ood_std_old if idx is None else self.ood_std_old[idx]
            mu_new = dist_new.mean
            std_new = dist_new.stddev.expand_as(mu_new)
            var_old = std_old.pow(2)
            var_new = std_new.pow(2)
            # KL(N_old || N_new) for diagonal Gaussians, summed over action dims
            kl = (
                torch.log(std_new)
                - torch.log(std_old)
                + (var_old + (mu_old - mu_new).pow(2)) / (2 * var_new)
                - 0.5
            ).sum(-1)
            return kl.mean()
        else:
            logp_old = (
                self.ood_logprob_old if idx is None else self.ood_logprob_old[idx]
            )
            logp_new = torch.log_softmax(dist_new.logits, dim=-1)
            p_old = logp_old.exp()
            kl = (p_old * (logp_old - logp_new)).sum(-1)
            return kl.mean()

    # ======================================================================
    # actor update = HAPPO update + LCPO out-of-distribution KL penalty
    # ======================================================================
    def update(self, sample):
        """Update actor network (HAPPO clipped surrogate + LCPO OOD anchoring).

        Args:
            sample: (Tuple) minibatch of training data.
        Returns:
            policy_loss, dist_entropy, actor_grad_norm, imp_weights
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

        # ---- LCPO local constraint on out-of-distribution states ----
        ood_kl = self._ood_kl()
        ood_penalty = self.lcpo_kappa * ood_kl
        self.last_ood_kl = float(ood_kl.detach().item())

        # accumulate diagnostics over this train() call (averaged in ``train``)
        self._ood_kl_sum += self.last_ood_kl
        self._ood_pen_sum += float(ood_penalty.detach().item())
        self._pol_loss_sum += abs(float(policy_loss.detach().item()))
        self._n_ood_updates += 1

        self.actor_optimizer.zero_grad()

        total_loss = (
            policy_loss
            - dist_entropy * self.entropy_coef
            + ood_penalty
        )
        total_loss.backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type):
        """HAPPO minibatch training, plus averaged LCPO diagnostics.

        Resets the per-call accumulators, runs the standard HAPPO update loop
        (whose ``update`` is this class's OOD-anchored version), then folds the
        averaged OOD KL / penalty / penalty-to-policy-loss ratio into ``train_info``
        so they are written to tensorboard as ``agent%i/ood_*`` and are available
        to the runner for console logging.
        """
        self._ood_kl_sum = 0.0
        self._ood_pen_sum = 0.0
        self._pol_loss_sum = 0.0
        self._n_ood_updates = 0

        train_info = super(LCPO, self).train(actor_buffer, advantages, state_type)

        n = max(self._n_ood_updates, 1)
        self.last_ood_kl_avg = self._ood_kl_sum / n
        self.last_ood_pen_avg = self._ood_pen_sum / n
        self.last_pol_loss_avg = self._pol_loss_sum / n
        self.last_ood_ratio = self.last_ood_pen_avg / (self.last_pol_loss_avg + 1e-8)

        train_info["ood_n_anchors"] = float(self.last_ood_n)
        train_info["ood_kl"] = self.last_ood_kl_avg
        train_info["ood_penalty"] = self.last_ood_pen_avg
        train_info["ood_kl_ratio"] = self.last_ood_ratio
        return train_info
