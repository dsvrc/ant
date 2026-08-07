"""Runner for FSAC (Forecaster-SAC) on top of HARL's off-policy HA stack.

FSAC is HASAC with one change: the policy-improvement step uses the *forecasted
future Q* produced by ``FSACCritic.get_forecasted_values`` instead of the current
Q. This runner therefore reuses HASAC's critic/actor training verbatim, and only:

* captures a frozen critic snapshot every ``fsac_snapshot_interval`` updates,
* swaps ``critic.get_values`` -> ``critic.get_forecasted_values`` in the actor
  loss, and
* (optionally) gates the policy update to a fraction ``fsac_update_ratio`` of each
  forecast cycle (the ``updateratio`` knob from the reference), so the critic can
  keep tracking the drift while the policy is held fixed.

``get_actions`` is overridden to use HASAC's stochastic-action path (the base
off-policy runner only takes that path when ``algo == "hasac"``).
"""

import numpy as np
import torch

from harl.utils.trans_tools import _t2n
from harl.runners.off_policy_ha_runner import OffPolicyHARunner


class OffPolicyFsacRunner(OffPolicyHARunner):
    """Runner for the FSAC algorithm."""

    def __init__(self, args, algo_args, env_args):
        super(OffPolicyFsacRunner, self).__init__(args, algo_args, env_args)
        algo_cfg = algo_args["algo"]
        self.fsac_past_length = int(algo_cfg.get("fsac_past_length", 10))
        self.fsac_future_length = int(algo_cfg.get("fsac_future_length", 10))
        self.fsac_snapshot_interval = int(algo_cfg.get("fsac_snapshot_interval", 1000))
        self.fsac_update_ratio = float(algo_cfg.get("fsac_update_ratio", 1.0))

    @torch.no_grad()
    def get_actions(self, obs, available_actions=None, add_random=True):
        """Get actions for rollout using HASAC's stochastic-policy path."""
        actions = []
        for agent_id in range(self.num_agents):
            if len(np.array(available_actions).shape) == 3:
                actions.append(
                    _t2n(
                        self.actor[agent_id].get_actions(
                            obs[:, agent_id],
                            available_actions[:, agent_id],
                            add_random,
                        )
                    )
                )
            else:
                actions.append(
                    _t2n(
                        self.actor[agent_id].get_actions(
                            obs[:, agent_id], stochastic=add_random
                        )
                    )
                )
        return np.array(actions).transpose(1, 0, 2)

    def train(self):
        """Train FSAC: HASAC critic update + forecasted-Q policy improvement."""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,
            sp_obs,
            sp_actions,
            sp_available_actions,
            sp_reward,
            sp_done,
            sp_valid_transition,
            sp_term,
            sp_next_share_obs,
            sp_next_obs,
            sp_next_available_actions,
            sp_gamma,
        ) = data

        # ---- train critic (identical to HASAC) ----
        self.critic.turn_on_grad()
        next_actions = []
        next_logp_actions = []
        for agent_id in range(self.num_agents):
            next_action, next_logp_action = self.actor[
                agent_id
            ].get_actions_with_logprobs(
                sp_next_obs[agent_id],
                sp_next_available_actions[agent_id]
                if sp_next_available_actions is not None
                else None,
            )
            next_actions.append(next_action)
            next_logp_actions.append(next_logp_action)
        self.critic.train(
            sp_share_obs,
            sp_actions,
            sp_reward,
            sp_done,
            sp_valid_transition,
            sp_term,
            sp_next_share_obs,
            next_actions,
            next_logp_actions,
            sp_gamma,
            self.value_normalizer,
        )
        self.critic.turn_off_grad()

        # ---- FSAC: capture a frozen critic snapshot periodically ----
        self.critic.maybe_snapshot(self.total_it)

        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device)

        if self.total_it % self.policy_freq == 0:
            # update-ratio gate: only improve the policy during a fraction of each
            # forecast cycle (critic keeps tracking the drift during the off-phase).
            cycle = max(1, self.fsac_future_length * self.fsac_snapshot_interval)
            do_actor = (self.total_it % cycle) < cycle * self.fsac_update_ratio

            if do_actor:
                actions = []
                logp_actions = []
                with torch.no_grad():
                    for agent_id in range(self.num_agents):
                        action, logp_action = self.actor[
                            agent_id
                        ].get_actions_with_logprobs(
                            sp_obs[agent_id],
                            sp_available_actions[agent_id]
                            if sp_available_actions is not None
                            else None,
                        )
                        actions.append(action)
                        logp_actions.append(logp_action)

                if self.fixed_order:
                    agent_order = list(range(self.num_agents))
                else:
                    agent_order = list(np.random.permutation(self.num_agents))
                for agent_id in agent_order:
                    self.actor[agent_id].turn_on_grad()
                    # refresh this agent's action with grad
                    actions[agent_id], logp_actions[agent_id] = self.actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )
                    if self.state_type == "EP":
                        logp_action = logp_actions[agent_id]
                        actions_t = torch.cat(actions, dim=-1)
                    elif self.state_type == "FP":
                        logp_action = torch.tile(
                            logp_actions[agent_id], (self.num_agents, 1)
                        )
                        actions_t = torch.tile(
                            torch.cat(actions, dim=-1), (self.num_agents, 1)
                        )
                    # *** FSAC: forecasted future Q drives policy improvement ***
                    value_pred = self.critic.get_forecasted_values(
                        sp_share_obs, actions_t
                    )
                    if self.algo_args["algo"]["use_policy_active_masks"]:
                        if self.state_type == "EP":
                            actor_loss = (
                                -torch.sum(
                                    (value_pred - self.alpha[agent_id] * logp_action)
                                    * sp_valid_transition[agent_id]
                                )
                                / sp_valid_transition[agent_id].sum()
                            )
                        elif self.state_type == "FP":
                            valid_transition = torch.tile(
                                sp_valid_transition[agent_id], (self.num_agents, 1)
                            )
                            actor_loss = (
                                -torch.sum(
                                    (value_pred - self.alpha[agent_id] * logp_action)
                                    * valid_transition
                                )
                                / valid_transition.sum()
                            )
                    else:
                        actor_loss = -torch.mean(
                            value_pred - self.alpha[agent_id] * logp_action
                        )
                    self.actor[agent_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor[agent_id].actor_optimizer.step()
                    self.actor[agent_id].turn_off_grad()
                    # train this agent's alpha
                    if self.algo_args["algo"]["auto_alpha"]:
                        log_prob = (
                            logp_actions[agent_id].detach()
                            + self.target_entropy[agent_id]
                        )
                        alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                        self.alpha_optimizer[agent_id].zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer[agent_id].step()
                        self.alpha[agent_id] = torch.exp(
                            self.log_alpha[agent_id].detach()
                        )
                    actions[agent_id], _ = self.actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )
                # train critic's alpha
                if self.algo_args["algo"]["auto_alpha"]:
                    self.critic.update_alpha(
                        logp_actions, np.sum(self.target_entropy)
                    )

            # soft update the critic target (kept on the policy_freq cadence)
            self.critic.soft_update()
