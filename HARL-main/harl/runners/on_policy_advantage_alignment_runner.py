"""Runner for Advantage Alignment (HAPPO + advantage-aligned policy gradient).

Advantage Alignment (Duque et al., arXiv:2406.14662) shapes each agent's
advantage with an opponent-shaping term built from *all* agents' advantages
(paper Eq. 10):

    A*_t^i = A_t^i + beta * ( sum_{k<t} gamma^{t-k} A_k^i ) * ( sum_{j != i} A_t^j )

The cross-agent factor ``sum_{j != i} A_t^j`` requires every agent's advantage at
once, so this runner computes the shaped advantage ``A*`` for all agents up-front
(advantages are produced from the centralized critic exactly as in HAPPO), then
runs HAPPO's standard sequential per-agent update, handing each ``AdvantageAlignment``
actor its own ``A*`` slice. The actor consumes ``A*`` directly in the clipped PPO
surrogate (Eq. 9) and does NOT re-normalize it.

Two forms of the alignment term are supported:

* ``aa_discounted: false`` (default; the released code's ``integrated_aa``):
  ``A_1s = cumsum_t(A)``  and the term is divided by ``t`` (time average).
* ``aa_discounted: true`` (the paper's exact Eq. 8/10): ``A_1s`` is the
  gamma-discounted running sum and the term carries the extra ``gamma`` factor.

Everything else (critic, observations, buffers, factor machinery) is unchanged
HAPPO, so Advantage Alignment reduces to HAPPO when ``aa_weight == 0``.
"""

import numpy as np
import torch

from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_ha_runner import OnPolicyHARunner


class OnPolicyAdvAlignRunner(OnPolicyHARunner):
    """Runner for the Advantage Alignment algorithm."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyAdvAlignRunner, self).__init__(args, algo_args, env_args)

        cfg = algo_args["algo"]
        self.aa_weight = float(cfg.get("aa_weight", 1.0))
        self.aa_discounted = bool(cfg.get("aa_discounted", False))
        self.aa_gamma = float(cfg.get("gamma", 0.99))
        self.aa_normalize_advantages = bool(cfg.get("aa_normalize_advantages", True))
        self.aa_center_rewards = bool(cfg.get("aa_center_rewards", False))
        self._aa_steps = 0

    # ======================================================================
    # advantage-alignment shaping
    # ======================================================================
    def _aa_shape(self, A_norm):
        """Compute the AA-shaped advantage A* from normalized per-agent advantages.

        Args:
            A_norm: (np.ndarray) shape (T, B, N) normalized advantages.
        Returns:
            A_star: (np.ndarray) shape (T, B, N) advantage-aligned advantages,
            aa_term_abs_mean: (float) mean |alignment term| (diagnostics).
        """
        A = torch.as_tensor(A_norm, dtype=torch.float32)
        T, B, N = A.shape
        gamma = self.aa_gamma

        # reshape (T, B, N) -> (B*N, T) [row index = b*N + i], matching the paper code
        A_bnt = A.permute(1, 2, 0).reshape(B * N, T)
        A_cur = A_bnt[:, 1:]  # advantages at t = 1..T-1, shape (B*N, T-1)

        # cumulative past own advantage  sum_{k<t} (gamma^{t-k}) A_k
        if self.aa_discounted:
            A_prev = A_bnt[:, :-1]  # A_0..A_{T-2}
            A_1s = torch.zeros_like(A_prev)
            running = torch.zeros(B * N, dtype=A.dtype)
            for p in range(A_prev.shape[1]):
                running = gamma * (A_prev[:, p] + running)
                A_1s[:, p] = running
        else:
            A_1s = torch.cumsum(A_bnt[:, :-1], dim=1)

        # sum of OTHER agents' current advantage  sum_{j != i} A_t^j
        mask = torch.ones((N, N), dtype=A.dtype) - torch.eye(N, dtype=A.dtype)
        A_cur_bnt = A_cur.reshape(B, N, -1)  # (B, N, T-1)
        A_2s = torch.einsum("ij,bjt->bit", mask, A_cur_bnt).reshape(B * N, -1)

        if self.aa_discounted:
            aa = gamma * A_1s * A_2s  # Eq. 10: beta * gamma * (...) * (...)
        else:
            denom = torch.arange(1, A_cur.shape[1] + 1, dtype=A.dtype)
            aa = (A_1s * A_2s) / denom

        # AA term is undefined at t=0 -> pad with zero so shapes match A
        aa = torch.cat([torch.zeros((B * N, 1), dtype=A.dtype), aa], dim=1)
        A_star = A_bnt + self.aa_weight * aa

        aa_term_abs_mean = float((self.aa_weight * aa).abs().mean().item())
        A_star = A_star.reshape(B, N, T).permute(2, 0, 1).contiguous()
        return A_star.numpy().astype(np.float32), aa_term_abs_mean

    def _compute_aligned_advantages(self):
        """Compute per-agent advantages, normalize, and AA-shape them.

        Returns:
            A_star: (np.ndarray) shape (T, B, N) advantage-aligned advantages.
            aa_term_abs_mean: (float) diagnostics.
        """
        # raw advantages from the centralized critic (identical to HAPPO)
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        # build per-agent advantages A: (T, B, N)
        if self.state_type == "EP":
            # shared critic -> advantages (T, B, 1) repeated across agents
            A = np.repeat(advantages, self.num_agents, axis=2)
        else:  # FP: per-agent critic -> (T, B, N, 1)
            A = advantages[..., 0]

        # active masks (T, B, N) for masked normalization
        active = np.stack(
            [self.actor_buffer[i].active_masks[:-1] for i in range(self.num_agents)],
            axis=2,
        )[..., 0]

        if self.aa_normalize_advantages:
            A_copy = A.copy()
            A_copy[active == 0.0] = np.nan
            mean_a = np.nanmean(A_copy)
            std_a = np.nanstd(A_copy)
            A = (A - mean_a) / (std_a + 1e-5)

        return self._aa_shape(A)

    # ======================================================================
    # train: HAPPO sequential update on advantage-aligned advantages
    # ======================================================================
    def train(self):
        """Advantage-aligned HAPPO update (mirrors OnPolicyHARunner.train)."""
        actor_train_infos = []

        # factor for HAPPO's sequential update
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # advantage-aligned advantages A*: (T, B, N)
        A_star, aa_term_abs_mean = self._compute_aligned_advantages()

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())

        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(factor)

            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # agent's advantage-aligned advantage, shape (T, B, 1)
            agent_adv = A_star[:, :, agent_id : agent_id + 1].copy()

            actor_train_info = self.actor[agent_id].train(
                self.actor_buffer[agent_id], agent_adv, self.state_type
            )

            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            factor = factor * _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            actor_train_infos.append(actor_train_info)

        # critic update (unchanged HAPPO)
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        # diagnostics
        self._aa_steps += (
            self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        if self.writter is not None:
            self.writter.add_scalar(
                "advantage_alignment/aa_term_abs_mean", aa_term_abs_mean, self._aa_steps
            )
            self.writter.add_scalar(
                "advantage_alignment/A_star_abs_mean",
                float(np.abs(A_star).mean()),
                self._aa_steps,
            )

        return actor_train_infos, critic_train_info
