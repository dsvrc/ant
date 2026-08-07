"""FP critic buffer with the COMARL distributionally-robust Bellman operator.

Feature-Pruned (FP) analogue of ``robust_critic_buffer_ep.RobustCriticBufferEP``
used for SMAC / SMACv2 (whose ``state_type`` is FP). Only ``compute_returns`` is
overridden: every GAE bootstrap ``gamma * V(s') * mask`` becomes
``gamma * sigma_rho(V(s')) * mask`` where ``sigma_rho`` is the COMARL robust
operator (``contamination`` or ``tv``; see
``harl.models.comarl.comarl_modules.robust_next_value``).

The only difference from the EP variant is the buffer shapes: value_preds /
returns / g_values carry an extra ``num_agents`` axis (they are per-agent global
states in FP). ``robust_next_value`` is elementwise, so the return-computation
logic is identical to the EP version and to the base FP buffer; ``g_values[step]``
pairs with the bootstrap value ``V(s_{step+1})`` exactly as in the reference.

With ``rho = 0`` this reduces exactly to the standard FP return computation.
"""

from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.models.comarl.comarl_modules import robust_next_value


class RobustCriticBufferFP(OnPolicyCriticBufferFP):
    """FP critic buffer whose returns use the COMARL robust Bellman operator."""

    def _robust(self, v_next, step):
        """Robust transform of the bootstrap value V(s_{step+1})."""
        g = None
        if self._comarl_mode == "tv" and self._comarl_g is not None:
            g = self._comarl_g[step]
        return robust_next_value(v_next, self._comarl_rho, self._comarl_mode, g)

    def compute_returns(
        self,
        next_value,
        value_normalizer=None,
        rho=0.0,
        mode="contamination",
        g_values=None,
    ):
        """Compute robust returns (GAE or discounted), robustifying the bootstrap.

        Extra args (vs the base class):
            rho: (float) robustness radius.
            mode: (str) "contamination" or "tv".
            g_values: (np.ndarray|None) G-network output per stored state,
                shape (episode_length + 1, n_rollout_threads, num_agents, 1),
                true value scale.
        """
        self._comarl_rho = rho
        self._comarl_mode = mode
        self._comarl_g = g_values

        if self.use_proper_time_limits:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:
                        v_next = value_normalizer.denormalize(self.value_preds[step + 1])
                        v_cur = value_normalizer.denormalize(self.value_preds[step])
                        delta = (
                            self.rewards[step]
                            + self.gamma * self._robust(v_next, step) * self.masks[step + 1]
                            - v_cur
                        )
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = self.bad_masks[step + 1] * gae
                        self.returns[step] = gae + v_cur
                    else:
                        v_next = self.value_preds[step + 1]
                        delta = (
                            self.rewards[step]
                            + self.gamma * self._robust(v_next, step) * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = self.bad_masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:
                        boot = self._robust(
                            value_normalizer.denormalize(self.value_preds[step]), step
                        )
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * boot
                    else:
                        boot = self._robust(self.value_preds[step], step)
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * boot
        else:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:
                        v_next = value_normalizer.denormalize(self.value_preds[step + 1])
                        v_cur = value_normalizer.denormalize(self.value_preds[step])
                        delta = (
                            self.rewards[step]
                            + self.gamma * self._robust(v_next, step) * self.masks[step + 1]
                            - v_cur
                        )
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + v_cur
                    else:
                        v_next = self.value_preds[step + 1]
                        delta = (
                            self.rewards[step]
                            + self.gamma * self._robust(v_next, step) * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * self.masks[step + 1]
                        + self.rewards[step]
                    )
