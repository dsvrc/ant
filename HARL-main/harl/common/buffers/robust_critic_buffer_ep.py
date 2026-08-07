"""EP critic buffer with the COMARL distributionally-robust Bellman operator.

This subclass only overrides ``compute_returns``: every place the standard GAE
bootstraps with the next-state value ``gamma * V(s') * mask`` it instead uses the
*robust* value ``gamma * sigma_rho(V(s')) * mask``, where ``sigma_rho`` is the
COMARL robust operator (``contamination`` or ``tv``; see
``harl.models.comarl.comarl_modules.robust_next_value``).

``g_values`` (the G-network output for each stored state, in true/denormalized
value scale) is indexed by the *current* step -- ``g_values[step]`` pairs with the
bootstrap value ``V(s_{step+1})`` -- exactly mirroring the reference, where
``g(s_t, a_t)`` is paired with ``V'(s_{t+1})``.

With ``rho = 0`` (or ``mode='contamination'`` and ``rho=0``) this reduces exactly
to the standard HAPPO return computation.
"""

from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.models.comarl.comarl_modules import robust_next_value


class RobustCriticBufferEP(OnPolicyCriticBufferEP):
    """EP critic buffer whose returns use the COMARL robust Bellman operator."""

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
                shape (episode_length + 1, n_rollout_threads, 1), true value scale.
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
