"""Advantage Alignment algorithm for HARL.

Advantage Alignment (Duque et al., "Advantage Alignment Algorithms",
arXiv:2406.14662) is an opponent-shaping method: it shapes each agent's policy
gradient so that an agent increases the probability of an action proportionally
to the product of (i) its own *cumulative past advantage* and (ii) the *other
agents' current advantage*. This "aligns" interacting agents' advantages,
promoting mutually beneficial behaviour. Concretely the agent optimizes a PPO
surrogate on a **modified advantage** (paper Eq. 10):

    A*_t^i = A_t^i + beta * ( sum_{k<t} gamma^{t-k} A_k^i ) * ( sum_{j != i} A_t^j )

and clips it exactly like PPO (Eq. 9). The released code's ``integrated_aa`` mode
uses a plain cumulative sum normalized by ``t`` in place of the discounted sum.

This class is the *actor* side of the HARL adaptation. The cross-agent term needs
*all* agents' advantages simultaneously, which are only available in the runner
(``OnPolicyAdvAlignRunner``); the runner therefore computes the shaped advantage
``A*`` for every agent and hands each agent its slice. Consequently this actor is
exactly HAPPO **except that it does not re-normalize the advantages** (they have
already been normalized and AA-shaped by the runner). The PPO clipped-surrogate
update itself is inherited unchanged from HAPPO, so it operates on ``A*`` — which
is precisely Eq. 9 + Eq. 10.
"""

import numpy as np

from harl.algorithms.actors.happo import HAPPO


class AdvantageAlignment(HAPPO):
    def train(self, actor_buffer, advantages, state_type):
        """HAPPO actor update on the runner-provided AA-shaped advantages.

        Identical to ``HAPPO.train`` but WITHOUT the EP advantage-normalization
        step: ``advantages`` here is already the normalized, advantage-aligned
        ``A*`` produced by ``OnPolicyAdvAlignRunner``. Re-normalizing it would
        destroy the alignment shaping, so we feed it straight into the clipped
        PPO surrogate (``HAPPO.update``).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        # NOTE: no advantage normalization here (the AA runner already normalized
        # the raw advantages before computing the alignment term).

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

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
