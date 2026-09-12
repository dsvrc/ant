import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.models.base.act import ACTLayer
from harl.utils.envs_tools import get_shape_from_obs_space


class StochasticPolicy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(StochasticPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        # PACT-1 reads its per-action predicted cost off the TAIL of the
        # observation, and the host network must never see it: with the tail
        # peeled off, the base network is sized and shaped exactly as it is on a
        # stock run, which is what makes `blind` and `pactoff` bit-identical.
        self.pact_k = int(args.get("pact_n_actions", 0) or 0)

        obs_shape = get_shape_from_obs_space(obs_space)
        if self.pact_k > 0:
            obs_shape = list(obs_shape)
            obs_shape[0] = int(obs_shape[0]) - self.pact_k
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,
        )

        # ================= PACT-1: the trust term, and nothing else ==========
        # The ONLY difference between the arms.  Everything above this line is
        # the host's, byte for byte, so an arm difference cannot be an algorithm
        # difference (P-9.1).
        #
        # The per-action predicted cost rides the TAIL of the observation, and the
        # base network never sees it (`obs[..., :-K]` below), so the host network
        # is structurally the stock one and `blind`/`pactoff` are bit-identical.
        #
        # P-5.1 -- INVERT THE PRIOR.  `w` starts at 0 and `trust_from_w` puts that
        # at sigmoid(2.2) = 0.90 of g_max: trust the estimator unless the return
        # says otherwise.  Starting at half asks a weak, noisy policy gradient to
        # walk uphill to a known-correct answer; URB measured 3642 against 5444.
        self.pact_mode = str(args.get("pact_trust", "off"))
        self.pact_kappa = float(args.get("pact_kappa", 1.0))
        self.pact_gmax = float(args.get("pact_g_max", 1.0))
        self.pact_bias = float(args.get("pact_trust_bias", 2.2))
        self.pact_gfixed = float(args.get("pact_g_fixed", 0.9))
        if self.pact_k > 0 and self.pact_mode == "learned":
            self.pact_w = nn.Parameter(torch.zeros(1))
        else:
            self.pact_w = None

        self.to(device)

    def _pact_split(self, obs):
        """Peel the predicted-cost tail off the observation.

        Returns ``(obs_for_the_base_network, cost_hat_or_None)``.  When the
        steering is off the observation is returned untouched, so this method is a
        no-op on every non-PACT arm.
        """
        if self.pact_k <= 0:
            return obs, None
        return obs[..., : -self.pact_k], obs[..., -self.pact_k:]

    def _pact_bias(self, cost, available_actions):
        """``-g * kappa * z`` -- the shift of II.6, as a logit bias.

        The z-score is taken over the agent's VALID options only, which is what
        makes ``kappa`` a single declared constant instead of a per-instance scale
        factor (P-6.1) -- and a tuned kappa would be a tuned result.

        THE FLOOR PROPERTY (P-7.1): at ``g == 0`` this returns None, so the host's
        logits are used bit for bit however wrong the estimate is.  The same holds
        when every predicted cost is identical, where the shift is defined to be
        zero rather than NaN.
        """
        if cost is None or self.pact_mode == "off":
            return None
        if self.pact_mode == "fixed":
            g = self.pact_gfixed * self.pact_gmax
        else:
            g = self.pact_gmax * torch.sigmoid(self.pact_w + self.pact_bias)
        m = (available_actions > 0).float() if available_actions is not None             else torch.ones_like(cost)
        # never let a masked option enter the statistics
        n = m.sum(-1, keepdim=True)
        mean = (cost * m).sum(-1, keepdim=True) / n.clamp(min=1.0)
        var = (((cost - mean) ** 2) * m).sum(-1, keepdim=True) / n.clamp(min=1.0)
        sd = var.clamp(min=0.0).sqrt()
        z = torch.where(sd > 1e-12, (cost - mean) / sd.clamp(min=1e-12),
                        torch.zeros_like(cost))
        z = torch.where(n > 1.5, z, torch.zeros_like(z))    # <2 options: no shift
        return -(g * self.pact_kappa) * z * m

    def forward(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        obs, pact_cost = self._pact_split(obs)
        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic,
            logit_bias=self._pact_bias(pact_cost, available_actions),
        )

        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            action: (np.ndarray / torch.Tensor) actions whose entropy and log probability to evaluate.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # The SAME split and the SAME shift as `forward`, recomputed from the SAME
        # stored observation, so the surrogate is evaluated under the distribution
        # that actually generated the action and the trust parameter receives its
        # gradient through the host's own clipped objective (P-6.2).
        obs, pact_cost = self._pact_split(obs)
        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self.use_policy_active_masks else None,
            logit_bias=self._pact_bias(pact_cost, available_actions),
        )

        return action_log_probs, dist_entropy, action_distribution
