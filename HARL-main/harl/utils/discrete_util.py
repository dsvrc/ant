import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def get_encoded_act_dim(act_space):
    """Dimensionality of the *encoded* action fed into auxiliary networks.

    The representation/dynamics modules of several baselines (COREP, WISDOM,
    ESCP, TRIO, MAMT, MBCD) consume actions as a real-valued vector. For a
    continuous ``Box`` action that is just its dimensionality; for a discrete
    action it is the number of classes, because discrete actions are one-hot
    encoded (an integer index is not a meaningful network input). This mirrors
    ``get_shape_from_act_space`` but returns the *encoded* size instead of the
    stored size (which is 1 for ``Discrete``).

    Args:
        act_space: (gym.spaces) action space of a single agent.
    Returns:
        (int) encoded action dimensionality.
    """
    cls = act_space.__class__.__name__
    if cls == "Discrete":
        return int(act_space.n)
    if cls == "MultiDiscrete":
        return int(sum(act_space.nvec))
    if cls in ("Box", "MultiBinary"):
        return int(act_space.shape[0])
    raise NotImplementedError(f"Unsupported action space for encoding: {cls}")


def encode_actions_torch(actions, act_space):
    """Encode stored actions into a real-valued tensor for auxiliary networks.

    * ``Discrete``: integer indices with a trailing singleton dim ``(..., 1)``
      (as stored by HARL's on-policy actor buffer) are converted to one-hot
      vectors ``(..., n)``.
    * ``MultiDiscrete``: each sub-action is one-hot encoded and the results are
      concatenated along the last dim.
    * ``Box`` / ``MultiBinary``: returned unchanged (already real-valued).

    Args:
        actions: (torch.Tensor) actions, any leading shape, last dim = stored
            action size (1 for Discrete, n_sub for MultiDiscrete, act_dim for Box).
        act_space: (gym.spaces) action space of a single agent.
    Returns:
        (torch.Tensor) float tensor with the encoded action on the last dim.
    """
    cls = act_space.__class__.__name__
    if cls == "Discrete":
        idx = actions
        if idx.shape[-1] == 1:
            idx = idx.squeeze(-1)
        onehot = F.one_hot(idx.long(), num_classes=int(act_space.n))
        return onehot.to(dtype=torch.float32)
    if cls == "MultiDiscrete":
        # actions last dim = number of sub-actions
        outs = []
        for i, n in enumerate(act_space.nvec):
            outs.append(
                F.one_hot(actions[..., i].long(), num_classes=int(n)).to(
                    dtype=torch.float32
                )
            )
        return torch.cat(outs, dim=-1)
    # Box / MultiBinary: already a real-valued vector
    return actions.to(dtype=torch.float32)


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(
        torch.eye(logits.shape[1])[
            [np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
        ],
        requires_grad=False,
    )
    # chooses between best and random actions using epsilon greedy
    return torch.stack(
        [
            argmax_acs[i] if r > eps else rand_acs[i]
            for i, r in enumerate(torch.rand(logits.shape[0]))
        ]
    )


def sample_gumbel(shape, device, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, device):
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data), device=device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, device, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, device=device)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y
