"""Network modules for MAMT (Multi-Agent trust-region decomposition; arXiv:2102.10616).

* ``tsallis_log_q`` -- the Tsallis q-logarithm used in the mirror-descent trust-region
  term (``log_q(x) = (x^{1-q} - 1)/(1-q)``, ``log(x)`` for ``q=1``).

* ``ModelingPolicy`` -- a teammate-modeling Gaussian policy: from an agent's own
  observation and a one-hot id of a teammate it predicts that teammate's action
  distribution. The mismatch between the model and the teammate's real policy is the
  per-agent **non-stationarity** measurement ``d_ns``.

* ``TRDNet`` -- the Trust-Region-Decomposition network (the paper's TRD-Net / TRAN):
  a message-passing GNN over the coordination graph that estimates each agent's
  contribution to the joint policy divergence from per-agent state-action features and
  the (learnable) local trust regions. Reimplemented with a **dense GCN** so no
  ``torch_geometric`` dependency is needed (mirrors COREP's dense-GAT port).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def tsallis_log_q(x, q=0.5):
    """Tsallis q-logarithm. x: tensor of (densities/probabilities)."""
    safe_x = torch.clamp(x, min=1e-6)
    if q == 1.0:
        return torch.log(safe_x)
    return (torch.pow(safe_x, 1.0 - q) - 1.0) / (1.0 - q)


class ModelingPolicy(nn.Module):
    """Teammate-modeling policy: (own obs, teammate one-hot) -> teammate policy.

    Continuous (``discrete=False``): a Gaussian ``N(mu, std)`` over the teammate's
    action. Discrete (``discrete=True``, SMAC/SMACv2): a categorical over the
    teammate's ``act_dim`` actions, returned as log-probabilities. ``forward``
    always returns a *tuple* of distribution parameters -- ``(mean, std)`` in the
    continuous case, ``(logprobs,)`` in the discrete case -- so callers dispatch
    on ``self.discrete`` via the ``dist_kl`` / ``model_loss`` helpers below.
    """

    def __init__(self, obs_dim, onehot_dim, act_dim, hidden_dim=128, discrete=False):
        super().__init__()
        self.act_dim = act_dim
        self.discrete = discrete
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + onehot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        if discrete:
            self.logits = nn.Linear(hidden_dim, act_dim)
        else:
            self.mean = nn.Linear(hidden_dim, act_dim)
            self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs, onehot):
        h = self.trunk(torch.cat([obs, onehot], dim=-1))
        if self.discrete:
            return (torch.log_softmax(self.logits(h), dim=-1),)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5.0, 2.0)
        return (mean, log_std.exp())

    @staticmethod
    def gaussian_kl(mu_p, std_p, mu_q, std_q):
        """KL( N(mu_p, std_p) || N(mu_q, std_q) ), summed over action dims -> (B,)."""
        var_p, var_q = std_p.pow(2), std_q.pow(2)
        kl = (
            torch.log(std_q)
            - torch.log(std_p)
            + (var_p + (mu_p - mu_q).pow(2)) / (2.0 * var_q)
            - 0.5
        ).sum(-1)
        return kl

    @staticmethod
    def categorical_kl(logp, logq):
        """KL( Cat(logp) || Cat(logq) ) over log-probabilities -> (B,)."""
        return (logp.exp() * (logp - logq)).sum(-1)

    @staticmethod
    def dist_kl(discrete, params_p, params_q):
        """KL(p || q) between two policy-parameter tuples (dispatch on action type)."""
        if discrete:
            return ModelingPolicy.categorical_kl(params_p[0], params_q[0])
        return ModelingPolicy.gaussian_kl(
            params_p[0], params_p[1], params_q[0], params_q[1]
        )

    @staticmethod
    def model_loss(discrete, pred, target):
        """Supervised loss regressing the model ``pred`` onto the teammate ``target``.

        Continuous: MSE on mean + MSE on log-std (as in the reference). Discrete:
        cross-entropy of the teammate's (target) action distribution under the
        model's predicted log-probs.
        """
        if discrete:
            target_probs = target[0].exp()
            return -(target_probs * pred[0]).sum(-1).mean()
        return F.mse_loss(pred[0], target[0]) + F.mse_loss(
            torch.log(pred[1]), torch.log(target[1])
        )


class TRDNet(nn.Module):
    """Trust-Region-Decomposition network (dense-GCN message passing).

    Estimates per-agent joint-policy-divergence contributions ``kl_hat`` from
    per-agent state-action features and the local trust regions, propagating
    information over the coordination graph (edge weights = coordination coeffs).
    Output ``kl_hat = lambda * k`` per agent (as in the reference TRAN).
    """

    def __init__(self, feat_dim, hidden_dim=32, sparse=0.05):
        super().__init__()
        self.sparse = sparse
        # k-network
        self.k_sa = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.LeakyReLU())
        self.k_tr = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
        )
        self.k_enc = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.LeakyReLU())
        self.gcn_lin = nn.Linear(hidden_dim, hidden_dim)
        self.k_dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1), nn.LeakyReLU(),
        )
        # lambda-network
        self.l_sa = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.LeakyReLU())
        self.l_tr = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
        )
        self.l_enc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1), nn.LeakyReLU(),
        )

    def _dense_gcn(self, x, ccs):
        """Dense GCN propagation. x: (B, N, H); ccs: (B, N, N) -> (B, N, H)."""
        B, N, _ = x.shape
        eye = torch.eye(N, device=x.device).unsqueeze(0)
        # sparsified adjacency, self-loops, weighted by coordination coeffs
        adj = (ccs >= self.sparse).float()
        adj = torch.clamp(adj + eye, max=1.0)
        weight = torch.where(eye.bool(), torch.ones_like(ccs), ccs)
        adj = adj * weight
        deg = adj.sum(-1).clamp(min=1e-6)
        dinv = deg.pow(-0.5)
        norm = dinv.unsqueeze(-1) * adj * dinv.unsqueeze(-2)  # (B, N, N)
        return torch.bmm(norm, self.gcn_lin(x))

    def forward(self, sa_feats, trs, ccs):
        """sa_feats: (B, N, feat); trs: (N,) local trust regions; ccs: (B, N, N).

        Returns kl_hat: (B, N).
        """
        B, N, _ = sa_feats.shape
        tr_in = trs.view(1, N, 1).expand(B, N, 1)

        # k branch
        k_sa = self.k_sa(sa_feats)
        k_tr = self.k_tr(tr_in)
        k_enc = self.k_enc(torch.cat([k_sa, k_tr], dim=-1))
        k_embed = self._dense_gcn(k_enc, ccs)
        k = self.k_dec(k_embed).squeeze(-1)  # (B, N)

        # lambda branch
        l_sa = self.l_sa(sa_feats)
        l_tr = self.l_tr(tr_in)
        lamb = self.l_enc(torch.cat([l_sa, l_tr], dim=-1)).squeeze(-1)  # (B, N)

        return lamb * k
