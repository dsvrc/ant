"""Neural modules for ESCP (Adapt to Environment Sudden Changes by Learning a
Context Sensitive Policy, AAAI 2022) adapted to HARL.

Faithful re-implementation of ESCP's two signature pieces from the official repo
(https://github.com/FanmingL/ESCP, ``algorithms/RMDM.py`` + ``models``):

* an **Environment Probe (EP)** -- a history-truncated GRU that maps the recent
  (state, last_action) context to a low-dim environment embedding, optionally
  bottlenecked (stochastic at train, deterministic at test); and
* the **RMDM loss** = *variance minimization* (consistency: same-task embeddings
  collapse to a stable EMA mean) + *Relational Matrix Determinant Maximization*
  (diversity: ``-logdet`` of an RBF relational matrix over the per-task mean
  embeddings, which pushes different tasks apart and avoids trivial collapse).

The EMA per-task means (``self.mean_vector``) persist across iterations, so the
relational-determinant term stays well-defined even when a single training batch
only contains one task -- exactly ESCP's ``history_env_mean`` mechanism. This is
what lets ESCP work on the TCC Ant, where the lock-step threads sit at one
(discretized) ``ambient`` bin per iteration but drift through all bins over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Environment Probe (history-truncated GRU context encoder)
# ---------------------------------------------------------------------------
class EnvProbe(nn.Module):
    def __init__(self, obs_dim, act_dim, ep_dim, hidden=64, bottleneck=False):
        super(EnvProbe, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ep_dim = ep_dim
        self.hidden = hidden
        self.bottleneck = bottleneck

        self.gru = nn.GRU(obs_dim + act_dim, hidden, num_layers=1, batch_first=True)
        if bottleneck:
            self.fc_mu = nn.Linear(hidden, ep_dim)
            self.fc_logvar = nn.Linear(hidden, ep_dim)
        else:
            self.fc = nn.Linear(hidden, ep_dim)

    def forward(self, obs_seq, act_seq, hidden=None, deterministic=False):
        """Encode a (obs, last_action) sequence into an environment embedding.

        Args:
            obs_seq: (B, L, obs_dim).
            act_seq: (B, L, act_dim) previous actions.
            hidden: (1, B, hidden) GRU state or None.
            deterministic: (bool) for the bottleneck, return the mean.
        Returns:
            ep: (B, L, ep_dim) embedding (tanh-bounded).
            mu, logvar: (B, L, ep_dim) (logvar None if no bottleneck).
            hidden: (1, B, hidden) updated GRU state.
        """
        x = torch.cat([obs_seq, act_seq], dim=-1)
        out, hidden = self.gru(x, hidden) if hidden is not None else self.gru(x)
        if self.bottleneck:
            mu = torch.tanh(self.fc_mu(out))
            logvar = self.fc_logvar(out)
            if deterministic:
                ep = mu
            else:
                std = torch.exp(0.5 * logvar)
                ep = mu + std * torch.randn_like(std)
                ep = torch.tanh(ep)
            return ep, mu, logvar, hidden
        else:
            ep = torch.tanh(self.fc(out))
            return ep, ep, None, hidden


# ---------------------------------------------------------------------------
# RMDM: relational-matrix-determinant diversity (DPP -logdet) helpers
# ---------------------------------------------------------------------------
def get_rbf_matrix(data, centers, alpha):
    """RBF Gram matrix K[i,j] = exp(-alpha * ||data_i - centers_j||^2)."""
    n, m, d = data.shape[0], centers.shape[0], data.shape[-1]
    data = data.unsqueeze(1).expand(n, m, d)
    centers = centers.unsqueeze(0).expand(n, m, d)
    return (-(centers - data).pow(2).sum(dim=-1) * alpha).exp()


def get_loss_dpp(y, kernel="rbf", rbf_radius=3000.0):
    """DPP diversity loss = -logdet(K). Maximizing logdet spreads the rows of y."""
    if kernel == "rbf":
        K = get_rbf_matrix(y, y, alpha=rbf_radius) + torch.eye(
            y.shape[0], device=y.device
        ) * 1e-3
    elif kernel == "inner":
        K = y.matmul(y.t()).exp() + torch.eye(y.shape[0], device=y.device) * 1e-3
    else:
        raise NotImplementedError(f"unknown kernel {kernel}")
    return -torch.logdet(K)


class RMDMLoss:
    """ESCP's RMDM representation loss with EMA per-task mean embeddings.

    Holds the running (EMA) mean embedding per task id so the diversity term can
    be computed over all tasks seen so far, not just those in the current batch.
    """

    def __init__(self, ep_dim, tau=0.995, rbf_radius=3000.0, kernel="rbf"):
        self.ep_dim = ep_dim
        self.tau = tau
        self.rbf_radius = rbf_radius
        self.kernel = kernel
        self.mean_vector = {}  # task_id -> (1, ep_dim) EMA mean (detached)

    def __call__(self, ep, tasks, consis_w, diverse_w):
        """Compute the RMDM loss.

        Args:
            ep: (N, ep_dim) embeddings (one row per representation point).
            tasks: (N,) integer task ids (0 = invalid, skipped).
            consis_w, diverse_w: (float) loss weights.
        Returns:
            loss, consistency, diversity (tensors) or (None, None, None).
        """
        device = ep.device
        unique = [int(t) for t in torch.unique(tasks).detach().cpu().tolist() if int(t) != 0]
        if len(unique) == 0:
            return None, None, None

        mean_list = []
        var_terms = []
        count_total = 0
        for b in unique:
            mask = tasks == b
            ep_b = ep[mask]  # (n_b, ep_dim)
            n_b = ep_b.shape[0]
            if n_b == 0:
                continue
            repre = ep_b.mean(dim=0, keepdim=True)  # (1, ep_dim)
            if b not in self.mean_vector:
                self.mean_vector[b] = repre.detach()
            else:
                self.mean_vector[b] = (
                    repre.detach() * (1 - self.tau) + self.mean_vector[b] * self.tau
                ).detach()
            mean_list.append(repre)
            var_b = (ep_b - self.mean_vector[b].detach()).pow(2).sum() / n_b / self.ep_dim
            var_terms.append(var_b * n_b)
            count_total += n_b

        if len(mean_list) == 0 or count_total == 0:
            return None, None, None

        # --- consistency (variance minimization) ---
        var = sum(var_terms) / count_total
        consistency = var.sqrt()
        if consistency.item() < 1e-3:
            consistency = consistency.detach()

        # --- diversity (relational matrix determinant maximization) ---
        repres = list(mean_list)
        batch_bins = set(unique)
        for b, m in self.mean_vector.items():
            if b not in batch_bins:
                repres.append(m)  # historical EMA means
        repre_tensor = torch.cat(repres, dim=0)  # (num_tasks_total, ep_dim)
        if repre_tensor.shape[0] >= 2:
            diversity = get_loss_dpp(
                repre_tensor, kernel=self.kernel, rbf_radius=self.rbf_radius
            )
        else:
            diversity = torch.zeros((), device=device)

        loss = consis_w * consistency + diverse_w * diversity
        return loss, consistency, diversity
