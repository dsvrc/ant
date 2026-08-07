"""Neural modules for COREP (Causal-Origin REPresentation) adapted to HARL.

This is a faithful re-implementation of the COREP encoder/decoder stack from the
official repository (https://github.com/PKU-RL/COREP), with two engineering
changes that do **not** alter the method:

1. The ``torch_geometric`` ``GATConv`` + per-graph ``Data``/``Batch`` Python loop
   is replaced by a mathematically-equivalent *dense batched multi-head GAT*
   (``DenseGAT``). This removes the heavyweight ``torch_geometric`` dependency
   and is dramatically faster for the large ``episode_length * n_rollout_threads``
   batches produced by HARL (the original looped over every graph in Python).
2. Everything is written to operate on tensors that already live on the right
   device (HARL passes the device explicitly), instead of the global ``device``
   the original code relied on.

The COREP-specific pieces are preserved exactly:
  * ``StateToGraph`` with a Gumbel-softmax learned adjacency,
  * a *dual* GAT (``stable`` + ``flexible``) whose mean-pooled node embeddings are
    summed to form the causal-origin state representation ``hs``,
  * a VariBAD-style GRU belief encoder producing a variational latent,
  * reward / state-transition decoders for the ELBO,
  * ``gumbel_adjacency_matrix`` / ``sigmoid_adjacency_matrix`` helpers used by the
    graph / sparsity / magnitude regularization losses.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adjacency helpers (kept identical in spirit to the original COREP code)
# ---------------------------------------------------------------------------
def gumbel_adjacency_matrix(node_embeddings, similarity_threshold, temperature, hard):
    """Build a (differentiable) adjacency matrix from node embeddings.

    Cosine similarity -> sigmoid -> center by threshold -> Gumbel-softmax (row-wise)
    -> remove self-loops. Shapes: node_embeddings (..., n, d) -> adjacency (..., n, n).
    """
    node_norm = torch.norm(node_embeddings, p=2, dim=-1, keepdim=True)
    norm_matrix = torch.matmul(node_norm, node_norm.transpose(-2, -1))
    similarity_matrix = torch.matmul(
        node_embeddings, node_embeddings.transpose(-2, -1)
    ) / (norm_matrix + 1e-8)
    similarity_matrix = torch.sigmoid(similarity_matrix)
    sim_matrix_centered = similarity_matrix - similarity_threshold
    adjacency_matrix = F.gumbel_softmax(sim_matrix_centered, temperature, hard=hard, dim=-1)
    n = node_embeddings.shape[-2]
    eye = torch.eye(n, device=node_embeddings.device)
    # broadcast the identity over all leading (batch) dimensions
    adjacency_matrix = adjacency_matrix * (1 - eye)
    return adjacency_matrix


def sigmoid_adjacency_matrix(node_embeddings, similarity_threshold):
    """Soft adjacency used by the graph/sparsity/magnitude regularization losses.

    Matches ``sigmoid_adjacency_matrix`` in the original ``vae.py``.
    Shapes: node_embeddings (..., n, d) -> adjacency (..., n, n).
    """
    node_norm = torch.norm(node_embeddings, p=2, dim=-1, keepdim=True)
    norm_matrix = torch.matmul(node_norm, node_norm.transpose(-2, -1))
    similarity_matrix = torch.matmul(
        node_embeddings, node_embeddings.transpose(-2, -1)
    ) / (norm_matrix + 1e-8)
    soft_similarity_matrix = similarity_matrix / -similarity_threshold
    soft_adjacency_matrix = torch.sigmoid(soft_similarity_matrix)
    n = node_embeddings.shape[-2]
    eye = torch.eye(n, device=node_embeddings.device)
    adjacency_matrix = soft_adjacency_matrix * (1 - eye)
    return adjacency_matrix


# ---------------------------------------------------------------------------
# Small feature extractor (linear + activation), like utils.helpers.FeatureExtractor
# ---------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size, activation_function=F.relu):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return torch.zeros(0, device=inputs.device)


# ---------------------------------------------------------------------------
# State -> graph: project state to node embeddings + learned (Gumbel) adjacency
# ---------------------------------------------------------------------------
class StateToGraph(nn.Module):
    def __init__(self, state_dim, node_feature_dim, num_nodes):
        super(StateToGraph, self).__init__()
        self.fc1 = nn.Linear(state_dim, num_nodes * node_feature_dim)
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim

    def forward(self, x, similarity_threshold, temperature, hard):
        # x: (N, state_dim)
        x = torch.relu(self.fc1(x))
        node_emb = x.reshape(-1, self.num_nodes, self.node_feature_dim)  # (N, n, F)
        adjacency_matrix = gumbel_adjacency_matrix(
            node_emb,
            similarity_threshold=similarity_threshold,
            temperature=temperature,
            hard=hard,
        )  # (N, n, n)
        return node_emb, adjacency_matrix


# ---------------------------------------------------------------------------
# Dense batched multi-head GAT layer (equivalent to torch_geometric GATConv)
# ---------------------------------------------------------------------------
class DenseGATLayer(nn.Module):
    """A single graph-attention layer operating on a dense adjacency mask.

    Implements the standard GAT attention
        e_ij = LeakyReLU(a_src . W h_i + a_dst . W h_j)
    with masked softmax over the neighbours given by ``adj`` (self-loops added,
    exactly as ``torch_geometric.nn.GATConv`` does by default).
    """

    def __init__(self, in_features, out_features, heads=1, concat=True, dropout=0.6,
                 negative_slope=0.2):
        super(DenseGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, 1, heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(1, 1, heads, out_features))
        self.bias = nn.Parameter(
            torch.zeros(heads * out_features if concat else out_features)
        )

        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h, adj):
        # h: (N, n, in_features); adj: (N, n, n)
        N, n, _ = h.shape
        Wh = self.W(h).view(N, n, self.heads, self.out_features)  # (N, n, H, out)

        e_src = (Wh * self.a_src).sum(dim=-1)  # (N, n, H)  contribution of source i
        e_dst = (Wh * self.a_dst).sum(dim=-1)  # (N, n, H)  contribution of target j
        # e[:, i, j, h] = src_i + dst_j  -> attention of node i over neighbour j
        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)  # (N, n, n, H)
        e = F.leaky_relu(e, self.negative_slope)

        # add self-loops to the adjacency mask (GATConv default add_self_loops=True)
        eye = torch.eye(n, device=h.device).unsqueeze(0)  # (1, n, n)
        mask = adj + eye
        mask = (mask > 0).float().unsqueeze(-1)  # (N, n, n, 1)

        # self-loops guarantee every row has at least one finite entry, so the
        # masked softmax can never produce a fully-(-inf) (NaN) row.
        e = e.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(e, dim=2)  # softmax over neighbours j
        attn = F.dropout(attn, self.dropout, training=self.training)

        # aggregate: out[:, i, h, :] = sum_j attn[:, i, j, h] * Wh[:, j, h, :]
        out = torch.einsum("nijh,njhf->nihf", attn, Wh)  # (N, n, H, out)

        if self.concat:
            out = out.reshape(N, n, self.heads * self.out_features)
        else:
            out = out.mean(dim=2)  # (N, n, out)
        out = out + self.bias
        return out


class DenseGAT(nn.Module):
    """Two-layer GAT, mirroring the original ``GAT`` (GATConv -> ELU -> GATConv)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, dropout=0.6):
        super(DenseGAT, self).__init__()
        self.gat1 = DenseGATLayer(
            input_dim, hidden_dim, heads=num_heads, concat=True, dropout=dropout
        )
        self.gat2 = DenseGATLayer(
            hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=dropout
        )

    def forward(self, node_emb, adj):
        x = self.gat1(node_emb, adj)
        x = F.elu(x)
        x = self.gat2(x, adj)
        return x  # (N, n, output_dim)


# ---------------------------------------------------------------------------
# COREP encoder: dual-GAT causal-origin representation + GRU belief
# ---------------------------------------------------------------------------
class CorepEncoder(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim=8,
        num_nodes=8,
        num_heads=4,
        node_feature_dim=16,
        gat_hidden_dim=32,
        state_embed_dim=16,
        action_embed_dim=16,
        reward_embed_dim=16,
        gru_hidden_size=128,
        gat_dropout=0.6,
        graph_threshold=0.8,
        graph_temperature=0.8,
    ):
        super(CorepEncoder, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.state_embed_dim = state_embed_dim
        self.hidden_size = gru_hidden_size
        self.graph_threshold = graph_threshold
        self.graph_temperature = graph_temperature

        # state -> graph -> dual GAT
        self.graph_encoder = StateToGraph(state_dim, node_feature_dim, num_nodes)
        self.stable_gat = DenseGAT(
            node_feature_dim, gat_hidden_dim, state_embed_dim, num_heads, gat_dropout
        )
        self.flexible_gat = DenseGAT(
            node_feature_dim, gat_hidden_dim, state_embed_dim, num_heads, gat_dropout
        )

        # action / reward embeddings
        self.action_encoder = FeatureExtractor(action_dim, action_embed_dim, F.relu)
        self.reward_encoder = FeatureExtractor(1, reward_embed_dim, F.relu)

        # GRU belief
        gru_input_dim = action_embed_dim + state_embed_dim + reward_embed_dim
        self.gru = nn.GRU(input_size=gru_input_dim, hidden_size=gru_hidden_size, num_layers=1)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        # latent output heads
        self.fc_mu = nn.Linear(gru_hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(gru_hidden_size, latent_dim)

    # -- utilities -----------------------------------------------------------
    def _device(self):
        return self.fc_mu.weight.device

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def reset_hidden(self, hidden_state, done):
        """Zero the GRU hidden state where ``done`` (start of a new episode)."""
        if done is None:
            return hidden_state
        if hidden_state.dim() != done.dim():
            if done.dim() == 2:
                done = done.unsqueeze(0)
            elif done.dim() == 1:
                done = done.unsqueeze(0).unsqueeze(2)
        return hidden_state * (1 - done)

    def prior(self, batch_size, sample=True):
        device = self._device()
        hidden_state = torch.zeros(
            (1, batch_size, self.hidden_size), device=device, requires_grad=True
        )
        latent_mean = self.fc_mu(hidden_state)
        latent_logvar = self.fc_logvar(hidden_state)
        latent_sample = (
            self.reparameterise(latent_mean, latent_logvar) if sample else latent_mean
        )
        return latent_sample, latent_mean, latent_logvar, hidden_state

    # -- graph representation ------------------------------------------------
    def _dual_gat(self, states_flat, freeze):
        """states_flat: (N, state_dim) -> stable_nodes, flexible_nodes (N, n, embed)."""
        node_emb, adj = self.graph_encoder(
            states_flat, self.graph_threshold, self.graph_temperature, True
        )
        if freeze:
            with torch.no_grad():
                stable_nodes = self.stable_gat(node_emb, adj)
        else:
            stable_nodes = self.stable_gat(node_emb, adj)
        flexible_nodes = self.flexible_gat(node_emb, adj)
        return stable_nodes, flexible_nodes

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        actions,
        states,
        rewards,
        hidden_state,
        return_prior,
        sample=True,
        detach_every=None,
        freeze=False,
    ):
        """Encode a (sequence of) transition(s).

        Shapes are ``(seq_len, batch, dim)``. For a single-step online update pass
        ``seq_len == 1`` and a non-None ``hidden_state``. For a full trajectory pass
        ``hidden_state=None`` and ``return_prior=True`` (the returned tensors then
        have length ``seq_len + 1`` because they include the prior at index 0).
        """
        # promote (batch, dim) -> (1, batch, dim)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        if states.dim() == 2:
            states = states.unsqueeze(0)
        if rewards.dim() == 2:
            rewards = rewards.unsqueeze(0)
        if hidden_state is not None and hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)

        T, B = states.shape[0], states.shape[1]

        if return_prior:
            prior_sample, prior_mean, prior_logvar, prior_hidden = self.prior(B)
            hidden_state = prior_hidden.clone()

        # dual-GAT causal-origin state representation
        states_flat = states.reshape(T * B, self.state_dim)
        stable_nodes, flexible_nodes = self._dual_gat(states_flat, freeze)
        stable_mean = stable_nodes.mean(dim=1).reshape(T, B, self.state_embed_dim)
        flexible_mean = flexible_nodes.mean(dim=1).reshape(T, B, self.state_embed_dim)
        hs = stable_mean + flexible_mean  # (T, B, state_embed_dim)

        ha = self.action_encoder(actions)  # (T, B, action_embed)
        hr = self.reward_encoder(rewards)  # (T, B, reward_embed)
        h = torch.cat((ha, hs, hr), dim=-1)  # (T, B, gru_input_dim)

        if detach_every is None:
            output, _ = self.gru(h, hidden_state)
        else:
            outputs = []
            for i in range(int(np.ceil(h.shape[0] / detach_every))):
                curr_input = h[i * detach_every: i * detach_every + detach_every]
                curr_output, hidden_state = self.gru(curr_input, hidden_state)
                outputs.append(curr_output)
                hidden_state = hidden_state.detach()  # truncated BPTT
            output = torch.cat(outputs, dim=0)

        latent_mean = self.fc_mu(output)
        latent_logvar = self.fc_logvar(output)
        latent_sample = (
            self.reparameterise(latent_mean, latent_logvar) if sample else latent_mean
        )

        # reshape node embeddings to (T, B, n, embed) for the graph losses
        stable_nodes = stable_nodes.reshape(T, B, self.num_nodes, self.state_embed_dim)
        flexible_nodes = flexible_nodes.reshape(T, B, self.num_nodes, self.state_embed_dim)

        if return_prior:
            latent_sample = torch.cat((prior_sample, latent_sample))
            latent_mean = torch.cat((prior_mean, latent_mean))
            latent_logvar = torch.cat((prior_logvar, latent_logvar))
            output = torch.cat((prior_hidden, output))

        return (
            latent_sample,
            latent_mean,
            latent_logvar,
            output,
            stable_nodes,
            flexible_nodes,
        )


# ---------------------------------------------------------------------------
# Decoders for the ELBO (reward and state-transition reconstruction)
# ---------------------------------------------------------------------------
class RewardDecoder(nn.Module):
    """Predict reward from (latent, prev_state, action, next_state)."""

    def __init__(self, latent_dim, state_dim, action_dim, layers=(64, 32)):
        super(RewardDecoder, self).__init__()
        curr = latent_dim + 2 * state_dim + action_dim
        modules = []
        for h in layers:
            modules.append(nn.Linear(curr, h))
            modules.append(nn.ReLU())
            curr = h
        modules.append(nn.Linear(curr, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, latent, prev_state, action, next_state):
        x = torch.cat((latent, prev_state, action, next_state), dim=-1)
        return self.net(x)


class StateTransitionDecoder(nn.Module):
    """Predict next state from (latent, prev_state, action)."""

    def __init__(self, latent_dim, state_dim, action_dim, layers=(64, 32)):
        super(StateTransitionDecoder, self).__init__()
        curr = latent_dim + state_dim + action_dim
        modules = []
        for h in layers:
            modules.append(nn.Linear(curr, h))
            modules.append(nn.ReLU())
            curr = h
        modules.append(nn.Linear(curr, state_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, latent, prev_state, action):
        x = torch.cat((latent, prev_state, action), dim=-1)
        return self.net(x)
