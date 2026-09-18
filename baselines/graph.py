import numpy as np
import torch
from torch import nn

def correlation_graph(train_values, k):
    values = np.asarray(train_values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or len(values) < 2:
        raise ValueError('Provide finite training observations [time,nodes].')
    nodes = values.shape[1]
    if not 1 <= k < nodes:
        raise ValueError('Graph k must be smaller than the number of nodes.')
    centered = values - values.mean(axis=0)
    norms = np.sqrt(np.square(centered).sum(axis=0))
    denominator = norms[:, None] * norms[None, :]
    correlation = np.divide(np.abs(centered.T @ centered), denominator,
                            out=np.zeros((nodes, nodes)), where=denominator > 0)
    np.fill_diagonal(correlation, 0.)
    graph = np.zeros_like(correlation)
    for node in range(nodes):
        scores = correlation[node].copy()
        scores[node] = -np.inf
        neighbors = np.argsort(-scores, kind='stable')[:k]
        graph[node, neighbors] = correlation[node, neighbors]
    return np.maximum(graph, graph.T).astype(np.float32)


class A3TGCN(nn.Module):
    def __init__(self, context, horizon, hidden_size, adjacency, batch_size):
        super().__init__()
        from .vendor.pgt.attentiontemporalgcn import A3TGCN2
        graph = torch.as_tensor(adjacency, dtype=torch.float32)
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1] or len(graph) < 1:
            raise ValueError('Expected a nonempty square monitoring-node graph.')
        if not torch.isfinite(graph).all() or torch.any(graph < 0) or not torch.allclose(graph, graph.T):
            raise ValueError('Expected finite nonnegative symmetric graph weights.')
        if torch.count_nonzero(graph.diagonal()) or min(context, horizon, hidden_size, batch_size) < 1:
            raise ValueError('Supply a graph without self-loops and positive model dimensions.')
        self.context, self.hidden_size = context, hidden_size
        self.register_buffer('adjacency', graph)
        edge_index = graph.nonzero().t().contiguous()
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', graph[edge_index[0], edge_index[1]])
        self.core = A3TGCN2(1, hidden_size, context, batch_size,
                           improved=False, cached=False, add_self_loops=True).cpu()
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.context, len(self.adjacency)) or x.shape[0] < 1:
            raise ValueError('A3TGCN expects [batch, context, monitoring nodes].')
        hidden = x.new_zeros(x.shape[0], x.shape[2], self.hidden_size)
        state = self.core(x.transpose(1, 2).unsqueeze(2), self.edge_index, self.edge_weight, hidden)
        return self.head(state).transpose(1, 2)
