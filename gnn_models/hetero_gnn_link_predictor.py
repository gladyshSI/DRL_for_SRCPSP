import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv


class HeteroGNN(nn.Module):
    def __init__(self, metadata, in_dims: dict[str, int], hidden_dim=64, num_layers=2, scorer_hidden=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_types = metadata[0]
        self.edge_types = metadata[1]

        # per-type input encoders -> hidden_dim
        self.encoders = nn.ModuleDict()
        for t in self.node_types:
            in_dim = in_dims.get(t, 0)
            if in_dim <= 0:
                # fallback: tiny linear from 1->hidden (user should replace with Embedding if needed)
                self.encoders[t] = nn.Linear(1, hidden_dim)
            else:
                self.encoders[t] = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )

        # HeteroConv stack: one conv module per edge type
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in self.edge_types:
                conv = GATv2Conv((-1, -1), hidden_dim, heads=2, concat=False, add_self_loops=False)
                conv_dict[edge_type] = conv
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        # to normalize
        self.norms = nn.ModuleList([
            nn.ModuleDict({t: nn.LayerNorm(hidden_dim) for t in self.node_types})
            for _ in range(num_layers)
        ])

        # final projection per node type
        self.post_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in self.node_types})

        # NULL embedding appended to every destination pool
        self.null_emb = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)

        # MLP scorer: consumes concatenation [emb_src || emb_dst] -> scalar score
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, scorer_hidden),
            nn.ReLU(),
            nn.Linear(scorer_hidden, 1)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # encode
        h_dict = {}
        for t, x in x_dict.items():
            if x is None:
                num_nodes = 1 if not hasattr(self, 'num_nodes_'+t) else getattr(self, 'num_nodes_'+t)
                dummy = torch.zeros(num_nodes, 1, device=self.null_emb.device)
                h_dict[t] = self.encoders[t](dummy)
            else:
                h_dict[t] = self.encoders[t](x)

        # hetero convs
        for i, conv in enumerate(self.convs):
            h_prev = h_dict
            h_dict = conv(h_dict, edge_index_dict)

            h_dict = {
                t: self.norms[i][t](h_dict[t] + h_prev[t])
                for t in h_dict
            }

            h_dict = {
                t: F.relu(h)
                for t, h in h_dict.items()
            }

        out = {t: self.post_proj[t](h) for t, h in h_dict.items()}
        return out
