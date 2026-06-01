import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv


class HeteroGNN(nn.Module):
    def __init__(self, metadata, in_dims, hidden_dim=64, num_layers=2):
        super().__init__()

        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # =========================
        # INPUT ENCODERS
        # =========================
        self.encoders = nn.ModuleDict()
        self.type_embeddings = nn.ParameterDict()

        for t in self.node_types:
            if in_dims[t] > 0:
                self.encoders[t] = nn.Sequential(
                    nn.Linear(in_dims[t], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            else:
                # learnable embedding for node types without features
                self.type_embeddings[t] = nn.Parameter(
                    torch.randn(1, hidden_dim) * 0.01
                )

        # =========================
        # CONVOLUTION LAYERS
        # =========================
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {}

            for edge_type in self.edge_types:
                conv_dict[edge_type] = GATv2Conv(
                    (-1, -1),
                    hidden_dim,
                    heads=4,
                    concat=False,
                    add_self_loops=False,  # IMPORTANT for hetero
                )

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

            # per-type normalization
            self.norms.append(
                nn.ModuleDict({
                    t: nn.LayerNorm(hidden_dim)
                    for t in self.node_types
                })
            )

        # =========================
        # LINK PREDICTION HEAD
        # =========================
        self.link_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # =========================
        # BUFFER / LENGTH PREDICTION HEAD
        # =========================
        self.buf_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # =========================
    # FORWARD
    # =========================
    def forward(self, batch):
        # ---- Encode input features ----
        h_dict = {}

        for t in self.node_types:
            if t in self.encoders:
                if "x" not in batch[t]:
                    raise ValueError(
                        f"Node type {t!r} was initialized with an encoder, "
                        f"but this batch has no x. "
                        f"Fix the dataset so every graph has data[{t!r}].x. "
                        f"For zero nodes, use torch.empty((0, in_dims[{t!r}]))."
                    )

                h_dict[t] = self.encoders[t](batch[t].x)

            else:
                if not hasattr(batch[t], "num_nodes") or batch[t].num_nodes is None:
                    raise ValueError(
                        f"Node type {t!r} is featureless, but num_nodes is missing. "
                        f"Set data[{t!r}].num_nodes explicitly."
                    )

                h_dict[t] = self.type_embeddings[t].expand(
                    batch[t].num_nodes,
                    -1,
                )

        # ---- Edge index dict ----
        edge_index_dict = {
            etype: batch[etype].edge_index
            for etype in batch.edge_types
            if 'edge_index' in batch[etype]
        }

        # ---- Message passing ----
        for i, conv in enumerate(self.convs):
            h_prev = h_dict
            h_dict = conv(h_dict, edge_index_dict)
            # Residual + normalization
            h_dict = {
                t: self.norms[i][t](h_dict[t] + h_prev[t])
                for t in h_dict
            }

            # Activation
            h_dict = {
                t: F.relu(h)
                for t, h in h_dict.items()
            }
        return h_dict


class HeteroGNN_difHeads(nn.Module):
    def __init__(self, metadata, in_dims, hidden_dim=64, num_layers=2):
        super().__init__()

        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # =========================
        # INPUT ENCODERS
        # =========================
        self.encoders = nn.ModuleDict()
        self.type_embeddings = nn.ParameterDict()

        for t in self.node_types:
            if in_dims[t] > 0:
                self.encoders[t] = nn.Sequential(
                    nn.Linear(in_dims[t], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            else:
                # learnable embedding for node types without features
                self.type_embeddings[t] = nn.Parameter(
                    torch.randn(1, hidden_dim) * 0.01
                )

        # =========================
        # CONVOLUTION LAYERS
        # =========================
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {}

            for edge_type in self.edge_types:
                conv_dict[edge_type] = GATv2Conv(
                    (-1, -1),
                    hidden_dim,
                    heads=4,
                    concat=False,
                    add_self_loops=False,  # IMPORTANT for hetero
                )

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

            # per-type normalization
            self.norms.append(
                nn.ModuleDict({
                    t: nn.LayerNorm(hidden_dim)
                    for t in self.node_types
                })
            )

        # =========================
        # RELATION-SPECIFIC HEADS
        # =========================
        self.link_scorers = nn.ModuleDict()
        self.buf_scorers = nn.ModuleDict()

        for edge_type in self.edge_types:
            src_type, rel_type, dst_type = edge_type

            # Only create prediction heads for supervised prediction relations.
            # Message-passing edge types do not need heads.
            if rel_type != "predict":
                continue

            key = self.edge_type_to_str(edge_type)

            self.link_scorers[key] = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

            self.buf_scorers[key] = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    @staticmethod
    def edge_type_to_str(edge_type):
        return "__".join(edge_type)

    # =========================
    # FORWARD
    # =========================
    def forward(self, batch):
        # ---- Encode input features ----
        h_dict = {}

        for t in self.node_types:
            if t in self.encoders:
                if "x" not in batch[t]:
                    raise ValueError(
                        f"Node type {t!r} was initialized with an encoder, "
                        f"but this batch has no x. "
                        f"Fix the dataset so every graph has data[{t!r}].x. "
                        f"For zero nodes, use torch.empty((0, in_dims[{t!r}]))."
                    )

                h_dict[t] = self.encoders[t](batch[t].x)

            else:
                if not hasattr(batch[t], "num_nodes") or batch[t].num_nodes is None:
                    raise ValueError(
                        f"Node type {t!r} is featureless, but num_nodes is missing. "
                        f"Set data[{t!r}].num_nodes explicitly."
                    )

                h_dict[t] = self.type_embeddings[t].expand(
                    batch[t].num_nodes,
                    -1,
                )

        # ---- Edge index dict ----
        edge_index_dict = {
            etype: batch[etype].edge_index
            for etype in batch.edge_types
            if 'edge_index' in batch[etype]
        }

        # ---- Message passing ----
        for i, conv in enumerate(self.convs):
            h_prev = h_dict
            h_dict = conv(h_dict, edge_index_dict)
            # Residual + normalization
            h_dict = {
                t: self.norms[i][t](h_dict[t] + h_prev[t])
                for t in h_dict
            }

            # Activation
            h_dict = {
                t: F.relu(h)
                for t, h in h_dict.items()
            }
        return h_dict
