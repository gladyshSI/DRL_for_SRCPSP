import torch
from torch_geometric.nn import GATv2Conv


# Simple GNN to produce node embeddings
class SimpleLinkValuePredictor(torch.nn.Module):
    def __init__(self, conv_layer, in_dim: int, hid_dim: int, out_dim: int, num_layers: int, edge_dim=None, heads: int = 1):
        super().__init__()
        if num_layers < 2:
            raise ValueError('Number of layers must be at least 2')
        self.config = {'conv_class': conv_layer.__name__,
                       'in_dim': in_dim,
                       'hid_dim': hid_dim,
                       'out_dim': out_dim,
                       'num_layers': num_layers,
                       'edge_dim': edge_dim,
                       'heads': heads}

        self.convs = torch.nn.ModuleList()
        self.convs.append(conv_layer(in_dim, hid_dim, edge_dim=edge_dim, heads=heads, concat=False))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hid_dim, hid_dim, edge_dim=edge_dim, heads=heads, concat=False))
        self.convs.append(conv_layer(hid_dim, out_dim, edge_dim=edge_dim, heads=heads, concat=False))
        self.relu = torch.nn.ReLU()

        # Edge head: takes [z_u || z_v] -> outputs [logit, value]
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2*out_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, 2)   # 0: logit, 1: value (raw)
        )

    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if i < len(self.convs) - 1:
                x = self.relu(x)
        return x

    def score_and_value(self, z_u, z_v):
        # returns logits (for BCE) and scalar values (regression preds)
        z = torch.cat([z_u, z_v], dim=1)
        out = self.edge_mlp(z)  # shape [num_pairs, 2]
        logits = out[:, 0]
        values = out[:, 1]
        return logits, values

