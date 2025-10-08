import torch
from torch_geometric.nn import GATv2Conv


# Simple GNN to produce node embeddings
class SimpleLinkPredictor(torch.nn.Module):
    def __init__(self,conv_layer, in_dim: int, hid_dim: int, out_dim: int, num_layers: int, edge_dim=None):
        super().__init__()
        if num_layers < 2:
            raise ValueError('Number of layers must be at least 2')

        self.convs = torch.nn.ModuleList()
        self.convs.append(conv_layer(in_dim, hid_dim, edge_dim=edge_dim))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hid_dim, hid_dim, edge_dim=edge_dim))
        self.convs.append(conv_layer(hid_dim, out_dim, edge_dim=edge_dim))
        self.relu = torch.nn.ReLU()

        # self.conv1 = GATv2Conv(in_dim, hid_dim, edge_dim=edge_dim, heads=1)
        # self.conv2 = GATv2Conv(hid_dim, hid_dim, edge_dim=edge_dim, heads=1)
        # self.conv3 = GATv2Conv(hid_dim, hid_dim, edge_dim=edge_dim, heads=1)
        # self.conv4 = GATv2Conv(hid_dim, out_dim, edge_dim=edge_dim, heads=1)

        # a small MLP to score an edge from concatenated node embeddings
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2*out_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, 1)   # scalar logit
        )

    def forward(self, x, edge_index, edge_attr=None):
        # x = self.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        # x = self.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        # x = self.relu(self.conv3(x, edge_index, edge_attr=edge_attr))
        # x = self.conv4(x, edge_index, edge_attr=edge_attr)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if i < len(self.convs) - 1:
                x = self.relu(x)
        return x

    def score_edges(self, z_u, z_v):
        # z_u, z_v: [num_pairs, emb_dim]
        z = torch.cat([z_u, z_v], dim=1)
        return self.edge_mlp(z).squeeze(dim=1)

