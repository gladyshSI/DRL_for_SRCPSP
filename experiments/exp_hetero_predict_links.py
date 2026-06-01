import torch

from dataset_generators.data_loader_hetero import create_train_val_test_loaders
from engines.hetero_link_train import train, evaluate
from gnn_models.hetero_gnn_link_predictor import HeteroGNN, HeteroGNN_difHeads


def infer_metadata_and_in_dims(dataset):
    node_types = set()
    edge_types = set()
    in_dims = {}

    # Collect all node and edge types from the whole dataset
    for data in dataset:
        node_types.update(data.node_types)
        edge_types.update(data.edge_types)

    node_types = sorted(node_types)
    edge_types = sorted(edge_types)

    # Infer feature dimensions
    for node_type in node_types:
        dim = None

        for data in dataset:
            if node_type in data.node_types and "x" in data[node_type]:
                dim = data[node_type].x.size(-1)
                break

        in_dims[node_type] = dim if dim is not None else 0

    return (node_types, edge_types), in_dims

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        datasets_dir='../data/occidata/datasets_hetero_new/',  # <-- REPLACE
        batch_size=2,
        num_files=40
    )

    # sample = train_loader.dataset[0]
    # in_dims = {t: sample[t].x.size(1) if 'x' in sample[t] else 0 for t in sample.node_types}

    metadata, in_dims = infer_metadata_and_in_dims(train_loader.dataset)

    # TODO: change for one or dif heads
    model = HeteroGNN_difHeads(  # HeteroGNN(
        metadata,
        in_dims,
        hidden_dim=128,
        num_layers=6,
    ).to(device)

    print(model)
    src_configs = {
        # 'left_dummy': ['left_right_candidate', 'left_candidate'],
        # 'right_dummy': ['left_right_candidate', 'right_candidate'],
        'left_dummy': ['not_scheduled'],
        'right_dummy': ['not_scheduled'],
    }

    train(model, train_loader, val_loader, device, src_configs, epochs=10, threshold=0.3, save_path="../checkpoints/best_hetero_assignment_mlp_test.pt")

    model.load_state_dict(
        torch.load('../checkpoints/best_hetero_assignment_mlp_test.pt',
                   map_location=device,
                   weights_only=False)
    )

    acc, f1, r2 = evaluate(
        model,
        test_loader,
        device,
        src_configs,
        threshold=0.3,
        split_name="Test",
    )

    print("Test acc:", acc, "f1:", f1, "r2:", r2)
