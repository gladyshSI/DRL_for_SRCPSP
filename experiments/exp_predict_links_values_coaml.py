import torch
from torch_geometric.nn import GATv2Conv

from dataset_generators.data_loader import create_train_val_test_loaders, create_train_val_test_loaders_from_sols
from engines.link_value_train_coaml import train_link_value_coaml_prediction, validate_diagnostics

from gnn_models.link_value_prediction import SimpleLinkValuePredictor


def predict_links_values_coaml_train_test():
    solutions_dir = '../data/occidata/solutions/'
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_train_val_test_loaders_from_sols(solutions_dir,
                                                                                    (0.8, 0.1, 0.1),
                                                                                    120)
    print("DONE!")

    it = iter(train_loader)
    batch = next(it)
    in_dim = batch.x.shape[1]
    edge_dim = batch.edge_attr.shape[1]
    hid_dim = 16 * in_dim
    out_dim = 8 * in_dim
    num_layers = 6
    pos_weight = 13.  # None
    heads = 2

    print(
        f'SHAPES\nx: {batch.x.shape}, edge_index: {batch.edge_index.shape}, edge_attr: {batch.edge_attr.shape}, edge_predict_index {batch.edge_predict_index.shape}, edge_predict_label: {batch.edge_predict_label.shape}, edge_predict_value: {batch.edge_predict_value.shape}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleLinkValuePredictor(conv_layer=GATv2Conv, in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
                                     num_layers=num_layers, edge_dim=edge_dim, heads=heads).to(device)
    print(model)

    # In engines/:
    reg_loss_weight = 0.01
    train_link_value_coaml_prediction(
        model,
        train_loader,
        val_loader,
        device,
        epochs=3,
        lr=1e-3,
        reg_loss_weight=reg_loss_weight
    )

    # Load best model
    checkpoint_path = '../checkpoints/SimpleLinkValuePredictor_Coaml.pth'
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model = SimpleLinkValuePredictor(conv_layer=GATv2Conv, in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
                                          num_layers=num_layers, edge_dim=edge_dim, heads=heads).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False
    )

    best_model.load_state_dict(
        checkpoint['model_state_dict']
    )

    # Final test evaluation
    reg_loss = torch.nn.MSELoss(reduction='mean')  # you can change to L1 if you prefer

    metrics = validate_diagnostics(
        best_model,
        test_loader,
        device,
        reg_loss,
        reg_loss_weight
    )

    print(f"\nTest", metrics.items())


if __name__ == '__main__':
    predict_links_values_coaml_train_test()
