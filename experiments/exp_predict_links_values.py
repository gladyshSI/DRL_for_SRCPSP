import torch
from torch_geometric.nn import GATv2Conv

from dataset_generators.data_loader import create_train_val_test_loaders
from engines.link_value_train import train_link_value_prediction, validate_link_value_prediction
from gnn_models.link_value_prediction import SimpleLinkValuePredictor


def predict_links_values_train_test():
    # datasets_dir = '../data/datasets_with_length/'
    datasets_dir = '../data/occidata/datasets/'
    train_loader, val_loader, test_loader = create_train_val_test_loaders(datasets_dir,
                                                                          (0.8, 0.1, 0.1),
                                                                          120)

    batch = next(iter(train_loader))
    in_dim = batch.x.shape[1]
    edge_dim = batch.edge_attr.shape[1]
    hid_dim = 16 * in_dim
    out_dim = 8 * in_dim
    num_layers = 6
    pos_weight = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleLinkValuePredictor(conv_layer=GATv2Conv, in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
                                     num_layers=num_layers, edge_dim=edge_dim).to(device)
    print(model)

    # In engines/:
    reg_loss_weight = 0.01
    trained_model = train_link_value_prediction(model, train_loader, val_loader, device, epochs=500,
                                                pos_weight=pos_weight, lr=1e-3, reg_loss_weight=reg_loss_weight)

    # Load best model
    checkpoint_path = '../checkpoints/SimpleLinkValuePredictor.pth'
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model = SimpleLinkValuePredictor(conv_layer=GATv2Conv, in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
                                          num_layers=num_layers, edge_dim=edge_dim).to(device)
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))

    # Final test evaluation
    cls_loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
    reg_loss = torch.nn.MSELoss(reduction='mean')  # you can change to L1 if you prefer
    metrics = validate_link_value_prediction(best_model, test_loader, device,
                                                                         cls_loss,
                                                                         reg_loss,
                                                                         reg_loss_weight)
    print(f"\nTest", metrics.items())


if __name__ == '__main__':
    predict_links_values_train_test()
