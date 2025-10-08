import torch
from torch_geometric.nn import GATv2Conv

from dataset_generators.data_loader import create_train_val_test_loaders
from engines.link_prediction_train_test import train_link_prediction, evaluate_link_prediction
from gnn_models.link_prediction import SimpleLinkPredictor


def predict_links_train_test():
    datasets_dir = '../data/datasets/'
    train_loader, val_loader, test_loader = create_train_val_test_loaders(datasets_dir,
                                                                          (0.8, 0.1, 0.1),
                                                                          30)

    batch = next(iter(train_loader))
    in_dim = batch.x.shape[1]
    edge_dim = batch.edge_attr.shape[1]
    hid_dim = 16 * in_dim
    out_dim = 8 * in_dim
    num_layers = 6
    pos_weight = 2.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleLinkPredictor(conv_layer=GATv2Conv, in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
                                num_layers=num_layers, edge_dim=edge_dim).to(device)
    print(model)

    trained_model = train_link_prediction(model, train_loader, val_loader, device, epochs=100, pos_weight=pos_weight,
                                          lr=1e-3, use_dot=False)

    # Load best model
    checkpoint_path = '../checkpoints/SimpleLinkPredictor.pth'
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model = SimpleLinkPredictor(conv_layer=GATv2Conv, in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
                                     num_layers=num_layers, edge_dim=edge_dim).to(device)
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))

    # Final test evaluation
    test_loss, test_auc, test_ap = evaluate_link_prediction(best_model, test_loader, device, use_dot=False)
    print(f"\nTest Results — Loss: {test_loss:.6f} | AUC: {test_auc:.4f} | AP: {test_ap:.4f}")


if __name__ == '__main__':
    predict_links_train_test()
