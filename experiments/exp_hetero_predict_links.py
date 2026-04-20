import torch

from dataset_generators.data_loader_hetero import create_train_val_test_loaders
from engines.hetero_link_train import train_loop, evaluate
from gnn_models.hetero_gnn_link_predictor import HeteroGNN

if __name__ == '__main__':
    # point this to the directory containing your saved HeteroData graph files
    datasets_dir = '../data/occidata/datasets_hetero_new/'   # <-- REPLACE

    # create loaders
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        datasets_dir=datasets_dir,
        train_val_test_ratio=(0.8, 0.1, 0.1),
        batch_size=128,
        shuffle_seed=42,
        file_filter=None,
        map_location='cpu',
        num_workers=0,
        pin_memory=False,
    )

    # sample to infer in_dims
    sample = train_loader.dataset[0]
    in_dims = {}
    for t in sample.node_types:
        if 'x' in sample[t]:
            in_dims[t] = sample[t].x.size(1)
        else:
            in_dims[t] = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroGNN(sample.metadata(), in_dims=in_dims, hidden_dim=128, num_layers=6, scorer_hidden=128).to(device)
    print(model)
    src_configs = {
        # 'left_dummy': ['left_right_candidate', 'left_candidate'],
        # 'right_dummy': ['left_right_candidate', 'right_candidate'],
        'left_dummy': ['not_scheduled'],
        'right_dummy': ['not_scheduled'],
    }

    train_loop(model, train_loader, val_loader, device, epochs=30, lr=1e-3, src_configs=src_configs,
               save_path='../checkpoints/best_hetero_assignment_mlp.pt')

    # load best and evaluate on test
    model.load_state_dict(torch.load('../checkpoints/best_hetero_assignment_mlp.pt', map_location=device))
    test_acc = evaluate(model, test_loader, device, src_configs)
    print("Test acc:", test_acc)
