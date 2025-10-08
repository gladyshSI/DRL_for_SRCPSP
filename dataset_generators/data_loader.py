import torch
from torch_geometric.loader import DataLoader
from os import listdir
from os.path import isfile, join
from random import shuffle


def create_train_val_test_loaders(datasets_dir: str,
                                  train_val_test_ratio: (float, float, float) = (0.8, 0.1, 0.1),
                                  batch_size: int = 30) -> (DataLoader, DataLoader, DataLoader):
    if sum(train_val_test_ratio) != 1.0:
        raise ValueError('sum of train_val_test_ratio must be 1.0')

    data_names = [f for f in listdir(datasets_dir) if isfile(join(datasets_dir, f))]
    shuffle(data_names)

    graphs = [torch.load(datasets_dir + f, weights_only=False) for f in data_names]

    # Split into train/val/test
    train_r = train_val_test_ratio[0]
    val_r = train_val_test_ratio[1]
    train_graphs = graphs[:int(train_r * len(graphs))]
    val_graphs = graphs[int(train_r * len(graphs)):int((train_r + val_r) * len(graphs))]
    test_graphs = graphs[int((train_r + val_r) * len(graphs)):]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)
    print("Data loaded")
    return train_loader, val_loader, test_loader
