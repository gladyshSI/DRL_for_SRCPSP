import os
import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData


def create_train_val_test_loaders(
    datasets_dir: str,
    train_val_test_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    batch_size: int = 32,
    shuffle_seed: Optional[int] = 42,
    file_filter: Optional[str] = None,  # Suffix to accept
    map_location=None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if len(train_val_test_ratio) != 3:
        raise ValueError("train_val_test_ratio must be a triple (train, val, test)")
    if not math.isclose(sum(train_val_test_ratio), 1.0, rel_tol=1e-9):
        raise ValueError("train_val_test_ratio must sum to 1.0 (use floats)")

    if not os.path.isdir(datasets_dir):
        raise ValueError(f"datasets_dir does not exist or is not a directory: {datasets_dir}")

    all_files = [
        f for f in os.listdir(datasets_dir)
        if os.path.isfile(os.path.join(datasets_dir, f))
           and (file_filter is None or f.endswith(file_filter))
    ]
    if len(all_files) == 0:
        raise ValueError(f"No files found in {datasets_dir} (filter={file_filter})")

    if shuffle_seed is not None:
        rnd = random.Random(shuffle_seed)
        rnd.shuffle(all_files)
    else:
        random.shuffle(all_files)

    graphs: list[HeteroData] = []
    for f_name in all_files:
        path = os.path.join(datasets_dir, f_name)
        obj = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(obj, HeteroData):
            raise ValueError(f"File {path} did not contain a HeteroData object.")
        graphs.append(obj)

    n = len(graphs)
    tr, vr, ter = train_val_test_ratio
    n_train = int(n * tr)
    n_val = int(n * vr)

    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:n_train + n_val]
    test_graphs = graphs[n_train + n_val:]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    print(f"Loaded {n} graphs -> train: {len(train_graphs)}, val: {len(val_graphs)}, test: {len(test_graphs)}")
    return train_loader, val_loader, test_loader
