import os
import math
import random
from typing import Tuple, Optional

import torch
from torch_geometric.loader import DataLoader

# =========================
# DATA LOADER
# =========================
# def create_train_val_test_loaders(
#     datasets_dir: str,
#     train_val_test_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
#     batch_size: int = 32,
#     shuffle_seed: Optional[int] = 42,
# ):
#     if not math.isclose(sum(train_val_test_ratio), 1.0):
#         raise ValueError("Ratios must sum to 1.0")
#
#     files = [f for f in os.listdir(datasets_dir) if os.path.isfile(os.path.join(datasets_dir, f))]
#     random.Random(shuffle_seed).shuffle(files)
#
#     graphs = [torch.load(os.path.join(datasets_dir, f), weights_only=False) for f in files]
#
#     n = len(graphs)
#     n_train = int(n * train_val_test_ratio[0])
#     n_val = int(n * train_val_test_ratio[1])
#
#     train_graphs = graphs[:n_train]
#     val_graphs = graphs[n_train:n_train + n_val]
#     test_graphs = graphs[n_train + n_val:]
#
#     return (
#         DataLoader(train_graphs, batch_size=batch_size, shuffle=True),
#         DataLoader(val_graphs, batch_size=batch_size),
#         DataLoader(test_graphs, batch_size=batch_size),
#     )


def create_train_val_test_loaders(
    datasets_dir: str,
    train_val_test_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    batch_size: int = 32,
    shuffle_seed: Optional[int] = 42,
    num_files: Optional[int] = None,
):
    if not math.isclose(sum(train_val_test_ratio), 1.0):
        raise ValueError("Ratios must sum to 1.0")

    files = [
        f for f in os.listdir(datasets_dir)
        if os.path.isfile(os.path.join(datasets_dir, f))
    ]

    rng = random.Random(shuffle_seed)
    rng.shuffle(files)

    # Take only a random subset of files, if requested
    if num_files is not None:
        if num_files <= 0:
            raise ValueError("num_files must be positive or None")

        files = files[:num_files]

    graphs = [
        torch.load(os.path.join(datasets_dir, f), weights_only=False)
        for f in files
    ]

    n = len(graphs)
    n_train = int(n * train_val_test_ratio[0])
    n_val = int(n * train_val_test_ratio[1])

    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:n_train + n_val]
    test_graphs = graphs[n_train + n_val:]

    return (
        DataLoader(train_graphs, batch_size=batch_size, shuffle=True),
        DataLoader(val_graphs, batch_size=batch_size),
        DataLoader(test_graphs, batch_size=batch_size),
    )
