import torch
from torch_geometric.loader import DataLoader
from os import listdir
from os.path import isfile, join
from random import shuffle

from tqdm import tqdm

from dataset_generators.solution_to_data import solution_to_slices, partial_sch_to_cropped_data_with_com_node
from lib.schedule import Schedule


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


def create_train_val_test_loaders_from_sols(solutions_dir: str,
                                            train_val_test_ratio: (float, float, float) = (0.8, 0.1, 0.1),
                                            batch_size: int = 30) -> (DataLoader, DataLoader, DataLoader):
    if sum(train_val_test_ratio) != 1.0:
        raise ValueError('sum of train_val_test_ratio must be 1.0')

    sol_names = [f for f in listdir(solutions_dir) if isfile(join(solutions_dir, f))]
    shuffle(sol_names)

    # sols = [torch.load(solutions_dir + f, weights_only=False) for f in sol_names]
    print("read solution schedules:")
    sols = [Schedule.read_from_file(solutions_dir + f) for f in tqdm(sol_names)]

    # Split into train/val/test
    train_r = train_val_test_ratio[0]
    val_r = train_val_test_ratio[1]
    train_sols = sols[:int(train_r * len(sols))]
    val_sols = sols[int(train_r * len(sols)):int((train_r + val_r) * len(sols))]
    test_sols = sols[int((train_r + val_r) * len(sols)):]

    def make_data_from_sol(sol, part_sch_to_data_f=partial_sch_to_cropped_data_with_com_node):
        slices = solution_to_slices(sol)
        data_list = []
        for i in range(len(slices) - 1):
            data = part_sch_to_data_f(slices[i], slices[i + 1])
            data_list.append(data)
        return data_list

    print("make train data:")
    train_data = [item for sol in tqdm(train_sols) for item in make_data_from_sol(sol)]
    print("make val data:")
    val_data = [item for sol in tqdm(val_sols) for item in make_data_from_sol(sol)]
    print("make test data:")
    test_data = [item for sol in tqdm(test_sols) for item in make_data_from_sol(sol)]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    print("Data loaded")
    return train_loader, val_loader, test_loader



