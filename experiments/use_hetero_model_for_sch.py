import copy
import random
from itertools import product
import numpy as np
import torch
import matplotlib
import torch.nn.functional as F

from dataset_generators.data_loader_hetero import create_train_val_test_loaders
from dataset_generators.solution_to_heterogeneous_data import round_slice_to_node_ids_mapping, \
    round_partial_sch_to_data
from experiments.exp_hetero_predict_links import infer_metadata_and_in_dims, evaluate

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from torch_geometric.config_store import Model
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GATv2Conv
import torch_geometric.transforms as T
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from dataset_generators.cp_problem_solving import rand_sgs, rand_sgs_best_of_k
from dataset_generators.occidata_to_schedule import read_jobs, read_graph_from_txt, make_schedule_from_str, \
    make_df_from_all_csv_files, filter_dataframe, occidata_graph_file_path_to_ours, occidata_jobs_file_path_to_ours
from gnn_models.link_prediction import SimpleLinkPredictor
from gnn_models.link_value_prediction import SimpleLinkValuePredictor
from lib.distribution import DiscreteDistribution, Distribution
from lib.graph import PrecedenceGraph
from lib.problem import Problem, get_longest_paths
from lib.schedule import Schedule, draw_schedule
from gnn_models.hetero_gnn_link_predictor import HeteroGNN, HeteroGNN_difHeads


def get_probs_values_for_edge_type(model, data, map_location='cpu') -> list[tuple[str, int, str, int, float, float]]:
    '''
    :return: [(src_type, fr_id, dst_type, to_id, logit, predicted buf), ...]
    '''
    result = []
    model.eval()
    with torch.no_grad():
        data = data.to(map_location)
        out = model(data)
        predict_edge_types = [et for et in data.edge_types if et[1] == 'predict']
        for edge_type in predict_edge_types:
            src_type, _, dst_type = edge_type

            eidx = data[edge_type].edge_label_index
            src_idx = eidx[0]  # shape [E]
            dst_idx = eidx[1]  # shape [E]

            src = out[src_type][eidx[0]]
            dst = out[dst_type][eidx[1]]

            pair = torch.cat([src, dst], dim=-1)

            # logits = model.link_scorer(pair).squeeze(-1)
            #
            # # Non-negative length prediction
            # pred_buf = F.softplus(model.buf_scorer(pair).squeeze(-1))

            key = model.edge_type_to_str(edge_type)

            if key not in model.link_scorers:
                raise KeyError(
                    f"No relation-specific scorer for edge_type={edge_type}. "
                    f"Available: {list(model.link_scorers.keys())}"
                )

            logits = model.link_scorers[key](pair).squeeze(-1)

            pred_buf = F.softplus(
                model.buf_scorers[key](pair).squeeze(-1)
            )

            result.extend([(src_type, s.item(), dst_type, d.item(), l.item(), b.item()) for s, d, l, b
                           in zip(src_idx, dst_idx, logits, pred_buf)])
    return result


def greedy_choose_to_schedule(edges_probs: list[tuple[str, int, str, int, float, float]]) -> list[tuple[str, int, str, int, float]]:
    edges_to_schedule = []
    # edges_probs = [_ for _ in edges_probs if (_[0] == 'left_dummy') or (_[0] == 'right_dummy' and _[4] >= 0.00)]
    # print([_ for _ in edges_probs if (_[0] == 'right_dummy')])
    while len(edges_probs) > 0:
        max_t = max(edges_probs, key=lambda x: x[4])
        fr_t, fr_id, to_t, to_id, value, buf = max_t
        edges_to_schedule.append((fr_t, fr_id, to_t, to_id, buf))
        edges_probs = [t for t in edges_probs if (t[0], t[1]) != (fr_t, fr_id) and (t[2], t[3]) != (to_t, to_id)]
    return edges_to_schedule

# TODO: Doesn't work so far
def hungarian_choose_to_schedule(edges_probs: list[tuple[str, int, str, int, float]]) -> list[tuple[str, int, str, int]]:
    rows = list(dict.fromkeys((fr_t, fr_id) for fr_t, fr_id, _, _, _ in edges_probs))
    cols = list(dict.fromkeys((to_t, to_id) for _, _, to_t, to_id, _ in edges_probs))
    r_idx = {v: i for i, v in enumerate(rows)}
    c_idx = {v: i for i, v in enumerate(cols)}

    probs = np.zeros((len(rows), len(cols)))
    for fr_t, fr_id, to_t, to_id, p in edges_probs:
        i, j = r_idx[(fr_t, fr_id)], c_idx[(to_t, to_id)]
        probs[i, j] = p

    # use maximize=True (SciPy >= 1.4.0). If older SciPy, use linear_sum_assignment(-probs)
    row_inds, col_inds = linear_sum_assignment(-probs)
    return [(str(rows[i][0]), int(rows[i][1]), str(cols[j][0]), int(cols[j][1])) for i, j in zip(row_inds, col_inds)]


def schedule_edges_round_slice(round_slice: tuple[Schedule, Schedule], edges_to_sch: list[tuple[str, int, str, int, float]]) -> None:
    _, gnn_id_to_pr_id = round_slice_to_node_ids_mapping(round_slice)
    for fr_t, fr_id, to_t, to_id, buf in edges_to_sch:
        if fr_t not in {'left_dummy', 'right_dummy'}:
            raise ValueError(f'fr_t {fr_t} is not correct: should be "left_dummy" or "right_dummy"')
        if to_t not in {'not_scheduled'}:
            raise ValueError(f'to_t {to_t} is not correct: should be "not_scheduled"')

        sch = round_slice[0] if fr_t == 'left_dummy' else round_slice[1]
        w_id = fr_id
        dst = gnn_id_to_pr_id[to_t][to_id]
        w_exec_order = sch.get_execution_orders()[w_id]
        prev_end_t = 0 if len(w_exec_order) == 0 else sch.get_scheduled_end_time(w_exec_order[-1])
        buf = max(0, round(buf))
        # st_time = max(sch.get_first_possible_st_t(worker_id=w_id, job_id=dst), prev_end_t + buf)  # TODO: this is better ???
        st_time = sch.get_first_possible_st_t(worker_id=w_id, job_id=dst) + buf
        sch.schedule_job(worker_id=w_id, job_id=dst, start_time=st_time)


def calculate_agg_f_ovp_dist_e(agg_f, ovp_dist: dict[int, Distribution]):
    return agg_f([x.e() for _, x in ovp_dist.items()])


def get_metrics(sch: Schedule) -> dict[str, float]:
    dist = sch.calculate_exact_overlap_distributions()
    return {'makespan': sch.get_makespan(),
            'sum_obj': calculate_agg_f_ovp_dist_e(sum, dist),
            'max_obj': calculate_agg_f_ovp_dist_e(max, dist)}


def get_final_sch_from_r_slice(round_slice: tuple[Schedule, Schedule]) -> Schedule:
    # draw_schedule(round_slice[0])
    # draw_schedule(round_slice[1])

    # find the min possible start time for the right schedule:
    def get_f_pos_st_t(sched_w_id, job_id):
        r_last_jobs = round_slice[1].get_last_jobs()
        if job_id in round_slice[0].get_candidates() & r_last_jobs:
            return round_slice[0].get_first_possible_st_t(worker_id=sched_w_id, job_id=job_id)
        else:
            predecessors = round_slice[0].get_problem().graph.get_predecessors(job_id)
            sched_l = round_slice[0].get_scheduled()
            last_end_t = 0
            if job_id in r_last_jobs:
                order = round_slice[0].get_execution_orders()[sched_w_id]
                last_cor_j = order[-1] if order else None
                if last_cor_j is not None:
                    last_end_t = round_slice[0].get_scheduled_end_time(last_cor_j)
            return max([round_slice[0].get_scheduled_end_time(i) for i in sched_l & predecessors] + [last_end_t])

    r_makespan = round_slice[1].get_makespan()
    min_st_t = -1  # where we will stick the right makespan
    for job_id in round_slice[1].get_scheduled():
        sched_w_id = round_slice[1].get_scheduled_worker(job_id)
        end_t = round_slice[1].get_scheduled_end_time(job_id)
        delta = r_makespan - end_t
        f_pos_st_t = get_f_pos_st_t(sched_w_id=sched_w_id, job_id=job_id)
        min_st_t = max(min_st_t, f_pos_st_t - delta)


    ########################

    f_sch = copy.deepcopy(round_slice[0])
    n_workers = f_sch.get_problem().n_workers
    r_exec_orders = round_slice[1].get_execution_orders()

    max_ids = [len(order)-1 for order in r_exec_orders]
    max_id = max(max_ids)
    ids = [len(order)-1 for order in r_exec_orders]
    for it in range(max_id+1):
        for w_id in range(n_workers):
            if ids[w_id] < 0:
                continue
            next_j = r_exec_orders[w_id][ids[w_id]]
            if next_j in f_sch.get_candidates():
                buf = 0
                if ids[w_id] < max_ids[w_id]:  # We are inside the right schedule
                    # Calculate the buffer time
                    previous_j = r_exec_orders[w_id][ids[w_id]+1]
                    prev_st_t = round_slice[1].get_scheduled_start_time(previous_j)
                    end_t = round_slice[1].get_scheduled_end_time(next_j)
                    buf = prev_st_t - end_t
                    # print(f'previous_j: {previous_j}, prev_st_t {prev_st_t}, next_j: {next_j}, end_t {end_t}, buf {buf}')
                last_sched_j = f_sch.get_execution_orders()[w_id][-1]
                last_end_t = f_sch.get_scheduled_end_time(last_sched_j)
                f_pos_st_t = f_sch.get_first_possible_st_t(worker_id=w_id, job_id=next_j)
                # print(f'w_id: {w_id}, last scheduled_j {last_sched_j} with end_t {last_end_t}; next_j: {next_j} with f_pos_st_t {f_pos_st_t} and buf {buf}')

                # WITH SLIDE TO THE LEFT AND WITHOUT BUFFERS
                # f_sch.schedule_job(job_id=next_j, worker_id=w_id, start_time=f_pos_st_t)  # + buf)

                # JUST SCHEDULE SYMMETRICALLY THE RIGHT SIDE
                new_st_t = min_st_t + r_makespan - round_slice[1].get_scheduled_end_time(next_j)
                f_sch.schedule_job(job_id=next_j, worker_id=w_id, start_time=new_st_t)
                ids[w_id] -= 1
    return f_sch


def solve_problem_with_hetero_nn(model, problem: Problem, map_location: str = 'cpu', mode: str = 'greedy') -> Schedule:
    round_slice = (Schedule(problem=problem), Schedule(problem=problem.reverse()))
    n_jobs = problem.n_jobs
    n_workers = problem.n_workers
    round_slice[0].schedule_job(worker_id=0, job_id=0)
    if n_jobs == 1:
        return round_slice[0]
    round_slice[1].schedule_job(worker_id=n_workers-1, job_id=n_jobs-1)

    all_jobs = set(range(problem.n_jobs))
    while round_slice[0].get_scheduled() | round_slice[1].get_scheduled() < all_jobs:
        data = round_partial_sch_to_data(round_slice_before=round_slice)
        probs_values = get_probs_values_for_edge_type(model, data, map_location=map_location)
        if mode == 'greedy':
            edges_to_schedule = greedy_choose_to_schedule(probs_values)
        elif mode == 'hungarian':  # TODO: Doesn't work so far
            edges_to_schedule = hungarian_choose_to_schedule(probs_values)
        else:
            raise ValueError("mode must be 'greedy' or 'hungarian'")
        schedule_edges_round_slice(round_slice, edges_to_schedule)

    final_sch = get_final_sch_from_r_slice(round_slice)
    return final_sch


def get_problem_data_example():
    # df = make_df_from_all_csv_files('../data/occidata/outputs_from400')
    df = make_df_from_all_csv_files('../data/occidata/outputs_200_400')
    include = {'distribution': ['uniform'], 'name': ['BBr'], 'jobs_num': [402]}
    df_filtered = filter_dataframe(df, include=include, exclude={}).drop_duplicates(subset=['name', 'jobs_f'])
    graph_f, jobs_f, n_workers, sch_str = df_filtered.iloc[1][["graph_f", "jobs_f", "workers_num", "schedule"]]

    jobs = read_jobs(occidata_jobs_file_path_to_ours(jobs_f))
    graph = read_graph_from_txt(occidata_graph_file_path_to_ours(graph_f))
    problem = Problem(n_workers=n_workers, n_jobs=len(jobs), graph=graph, jobs=jobs)
    rev_problem = problem.reverse()

    sch = Schedule(problem=problem)
    make_schedule_from_str(sch, sch_str)
    print("BBr", get_metrics(sch))
    draw_schedule(sch)

    left_r, right_r = Schedule(problem=problem), Schedule(problem=rev_problem)
    left_r.schedule_job(worker_id=0, job_id=0)
    right_r.schedule_job(worker_id=n_workers - 1, job_id=len(jobs) - 1)

    round_slice = (left_r, right_r)
    data = round_partial_sch_to_data(round_slice_before=round_slice)
    return problem, data


def load_best_model(train_loader, map_location):
    device = torch.device(map_location)
    # checkpoint = '../checkpoints/best_hetero_assignment_mlp_f1032.pt'
    # checkpoint = '../checkpoints/best_hetero_assignment_mlp_with_buf_f1032.pt'
    checkpoint = '../checkpoints/best_hetero_assignment_mlp_buf_dif_heads_f1033.pt'

    metadata, in_dims = infer_metadata_and_in_dims(train_loader.dataset)

    # TODO: change for one or dif heads
    best_model = HeteroGNN_difHeads(
        metadata,
        in_dims,
        hidden_dim=128,
        num_layers=6,
    ).to(device)

    print(best_model)

    best_model.load_state_dict(
        torch.load(checkpoint,
                   map_location=device,
                   weights_only=False)
    )

    return best_model

def one_problem_try():
    problem, data = get_problem_data_example()
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        datasets_dir='../data/occidata/datasets_hetero_new/',  # <-- REPLACE
        batch_size=32,
        num_files=None
    )

    best_model = load_best_model(train_loader=train_loader, map_location=map_location)

    ###############################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_configs = {
        # 'left_dummy': ['left_right_candidate', 'left_candidate'],
        # 'right_dummy': ['left_right_candidate', 'right_candidate'],
        'left_dummy': ['not_scheduled'],
        'right_dummy': ['not_scheduled'],
    }
    acc, f1, r2 = evaluate(
        best_model,
        test_loader,
        device,
        src_configs,
        threshold=0.,
        split_name="Test",
    )
    print("Test acc:", acc, "f1:", f1, "r2:", r2)
    #############################


    sch = solve_problem_with_hetero_nn(best_model, problem, map_location=map_location, mode='greedy')
    print("greedy", get_metrics(sch))
    draw_schedule(sch)
    # sch = solve_problem_with_hetero_nn(best_model, problem, map_location=map_location, mode='hungarian')
    # print("hungarian", get_metrics(sch))
    # draw_schedule(sch)


def main():
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, _, _ = create_train_val_test_loaders(
        datasets_dir='../data/occidata/datasets_hetero_new/',  # <-- REPLACE
        batch_size=128,
        num_files=None
    )

    best_model = load_best_model(train_loader=train_loader, map_location=map_location)

    print("COLLECT CP SOLUTIONS")
    # df = make_df_from_all_csv_files('../data/occidata/outputs')
    df = make_df_from_all_csv_files('../data/occidata/outputs_from400')
    # df = make_df_from_all_csv_files('../data/occidata/outputs_200_400')
    include = {'distribution': ['uniform'], 'name': ['DET', 'STs2', 'BBr'], 'jobs_num': [62]}
    df_filtered = filter_dataframe(df, include=include, exclude={}).drop_duplicates(subset=['name', 'jobs_f'])
    print(df_filtered)
    print("ITERATE THROUGH CP SOLUTIONS, LOAD SCHEDULES")
    jobs_f_to_problem = dict()
    jobs_f_name_to_sch = dict()
    for name, graph_f, jobs_f, n_workers, sch_str in tqdm(zip(df_filtered['name'],
                                                              df_filtered['graph_f'],
                                                              df_filtered['jobs_f'],
                                                              df_filtered['workers_num'],
                                                              df_filtered['schedule'])):
        if jobs_f not in jobs_f_to_problem.keys():
            graph_path = occidata_graph_file_path_to_ours(graph_f)
            jobs_path = occidata_jobs_file_path_to_ours(jobs_f)
            jobs = read_jobs(jobs_path)
            graph = read_graph_from_txt(graph_path)
            problem = Problem(n_workers=n_workers, n_jobs=len(jobs), graph=graph, jobs=jobs)
            jobs_f_to_problem[jobs_f] = problem
        problem = jobs_f_to_problem[jobs_f]
        sch = Schedule(problem=problem)
        make_schedule_from_str(sch, sch_str)

        if jobs_f not in jobs_f_name_to_sch.keys():
            jobs_f_name_to_sch[jobs_f] = dict()
        jobs_f_name_to_sch[jobs_f][name] = sch

    print("ITERATE THROUGH PROBLEMS AND SOLVE THEM WITH MODEL:")
    # part_sch_to_data_f = partial_sch_to_cropped_data
    round_slice_to_data = round_partial_sch_to_data
    for jobs_f, problem in tqdm(jobs_f_to_problem.items()):
        sch_greedy = solve_problem_with_hetero_nn(best_model, problem, map_location=map_location, mode='greedy')
        jobs_f_name_to_sch[jobs_f]["NN_greedy"] = sch_greedy
        # sch_hungarian = solve_problem_with_cropped_nn(best_model, problem, part_sch_to_data_f, map_location=map_location, mode='hungarian')
        # jobs_f_name_to_sch[jobs_f]["NN_hungarian"] = sch_hungarian
        # sch_nn_random = solve_problem_with_cropped_nn(best_model, problem, part_sch_to_data_f, map_location=map_location, mode='random')
        # jobs_f_name_to_sch[jobs_f]["NN_random"] = sch_nn_random
        # sch_sgs_random = rand_sgs_best_of_k(problem, k=20)
        # jobs_f_name_to_sch[jobs_f]["SGS_rand"] = sch_sgs_random

    print("COLLECT ALL METRICS:")
    name_to_metrics = dict()  # name -> metric -> list[values]
    for jobs_f, name_to_sch in tqdm(jobs_f_name_to_sch.items()):
        for name, sch in name_to_sch.items():
            metrics = get_metrics(sch)

            if name not in name_to_metrics.keys():
                name_to_metrics[name] = dict()

            for metric_name, metric_val in metrics.items():
                if metric_name not in name_to_metrics[name].keys():
                    name_to_metrics[name][metric_name] = []
                name_to_metrics[name][metric_name].append(metric_val)

    print("PRINT METRIC AVERAGES:")
    for name, metrics in name_to_metrics.items():
        for metric_name, metric_vals in metrics.items():
            print(f'{name}: {metric_name}: avg = {np.mean(metric_vals)}, std = {np.std(metric_vals)}')
    # draw_box_plots(name_to_metrics, 'makespan', labelsize=20, fontsize=11)
    # draw_box_plots(name_to_metrics, 'sum_obj', labelsize=20, fontsize=11)
    # draw_box_plots(name_to_metrics, 'max_obj', labelsize=20, fontsize=11)


def draw_box_plots(name_to_metric: dict[str, dict[str, list]], metric_name: str, labelsize: int=20, fontsize: int=20, ):
    # model_name -> metric name -> [metric values]
    names = list(name_to_metric.keys())
    print('names', names)
    instances_num = len(name_to_metric[names[0]][metric_name])
    print('instances num', instances_num)

    min_metric_for_each_instance = [min([name_to_metric[m_name][metric_name][i] for m_name in names])
                                    for i in range(instances_num)]
    print('len min metric', len(min_metric_for_each_instance))

    delta_with_min = [[name_to_metric[model_name][metric_name][i] - min_metric_for_each_instance[i] for i in range(instances_num)]
                      for model_name in names]
    fig, ax = plt.subplots()
    # ax.set_ylabel(f'{metric_name} (Δ with min.)', fontsize=fontsize//2)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.boxplot(delta_with_min,  tick_labels=names)
    ax.set_title(f'{metric_name} (Δ with min.)', fontsize=fontsize)

    plt.show()


if __name__ == '__main__':
    # main()
    one_problem_try()
    # one_problem_try(mode="links_values")
