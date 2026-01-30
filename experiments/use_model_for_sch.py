import copy
import random

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from dataset_generators.cp_problem_solving import rand_sgs
from dataset_generators.occidata_to_schedule import read_jobs, read_graph_from_txt, make_schedule_from_str, \
    make_df_from_all_csv_files, filter_dataframe, occidata_graph_file_path_to_ours, occidata_jobs_file_path_to_ours
from gnn_models.link_prediction import SimpleLinkPredictor
from lib.distribution import DiscreteDistribution, Distribution
from lib.graph import PrecedenceGraph
from lib.problem import Problem
from lib.schedule import Schedule, draw_schedule


def part_sch_to_data(part_sch: Schedule) -> Data:
    """
    Node features:
    data.x → [N, F] float
        F = [Scheduled: bool; Last performed: bool; Candidate: bool; Worker ID + 1: int;
        Duration distribution: (min, max, avg); Competition time distribution approximation: (min, max, avg)]

    Graph structure (observed edges used by the encoder):
    data.edge_index → [2, M] long
    data.edge_attr → [M, Fe] float (optional)
        Fe = [precedence: bool, exec_order: bool, reverse precedence: bool, reverse_exec_order: bool]

    Pairs for link classification (positive + negatives):
    data.edges_eval_index → [2, P] long containing all possible edges from last scheduled to candidates
    """
    problem = part_sch.get_problem()
    # Candidates on the previous step:
    scheduled = part_sch.get_scheduled()
    candidates = part_sch.get_candidates()

    # Calculate exact end time distributions
    end_t_distributions = part_sch.calculate_exact_end_time_distributions()

    # Precedence relation edges
    precedence_fr_ids = []
    precedence_to_ids = []
    for fr_id, to_ids in problem.graph.get_copy_of_all_edges().items():
        precedence_fr_ids.extend(fr_id for _ in to_ids)
        precedence_to_ids.extend(to_ids)

    # Execution order edges
    execution_order_fr_ids = []
    execution_order_to_ids = []
    for w_exec_order in part_sch.get_execution_orders():
        for i in range(len(w_exec_order) - 1):
            execution_order_fr_ids.append(w_exec_order[i])
            execution_order_to_ids.append(w_exec_order[i + 1])

    # Make dummy jobs at the end of each workers' schedule.
    # dummy job for worker w_i has id = max_job_id + 1 + w_i
    for w_id, w_exec_order in enumerate(part_sch.get_execution_orders()):
        if len(w_exec_order) >= 1:
            execution_order_fr_ids.append(w_exec_order[-1])
            execution_order_to_ids.append(problem.n_jobs + w_id)

    # Predict edges from dummy job to the last scheduled jobs
    possible_edges_from_ids = []
    possible_edges_to_ids = []
    for fr_id in [problem.n_jobs + w_id for w_id in range(problem.n_workers)]:
        for to_ids in candidates:
            possible_edges_from_ids.append(fr_id)
            possible_edges_to_ids.append(to_ids)


    w_id_to_last_ct_distribution = dict()
    # Vertex attributes:
    v_attributes = []
    for j_id in range(problem.n_jobs):
        is_scheduled = 1 if j_id in scheduled else 0
        last_performed = 0
        candidate = 1 if j_id in candidates else 0
        w_id = 0 if j_id not in scheduled else part_sch.get_scheduled_worker(j_id) + 1  # 0 is for None

        min_dur = problem.jobs[j_id].get_distribution().min_v()
        max_dur = problem.jobs[j_id].get_distribution().max_v()
        e_dur = problem.jobs[j_id].get_distribution().e()
        min_ct, max_ct, e_ct = 0, 0, 0
        if is_scheduled == 1:
            min_ct = end_t_distributions[j_id].min_v()
            max_ct = end_t_distributions[j_id].max_v()
            e_ct = end_t_distributions[j_id].e()

        # save ct distribution if it is prev last performed
        if w_id > 0:
            w_exec_order = part_sch.get_execution_orders()[w_id - 1]
            if len(w_exec_order) >= 1 and w_exec_order[-1] == j_id:
                w_id_to_last_ct_distribution[w_id - 1] = end_t_distributions[j_id]

        v_i_attr = [is_scheduled, last_performed, candidate, w_id, min_dur, max_dur, e_dur, min_ct, max_ct, e_ct]
        v_attributes.append(v_i_attr)
    # additional dummy jobs at the end of each workers' schedule
    for w_id in range(problem.n_workers):
        last_ct_distribution = w_id_to_last_ct_distribution[w_id] if w_id in w_id_to_last_ct_distribution.keys() \
            else DiscreteDistribution(np.array([0]), np.array([1.]))
        v_i_attr = [0, 1, 0, w_id + 1, 0, 0, 0, last_ct_distribution.min_v(), last_ct_distribution.max_v(), last_ct_distribution.e()]
        v_attributes.append(v_i_attr)

    total_fr_ids = precedence_fr_ids + execution_order_fr_ids + precedence_to_ids + execution_order_to_ids
    total_to_ids = precedence_to_ids + execution_order_to_ids + precedence_fr_ids + execution_order_fr_ids
    prec_edge_atr = [[1, 0, 0, 0] for _ in range(len(precedence_fr_ids))]
    exec_order_edge_atr = [[0, 1, 0, 0] for _ in range(len(execution_order_fr_ids))]
    rev_prec_edge_atr = [[0, 0, 1, 0] for _ in range(len(precedence_to_ids))]
    rev_exec_order_edge_atr = [[0, 0, 0, 1] for _ in range(len(execution_order_to_ids))]
    total_edge_atr = prec_edge_atr + exec_order_edge_atr + rev_prec_edge_atr + rev_exec_order_edge_atr
    # Combining into data object:
    data = Data(
        x=torch.FloatTensor(v_attributes),  # [N, F]
        edge_index=torch.LongTensor([total_fr_ids, total_to_ids]),  # [2, M_train]
        edge_attr=torch.FloatTensor(total_edge_atr),  # [M_train, Fe] optional
        edges_eval_index=torch.LongTensor([possible_edges_from_ids, possible_edges_to_ids])
    )
    return data


def get_probs(model, data, map_location='cpu') -> list[tuple[int, int, float]]:
    model.eval()
    with torch.no_grad():
        data = data.to(map_location)
        z = model(data.x, data.edge_index, edge_attr=getattr(data, 'edge_attr', None))

        src_idx = data.edges_eval_index[0]  # shape [E]
        dst_idx = data.edges_eval_index[1]  # shape [E]

        src_emb = z[src_idx]  # shape [E, embedding_dim]
        dst_emb = z[dst_idx]  # shape [E, embedding_dim]

        logits = model.score_edges(src_emb, dst_emb)  # shape [E]
        probs = torch.sigmoid(logits)  # if you want probabilities in [0,1]

    probs = probs.cpu().numpy()
    src_cpu = src_idx.cpu().numpy()
    dst_cpu = dst_idx.cpu().numpy()
    results = [(int(u), int(v), float(p)) for u, v, p in zip(src_cpu, dst_cpu, probs)]
    return results


def greedy_choose_to_schedule(edges_probs: list[tuple[int, int, float]]):
    edges_to_schedule = []
    while len(edges_probs) > 0:
        max_t = max(edges_probs, key=lambda x: x[2])
        fr_id, to_id = max_t[0], max_t[1]
        edges_to_schedule.append((fr_id, to_id))
        edges_probs = [t for t in edges_probs if t[0] != fr_id and t[1] != to_id]
    return edges_to_schedule


def hungarian_choose_to_schedule(edges_probs: list[tuple[int, int, float]]):
    rows = list(dict.fromkeys(fr for fr, _, _ in edges_probs))
    cols = list(dict.fromkeys(to for _, to, _ in edges_probs))
    r_idx = {v: i for i, v in enumerate(rows)}
    c_idx = {v: i for i, v in enumerate(cols)}

    probs = np.zeros((len(rows), len(cols)))
    for fr, to, p in edges_probs:
        i, j = r_idx[fr], c_idx[to]
        probs[i, j] = p

    # use maximize=True (SciPy >= 1.4.0). If older SciPy, use linear_sum_assignment(-probs)
    row_inds, col_inds = linear_sum_assignment(-probs)
    return [(rows[i], cols[j]) for i, j in zip(row_inds, col_inds)]


def random_choose_to_schedule(edges_probs: list[tuple[int, int, float]]):
    edges_to_schedule = []
    while len(edges_probs) > 0:
        items, weights = zip(*[((fr_id, to_id), p) for fr_id, to_id, p in edges_probs])
        choice = random.choices(items, weights=weights, k=1)[0]
        edges_to_schedule.append(choice)
        edges_probs = [t for t in edges_probs if t[0] != choice[0] and t[1] != choice[1]]
    return edges_to_schedule


def schedule_edges(sch: Schedule, edges_to_sch: list[tuple[int, int]]):
    n_jobs = sch.get_problem().n_jobs
    for fr_id, to_id in edges_to_sch:
        w_id = fr_id - n_jobs
        sch.schedule_job(worker_id=w_id, job_id=to_id)


def calculate_agg_f_ovp_dist_e(agg_f, ovp_dist: dict[int, Distribution]):
    return agg_f([x.e() for _, x in ovp_dist.items()])


def get_metrics(sch: Schedule) -> dict[str, float]:
    dist = sch.calculate_exact_overlap_distributions()
    return {'makespan': sch.get_makespan(),
            'sum_obj': calculate_agg_f_ovp_dist_e(sum, dist),
            'max_obj': calculate_agg_f_ovp_dist_e(max, dist)}


def solve_problem_with_nn(model, problem: Problem, map_location: str = 'cpu', mode: str = 'greedy') -> Schedule:
    sch = Schedule(problem=problem)
    sch.schedule_job(worker_id=0, job_id=0)
    while len(sch.get_candidates()) > 0:
        data = part_sch_to_data(sch)
        probs = get_probs(model, data, map_location=map_location)

        if mode == 'greedy':
            variants_of_assignment = [greedy_choose_to_schedule(probs)]
        elif mode == 'hungarian':
            variants_of_assignment = [hungarian_choose_to_schedule(probs)]
        elif mode == 'random':
            variants_of_assignment = [random_choose_to_schedule(probs) for _ in range(10)]
        else:
            raise ValueError("mode must be 'greedy' or 'hungarian' or 'random'")

        best_variant, best_obj = [], float('inf')
        for edges_to_schedule in variants_of_assignment:
            # try to schedule and get metrics:
            sch_copy = copy.deepcopy(sch)
            schedule_edges(sch_copy, edges_to_schedule)
            metrics = get_metrics(sch_copy)
            obj_f = metrics['makespan'] + metrics['sum_obj']
            if obj_f < best_obj:
                best_obj = obj_f
                best_variant = edges_to_schedule

        schedule_edges(sch, best_variant)
    return sch


def one_problem_try():
    graph_file = '../data/occidata/graphs/parsedPSPLib/graph_62_100.txt'
    jobs_file = '../data/occidata/tasks/uniform/tasks_uniform_62_0.txt'
    n_workers = 5
    jobs = read_jobs(jobs_file)
    graph = read_graph_from_txt(graph_file)
    problem = Problem(n_workers=n_workers, n_jobs=len(jobs), graph=graph, jobs=jobs)

    checkpoint_path = '../checkpoints/SimpleLinkPredictor_079.pth'
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(map_location)
    best_model = SimpleLinkPredictor(conv_layer=GATv2Conv, in_dim=10, hid_dim=16 * 10, out_dim=8 * 10,
                                     num_layers=6, edge_dim=4).to(device)
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))

    sch = solve_problem_with_nn(best_model, problem, map_location=map_location, mode='greedy')
    print(get_metrics(sch))
    draw_schedule(sch)
    sch = solve_problem_with_nn(best_model, problem, map_location=map_location, mode='hungarian')
    print(get_metrics(sch))
    draw_schedule(sch)
    sch = solve_problem_with_nn(best_model, problem, map_location=map_location, mode='random')
    print(get_metrics(sch))
    draw_schedule(sch)
    sch = rand_sgs(problem)
    print(get_metrics(sch))
    draw_schedule(sch)


def main():
    print("LOAD BEST MODEL:")
    checkpoint_path = '../checkpoints/SimpleLinkPredictor_079.pth'
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(map_location)
    best_model = SimpleLinkPredictor(conv_layer=GATv2Conv, in_dim=10, hid_dim=16*10, out_dim=8*10,
                                     num_layers=6, edge_dim=4).to(device)
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))

    print("COLLECT CP SOLUTIONS")
    # df = make_df_from_all_csv_files('../data/occidata/outputs')
    df = make_df_from_all_csv_files('../data/occidata/outputs_from100')
    include = {'distribution': ['uniform'], 'name': ['DET', 'STs2', 'BBr'], 'jobs_num': [122]}
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
    for jobs_f, problem in tqdm(jobs_f_to_problem.items()):
        sch_greedy = solve_problem_with_nn(best_model, problem, map_location=map_location, mode='greedy')
        jobs_f_name_to_sch[jobs_f]["NN_greedy"] = sch_greedy
        sch_hungarian = solve_problem_with_nn(best_model, problem, map_location=map_location, mode='hungarian')
        jobs_f_name_to_sch[jobs_f]["NN_hungarian"] = sch_hungarian
        sch_nn_random = solve_problem_with_nn(best_model, problem, map_location=map_location, mode='random')
        jobs_f_name_to_sch[jobs_f]["NN_random"] = sch_nn_random

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


if __name__ == '__main__':
    main()
    # one_problem_try()
