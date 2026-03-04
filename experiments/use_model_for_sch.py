import copy
import random

import numpy as np
import torch
from torch_geometric.config_store import Model
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
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


def get_j_id_node_id_relationship(part_sch: Schedule):
    problem = part_sch.get_problem()

    # Candidates on the previous step:
    scheduled = part_sch.get_scheduled()
    last_executed = set()
    for seq in part_sch.get_execution_orders():
        if len(seq) > 0:
            last_executed.add(seq[-1])


    j_id_to_gnn_node_id = dict()
    gnn_node_id_to_j_id = dict()
    n_not_sch_jobs = problem.n_jobs - len(scheduled)
    free_id = 0
    for j_id in range(problem.n_jobs):
        if j_id in last_executed:
            # assign to the last executed jobs max_id + w_id
            # for workers that doesn't perform any jobs we will create dummy jobs
            w_id = part_sch.get_scheduled_worker(j_id)
            node_id = n_not_sch_jobs + w_id
            j_id_to_gnn_node_id[j_id] = node_id
            gnn_node_id_to_j_id[node_id] = j_id
        elif j_id in scheduled:
            continue
        else:
            j_id_to_gnn_node_id[j_id] = free_id
            gnn_node_id_to_j_id[free_id] = j_id
            free_id += 1

    return j_id_to_gnn_node_id, gnn_node_id_to_j_id


def partial_sch_to_cropped_data(part_sch: Schedule) -> Data:
    """
    ***
    Here we DO NOT keep the information about scheduled jobs (only dummy jobs at the end of workers schedule and others
    ***

    Node features:
    data.x → [N, F] float
        F = [Last performed: bool; Candidate: bool;
        Duration distribution: (min, max, avg); Completion time distribution approximation: (min, max, avg),
        (NEW) left longest path (relative): float, (NEW) right longest path (relative): float]

    Graph structure (observed edges used by the encoder):
    data.edge_index → [2, M] long
    data.edge_attr → [M, Fe] float (optional)
        Fe = [precedence: bool, reverse precedence: bool]

    Pairs for link classification (positive + negatives):
    data.edge_predict_index → [2, P] long containing both positives and negatives from last scheduled to candidates
    data.edge_predict_label → [P] float (1.0 = positive, 0.0 = negative)
    data.edge_predict_value → [P_pos] float (values for positives only).
    """
    problem = part_sch.get_problem()

    # get longest paths:
    left_longest_paths = get_longest_paths(problem)  # dict[job_id -> longest path]
    right_longest_paths = get_longest_paths(problem, reverse=True)
    the_longest_path = max(left_longest_paths.values())

    # Candidates on the previous step:
    scheduled = part_sch.get_scheduled()
    candidates = part_sch.get_candidates()

    # Calculate exact end time distributions
    end_t_distributions = part_sch.calculate_exact_end_time_distributions()

    n_not_sch_jobs = problem.n_jobs - len(scheduled)
    j_id_to_gnn_node_id, gnn_node_id_to_j_id = get_j_id_node_id_relationship(part_sch)

    # Precedence relation edges
    precedence_fr_ids = []
    precedence_to_ids = []
    for fr_id, to_ids in problem.graph.get_copy_of_all_edges().items():
        for to_id in to_ids:
            if fr_id in j_id_to_gnn_node_id.keys() and to_id in j_id_to_gnn_node_id.keys():
                precedence_fr_ids.append(j_id_to_gnn_node_id[fr_id])
                precedence_to_ids.append(j_id_to_gnn_node_id[to_id])

    # Predict edges from dummy job to the last scheduled jobs
    prev_candidates_gnn_nodes = [j_id_to_gnn_node_id[j_id] for j_id in candidates]
    possible_edges_from_ids = []
    possible_edges_to_ids = []
    for fr_id in [n_not_sch_jobs + w_id for w_id in range(problem.n_workers)]:
        for to_id in prev_candidates_gnn_nodes:
            possible_edges_from_ids.append(fr_id)
            possible_edges_to_ids.append(to_id)

    # Vertex attributes:
    v_attributes = []
    for node_id in range(n_not_sch_jobs + problem.n_workers):
        j_id = None
        if node_id in gnn_node_id_to_j_id.keys():
            j_id = gnn_node_id_to_j_id[node_id]
        last_performed = 1 if node_id >= n_not_sch_jobs else 0
        candidate = 1 if node_id in prev_candidates_gnn_nodes else 0

        min_dur, max_dur, e_dur = 0, 0, 0
        if j_id:
            min_dur = problem.jobs[j_id].get_distribution().min_v()
            max_dur = problem.jobs[j_id].get_distribution().max_v()
            e_dur = problem.jobs[j_id].get_distribution().e()
        min_ct, max_ct, e_ct = 0, 0, 0
        if j_id and last_performed == 1:
            min_ct = end_t_distributions[j_id].min_v()
            max_ct = end_t_distributions[j_id].max_v()
            e_ct = end_t_distributions[j_id].e()

        rel_l_longest_path, rel_r_longest_path = 0., 1.
        if j_id:
            rel_l_longest_path = left_longest_paths[j_id] / the_longest_path
            rel_r_longest_path = right_longest_paths[j_id] / the_longest_path

        v_i_attr = [last_performed, candidate, min_dur, max_dur, e_dur, min_ct, max_ct, e_ct, rel_l_longest_path, rel_r_longest_path]
        v_attributes.append(v_i_attr)

    total_fr_ids = precedence_fr_ids + precedence_to_ids
    total_to_ids = precedence_to_ids + precedence_fr_ids
    prec_edge_atr = [[1, 0] for _ in range(len(precedence_fr_ids))]
    rev_prec_edge_atr = [[0, 1] for _ in range(len(precedence_to_ids))]
    total_edge_atr = prec_edge_atr + rev_prec_edge_atr
    # Combining into data object:
    data = Data(
        x=torch.FloatTensor(v_attributes),  # [N, F]
        edge_index=torch.LongTensor([total_fr_ids, total_to_ids]),  # [2, M_train]
        edge_attr=torch.FloatTensor(total_edge_atr),  # [M_train, Fe] optional
        edges_eval_index=torch.LongTensor([possible_edges_from_ids, possible_edges_to_ids])
    )
    return data


def partial_sch_to_cropped_data_with_com_node(part_sch: Schedule) -> Data:
    """
    ***
    Here we DO NOT keep the information about scheduled jobs (only dummy jobs at the end of workers schedule and others
    Additionally we create one COMMUNICATION NODE that connected to each dummy job and to each candidate on this step
    We hope that this allows candidates to communicate with each other and allows to do a better choice.
    ***

    Node features:
    data.x → [N, F] float
        F = [Last performed: bool; Candidate: bool; (NEW) Communication node: bool
        Duration distribution: (min, max, avg); Completion time distribution approximation: (min, max, avg, (NEW) Disp),
        left longest path (relative): float, right longest path (relative): float]

    Graph structure (observed edges used by the encoder):
    data.edge_index → [2, M] long
    data.edge_attr → [M, Fe] float
        Fe = [precedence: bool, reverse precedence: bool,
              (NEW) dummy-to-com: bool, (NEW) com-to-dummy: bool,
              (NEW) candidate-to-com: bool, (NEW) com-to-candidate: bool]

    Pairs for link classification (positive + negatives):
    data.edge_predict_index → [2, P] long containing both positives and negatives from last scheduled to candidates
    data.edge_predict_label → [P] float (1.0 = positive, 0.0 = negative)
    data.edge_predict_value → [P_pos] float (values for positives only).
    """
    problem = part_sch.get_problem()

    # get longest paths:
    left_longest_paths = get_longest_paths(problem)  # dict[job_id -> longest path]
    right_longest_paths = get_longest_paths(problem, reverse=True)
    the_longest_path = max(left_longest_paths.values())

    # Candidates on the previous step:
    scheduled = part_sch.get_scheduled()
    candidates = part_sch.get_candidates()

    # Calculate exact end time distributions
    end_t_distributions = part_sch.calculate_exact_end_time_distributions()

    n_not_sch_jobs = problem.n_jobs - len(scheduled)
    j_id_to_gnn_node_id, gnn_node_id_to_j_id = get_j_id_node_id_relationship(part_sch)

    # COMMUNICATION NODE ID = jobs max_id + workers num
    communication_node_id = n_not_sch_jobs + problem.n_workers

    # Precedence relation edges
    precedence_fr_ids = []
    precedence_to_ids = []
    for fr_id, to_ids in problem.graph.get_copy_of_all_edges().items():
        for to_id in to_ids:
            if fr_id in j_id_to_gnn_node_id.keys() and to_id in j_id_to_gnn_node_id.keys():
                precedence_fr_ids.append(j_id_to_gnn_node_id[fr_id])
                precedence_to_ids.append(j_id_to_gnn_node_id[to_id])

    # Communication edges
    com_workers_fr_ids = [n_not_sch_jobs + w_id for w_id in range(problem.n_workers)]
    com_workers_to_ids = [communication_node_id] * len(com_workers_fr_ids)
    com_candidates_fr_ids = [j_id_to_gnn_node_id[j_id] for j_id in candidates]
    com_candidates_to_ids = [communication_node_id] * len(com_candidates_fr_ids)

    # Predict edges from dummy job to the last scheduled jobs
    prev_candidates_gnn_nodes = [j_id_to_gnn_node_id[j_id] for j_id in candidates]
    possible_edges_from_ids = []
    possible_edges_to_ids = []
    for fr_id in [n_not_sch_jobs + w_id for w_id in range(problem.n_workers)]:
        for to_id in prev_candidates_gnn_nodes:
            possible_edges_from_ids.append(fr_id)
            possible_edges_to_ids.append(to_id)

    # Vertex attributes:
    v_attributes = []
    for node_id in range(n_not_sch_jobs + problem.n_workers):
        j_id = None
        if node_id in gnn_node_id_to_j_id.keys():
            j_id = gnn_node_id_to_j_id[node_id]
        last_performed = 1 if node_id >= n_not_sch_jobs else 0
        candidate = 1 if node_id in prev_candidates_gnn_nodes else 0

        min_dur, max_dur, e_dur = 0, 0, 0
        if j_id:
            min_dur = problem.jobs[j_id].get_distribution().min_v()
            max_dur = problem.jobs[j_id].get_distribution().max_v()
            e_dur = problem.jobs[j_id].get_distribution().e()
        min_ct, max_ct, e_ct, d_ct = 0, 0, 0, 0
        if j_id and last_performed == 1:
            min_ct = end_t_distributions[j_id].min_v()
            max_ct = end_t_distributions[j_id].max_v()
            e_ct = end_t_distributions[j_id].e()
            d_ct = end_t_distributions[j_id].d()

        rel_l_longest_path, rel_r_longest_path = 0., 1.
        if j_id:
            rel_l_longest_path = left_longest_paths[j_id] / the_longest_path
            rel_r_longest_path = right_longest_paths[j_id] / the_longest_path

        v_i_attr = [last_performed, candidate, 0, min_dur, max_dur, e_dur, min_ct, max_ct, e_ct, d_ct,
                    rel_l_longest_path, rel_r_longest_path]
        v_attributes.append(v_i_attr)
    # COMMUNICATION NODE ATTRIBUTES:
    com_v_attr = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    v_attributes.append(com_v_attr)

    total_fr_ids = precedence_fr_ids + precedence_to_ids + com_workers_fr_ids + com_workers_to_ids + com_candidates_fr_ids + com_candidates_to_ids
    total_to_ids = precedence_to_ids + precedence_fr_ids + com_workers_to_ids + com_workers_fr_ids + com_candidates_to_ids + com_candidates_fr_ids
    prec_edge_atr = [[1, 0, 0, 0, 0, 0] for _ in range(len(precedence_fr_ids))]
    rev_prec_edge_atr = [[0, 1, 0, 0, 0, 0] for _ in range(len(precedence_to_ids))]
    com_w_edge_atr = [[0, 0, 1, 0, 0, 0] for _ in range(len(com_workers_fr_ids))]
    rev_com_w_edge_atr = [[0, 0, 0, 1, 0, 0] for _ in range(len(com_workers_to_ids))]
    com_c_edge_atr = [[0, 0, 0, 0, 1, 0] for _ in range(len(com_candidates_fr_ids))]
    rev_com_c_edge_atr = [[0, 0, 0, 0, 0, 1] for _ in range(len(com_candidates_to_ids))]
    total_edge_atr = prec_edge_atr + rev_prec_edge_atr + com_w_edge_atr + rev_com_w_edge_atr + com_c_edge_atr + rev_com_c_edge_atr
    # Combining into data object:
    data = Data(
        x=torch.FloatTensor(v_attributes),  # [N, F]
        edge_index=torch.LongTensor([total_fr_ids, total_to_ids]),  # [2, M_train]
        edge_attr=torch.FloatTensor(total_edge_atr),  # [M_train, Fe] optional
        edges_eval_index=torch.LongTensor([possible_edges_from_ids, possible_edges_to_ids])
    )
    return data


def get_probs_values(model, data, map_location='cpu') -> list[tuple[int, int, float, int]]:
    if isinstance(model, SimpleLinkPredictor):
        inf = "links_only"
    elif isinstance(model, SimpleLinkValuePredictor):
        inf = "links_values"
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    model.eval()
    with torch.no_grad():
        data = data.to(map_location)
        z = model(data.x, data.edge_index, edge_attr=getattr(data, 'edge_attr', None))

        src_idx = data.edges_eval_index[0]  # shape [E]
        dst_idx = data.edges_eval_index[1]  # shape [E]

        src_emb = z[src_idx]  # shape [E, embedding_dim]
        dst_emb = z[dst_idx]  # shape [E, embedding_dim]

        if inf == "links_only":
            logits = model.score_edges(src_emb, dst_emb)  # shape [E]
            values = torch.zeros(logits.size(), dtype=torch.int)
        elif inf == "links_values":
            logits, values = model.score_and_value(src_emb, dst_emb)
            values = (values > 0.5).int()
        probs = torch.sigmoid(logits)  # if you want probabilities in [0,1]

    probs = probs.cpu().numpy()
    values = values.cpu().numpy()
    src_cpu = src_idx.cpu().numpy()
    dst_cpu = dst_idx.cpu().numpy()
    results = [(int(i), int(j), float(p), v) for i, j, p, v in zip(src_cpu, dst_cpu, probs, values)]
    return results


def greedy_choose_to_schedule(edges_probs: list[tuple[int, int, float, int]]) -> list[tuple[int, int, int]]:
    edges_to_schedule = []
    while len(edges_probs) > 0:
        max_t = max(edges_probs, key=lambda x: x[2])
        fr_id, to_id, value = max_t[0], max_t[1], max_t[3]
        edges_to_schedule.append((fr_id, to_id, value))
        edges_probs = [t for t in edges_probs if t[0] != fr_id and t[1] != to_id]
    return edges_to_schedule


def hungarian_choose_to_schedule(edges_probs: list[tuple[int, int, float, int]]) -> list[tuple[int, int, int]]:
    rows = list(dict.fromkeys(fr for fr, _, _, _ in edges_probs))
    cols = list(dict.fromkeys(to for _, to, _, _ in edges_probs))
    r_idx = {v: i for i, v in enumerate(rows)}
    c_idx = {v: i for i, v in enumerate(cols)}

    probs = np.zeros((len(rows), len(cols)))
    values = np.zeros((len(rows), len(cols)))
    for fr, to, p, v in edges_probs:
        i, j = r_idx[fr], c_idx[to]
        probs[i, j] = p
        values[i, j] = v

    # use maximize=True (SciPy >= 1.4.0). If older SciPy, use linear_sum_assignment(-probs)
    row_inds, col_inds = linear_sum_assignment(-probs)
    return [(int(rows[i]), int(cols[j]), int(values[i, j])) for i, j in zip(row_inds, col_inds)]


def random_choose_to_schedule(edges_probs: list[tuple[int, int, float, int]]) -> list[tuple[int, int, int]]:
    edges_to_schedule = []
    while len(edges_probs) > 0:
        items, weights = zip(*[((fr_id, to_id, value), p) for fr_id, to_id, p, value in edges_probs])
        choice = random.choices(items, weights=weights, k=1)[0]
        edges_to_schedule.append(choice)
        edges_probs = [t for t in edges_probs if t[0] != choice[0] and t[1] != choice[1]]
    return edges_to_schedule


def schedule_edges(sch: Schedule, edges_to_sch: list[tuple[int, int, int]], shift: int):
    for fr_id, to_id, value in edges_to_sch:
        w_id = fr_id - shift
        w_exec_order = sch.get_execution_orders()[w_id]
        prev_end_t = 0 if len(w_exec_order) == 0 else sch.get_scheduled_end_time(w_exec_order[-1])
        st_time = max(sch.get_first_possible_st_t(worker_id=w_id, job_id=to_id), prev_end_t + value)
        sch.schedule_job(worker_id=w_id, job_id=to_id, start_time=st_time)


def calculate_agg_f_ovp_dist_e(agg_f, ovp_dist: dict[int, Distribution]):
    return agg_f([x.e() for _, x in ovp_dist.items()])


def get_metrics(sch: Schedule) -> dict[str, float]:
    dist = sch.calculate_exact_overlap_distributions()
    return {'makespan': sch.get_makespan(),
            'sum_obj': calculate_agg_f_ovp_dist_e(sum, dist),
            'max_obj': calculate_agg_f_ovp_dist_e(max, dist)}


def solve_problem_with_nn(model, problem: Problem, map_location: str = 'cpu', mode: str = 'greedy') -> Schedule:
    sch = Schedule(problem=problem)
    n_jobs = problem.n_jobs
    sch.schedule_job(worker_id=0, job_id=0)
    while len(sch.get_candidates()) > 0:
        data = part_sch_to_data(sch)
        probs_values = get_probs_values(model, data, map_location=map_location)

        if mode == 'greedy':
            variants_of_assignment = [greedy_choose_to_schedule(probs_values)]
        elif mode == 'hungarian':
            variants_of_assignment = [hungarian_choose_to_schedule(probs_values)]
        elif mode == 'random':
            variants_of_assignment = [random_choose_to_schedule(probs_values) for _ in range(10)]
        else:
            raise ValueError("mode must be 'greedy' or 'hungarian' or 'random'")

        best_variant, best_obj = [], float('inf')
        for edges_to_schedule in variants_of_assignment:
            # try to schedule and get metrics:
            sch_copy = copy.deepcopy(sch)
            schedule_edges(sch_copy, edges_to_schedule, shift=n_jobs)
            metrics = get_metrics(sch_copy)
            obj_f = metrics['makespan'] + metrics['sum_obj']
            if obj_f < best_obj:
                best_obj = obj_f
                best_variant = edges_to_schedule

        schedule_edges(sch, best_variant, shift=n_jobs)
    return sch


def solve_problem_with_cropped_nn(model, problem: Problem, part_sch_to_data_f, map_location: str = 'cpu', mode: str = 'greedy') -> Schedule:
    sch = Schedule(problem=problem)
    sch.schedule_job(worker_id=0, job_id=0)
    while len(sch.get_candidates()) > 0:
        j_id_to_gnn_node_id, gnn_node_id_to_j_id = get_j_id_node_id_relationship(sch)
        n_not_sch_jobs = problem.n_jobs - len(sch.get_scheduled())
        # data = partial_sch_to_cropped_data(sch)
        data = part_sch_to_data_f(sch)
        probs_values = get_probs_values(model, data, map_location=map_location)

        if mode == 'greedy':
            variants_of_assignment = [greedy_choose_to_schedule(probs_values)]
        elif mode == 'hungarian':
            variants_of_assignment = [hungarian_choose_to_schedule(probs_values)]
        elif mode == 'random':
            variants_of_assignment = [random_choose_to_schedule(probs_values) for _ in range(10)]
        else:
            raise ValueError("mode must be 'greedy' or 'hungarian' or 'random'")

        best_variant, best_obj = [], float('inf')
        for edges_to_schedule in variants_of_assignment:
            # transfer to our ids:
            edges_to_schedule_our_ids = [(x[0], gnn_node_id_to_j_id[x[1]], x[2]) for x in edges_to_schedule]
            # try to schedule and get metrics:
            sch_copy = copy.deepcopy(sch)
            schedule_edges(sch_copy, edges_to_schedule_our_ids, shift=n_not_sch_jobs)
            metrics = get_metrics(sch_copy)
            obj_f = metrics['makespan'] + metrics['sum_obj']
            if obj_f < best_obj:
                best_obj = obj_f
                best_variant = edges_to_schedule_our_ids

        schedule_edges(sch, best_variant, shift=n_not_sch_jobs)
    return sch


def load_model_from_checkpoint(checkpoint_path: str, map_location: str = 'cpu'):
    print("LOAD BEST MODEL:")
    device = torch.device(map_location)
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    config = checkpoint['config']
    print(config)
    if config['conv_class'] == 'GATv2Conv':
        conv_layer = GATv2Conv
    else:
        raise ValueError("Only GATv2Conv supported now")
    if checkpoint['model_name'] == 'SimpleLinkPredictor':
        model = SimpleLinkValuePredictor(conv_layer=conv_layer, in_dim=config['in_dim'],
                                         hid_dim=config['hid_dim'], out_dim=config['out_dim'],
                                         num_layers=config['num_layers'], edge_dim=config['edge_dim']).to(device)
    elif checkpoint['model_name'] == 'SimpleLinkValuePredictor':
        model = SimpleLinkValuePredictor(conv_layer=conv_layer, in_dim=config['in_dim'],
                                         hid_dim=config['hid_dim'], out_dim=config['out_dim'],
                                         num_layers=config['num_layers'], edge_dim=config['edge_dim'],
                                         heads=config['heads']).to(device)
    else:
        raise ValueError("Only SimpleLinkPredictor and SimpleLinkValuePredictor supported now")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def one_problem_try():
    # df = make_df_from_all_csv_files('../data/occidata/outputs')
    df = make_df_from_all_csv_files('../data/occidata/outputs_from400')
    # df = make_df_from_all_csv_files('../data/occidata/outputs_200_400')
    include = {'distribution': ['uniform'], 'name': ['BBr'], 'jobs_num': [122]}
    df_filtered = filter_dataframe(df, include=include, exclude={}).drop_duplicates(subset=['name', 'jobs_f'])
    graph_f, jobs_f, n_workers, sch_str = df_filtered.iloc[1][["graph_f", "jobs_f", "workers_num", "schedule"]]

    jobs = read_jobs(occidata_jobs_file_path_to_ours(jobs_f))
    graph = read_graph_from_txt(occidata_graph_file_path_to_ours(graph_f))
    problem = Problem(n_workers=n_workers, n_jobs=len(jobs), graph=graph, jobs=jobs)


    sch = Schedule(problem=problem)
    make_schedule_from_str(sch, sch_str)
    print("BBr", get_metrics(sch))
    draw_schedule(sch)

    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = '../checkpoints/CroppedComNodeLinkValuePredictor.pth'
    best_model = load_model_from_checkpoint(checkpoint)

    # part_sch_to_data_f = partial_sch_to_cropped_data
    part_sch_to_data_f = partial_sch_to_cropped_data_with_com_node
    sch = solve_problem_with_cropped_nn(best_model, problem, part_sch_to_data_f, map_location=map_location, mode='greedy')
    print("greedy", get_metrics(sch))
    draw_schedule(sch)
    sch = solve_problem_with_cropped_nn(best_model, problem, part_sch_to_data_f, map_location=map_location, mode='hungarian')
    print("hungarian", get_metrics(sch))
    draw_schedule(sch)
    sch = solve_problem_with_cropped_nn(best_model, problem, part_sch_to_data_f, map_location=map_location, mode='random')
    print("random", get_metrics(sch))
    draw_schedule(sch)
    sch = rand_sgs_best_of_k(problem, k=20)
    print("rand sgs best of 20", get_metrics(sch))
    draw_schedule(sch)


def main():
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = '../checkpoints/CroppedComNodeLinkValuePredictor.pth'
    best_model = load_model_from_checkpoint(checkpoint)

    print("COLLECT CP SOLUTIONS")
    # df = make_df_from_all_csv_files('../data/occidata/outputs')
    # df = make_df_from_all_csv_files('../data/occidata/outputs_from400')
    df = make_df_from_all_csv_files('../data/occidata/outputs_200_400')
    include = {'distribution': ['uniform'], 'name': ['DET', 'STs2', 'BBr'], 'jobs_num': [402]}
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
    part_sch_to_data_f = partial_sch_to_cropped_data_with_com_node
    for jobs_f, problem in tqdm(jobs_f_to_problem.items()):
        sch_greedy = solve_problem_with_cropped_nn(best_model, problem, part_sch_to_data_f, map_location=map_location, mode='greedy')
        jobs_f_name_to_sch[jobs_f]["NN_greedy"] = sch_greedy
        sch_hungarian = solve_problem_with_cropped_nn(best_model, problem, part_sch_to_data_f, map_location=map_location, mode='hungarian')
        jobs_f_name_to_sch[jobs_f]["NN_hungarian"] = sch_hungarian
        sch_nn_random = solve_problem_with_cropped_nn(best_model, problem, part_sch_to_data_f, map_location=map_location, mode='random')
        jobs_f_name_to_sch[jobs_f]["NN_random"] = sch_nn_random
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


if __name__ == '__main__':
    main()
    # one_problem_try()
    # one_problem_try(mode="links_values")
