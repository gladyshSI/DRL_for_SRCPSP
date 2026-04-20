import copy
import glob
import os

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from lib.distribution import DiscreteDistribution
from lib.graph import PrecedenceGraph
from lib.job import Job
from lib.problem import get_longest_paths, Problem
from lib.schedule import Schedule, draw_schedule


def solution_to_slices(sol: Schedule) -> list[Schedule]:
    """Return list of partial schedules for each slice"""
    problem = sol.get_problem()
    sch = Schedule(problem)
    res = []

    first_job_ids = problem.graph.get_start_ids()
    first_scheduled_job_ids = sol.get_first_jobs()
    for job_id in first_job_ids:
        w_id, st_t = sol.get_scheduled_worker(job_id), sol.get_scheduled_start_time(job_id)
        sch.schedule_job(w_id, job_id, st_t)
    first_slice = copy.deepcopy(sch)
    res.append(first_slice)

    while sch.get_candidates():
        last_scheduled_jobs = sch.get_last_jobs()
        exec_seq = sol.get_execution_orders()

        next_jobs_in_sol = set()
        # fill next_jobs in solution
        for job_id in last_scheduled_jobs:
            w_id, st_t = sol.get_scheduled_worker(job_id), sol.get_scheduled_start_time(job_id)
            idx = exec_seq[w_id].index(job_id) + 1
            if idx < len(exec_seq[w_id]):
                next_jobs_in_sol.add(exec_seq[w_id][idx])
        # intersect with candidates:
        jobs_to_schedule = (next_jobs_in_sol | first_scheduled_job_ids) & sch.get_candidates()
        # schedule these jobs:
        for job_id in jobs_to_schedule:
            w_id, st_t = sol.get_scheduled_worker(job_id), sol.get_scheduled_start_time(job_id)
            sch.schedule_job(w_id, job_id, st_t)
        res.append(copy.deepcopy(sch))

    return res


def partial_sch_to_data(part_sch_before: Schedule, part_sch_after: Schedule) -> Data:
    """
    Node features:
    data.x → [N, F] float
        F = [Scheduled: bool; Last performed: bool; Candidate: bool; Worker ID + 1: int;
        Duration distribution: (min, max, avg); Competition time distribution approximation: (min, max, avg),
        (NEW) left longest path %: float, (NEW) right longest path %: float]

    Graph structure (observed edges used by the encoder):
    data.edge_index → [2, M] long
    data.edge_attr → [M, Fe] float (optional)
        Fe = [precedence: bool, exec_order: bool, reverse precedence: bool, reverse_exec_order: bool]

    Pairs for link classification (positive + negatives):
    data.edge_predict_index → [2, P] long containing both positives and negatives from last scheduled to candidates
    data.edge_predict_label → [P] float (1.0 = positive, 0.0 = negative)
    data.edge_predict_value → [P_pos] float (values for positives only).
    """
    problem = part_sch_after.get_problem()

    # get longest paths:
    left_longest_paths = get_longest_paths(problem)  # dict[job_id -> longest path]
    right_longest_paths = get_longest_paths(problem, reverse=True)
    the_longest_path = max(left_longest_paths.values())

    # Candidates on the previous step:
    prev_scheduled = part_sch_before.get_scheduled()
    prev_candidates = part_sch_before.get_candidates()

    # Calculate exact end time distributions
    end_t_distributions = part_sch_before.calculate_exact_end_time_distributions()

    # Precedence relation edges
    precedence_fr_ids = []
    precedence_to_ids = []
    for fr_id, to_ids in problem.graph.get_copy_of_all_edges().items():
        precedence_fr_ids.extend(fr_id for _ in to_ids)
        precedence_to_ids.extend(to_ids)

    # Execution order edges
    execution_order_fr_ids = []
    execution_order_to_ids = []
    for w_exec_order in part_sch_before.get_execution_orders():
        for i in range(len(w_exec_order) - 1):
            execution_order_fr_ids.append(w_exec_order[i])
            execution_order_to_ids.append(w_exec_order[i + 1])

    # Make dummy jobs at the end of each workers' schedule.
    # dummy job for worker w_i has id = max_job_id + 1 + w_i
    for w_id, w_exec_order in enumerate(part_sch_before.get_execution_orders()):
        if len(w_exec_order) >= 1:
            execution_order_fr_ids.append(w_exec_order[-1])
            execution_order_to_ids.append(problem.n_jobs + w_id)

    # Predict edges from dummy job to the last scheduled jobs
    set_of_all_possible_edges_to_predict = {(fr_id, to_id) for fr_id in [problem.n_jobs + w_id for w_id in range(problem.n_workers)]
                                            for to_id in prev_candidates}
    edges_to_predict_fr_ids = []
    edges_to_predict_to_ids = []
    lengths_to_predict = []
    edges_not_to_predict_fr_ids = []
    edges_not_to_predict_to_ids = []
    for w_id, w_exec_order in enumerate(part_sch_after.get_execution_orders()):
        if len(w_exec_order) >= 1 and w_exec_order[-1] in prev_candidates:
            fr_id = problem.n_jobs + w_id
            to_id = w_exec_order[-1]
            edges_to_predict_fr_ids.append(fr_id)
            edges_to_predict_to_ids.append(to_id)
            set_of_all_possible_edges_to_predict.remove((fr_id, to_id))
            # length to predict = buffer time after previous scheduled job on this machine or 0 time point
            prev_end_t = 0 if len(w_exec_order) == 1 else part_sch_after.get_scheduled_end_time(w_exec_order[-2])
            lengths_to_predict.append(part_sch_after.get_scheduled_start_time(to_id) - prev_end_t)

    # the rest of edges that are not in to_predict add to not_to_predict
    for fr_id, to_id in set_of_all_possible_edges_to_predict:
        edges_not_to_predict_fr_ids.append(fr_id)
        edges_not_to_predict_to_ids.append(to_id)

    w_id_to_last_ct_distribution = dict()
    w_id_to_relative_longest_paths = dict()
    # Vertex attributes:
    v_attributes = []
    for j_id in range(problem.n_jobs):
        scheduled = 1 if j_id in prev_scheduled else 0
        last_performed = 0
        candidate = 1 if j_id in prev_candidates else 0
        w_id = 0 if j_id not in prev_scheduled else part_sch_after.get_scheduled_worker(j_id) + 1  # 0 is for None

        min_dur = problem.jobs[j_id].get_distribution().min_v()
        max_dur = problem.jobs[j_id].get_distribution().max_v()
        e_dur = problem.jobs[j_id].get_distribution().e()
        min_ct, max_ct, e_ct = 0, 0, 0
        if scheduled == 1:
            min_ct = end_t_distributions[j_id].min_v()
            max_ct = end_t_distributions[j_id].max_v()
            e_ct = end_t_distributions[j_id].e()

        relative_left_longest_path = left_longest_paths[j_id] / the_longest_path
        relative_right_longest_path = right_longest_paths[j_id] / the_longest_path

        # save info if it is prev last performed
        if w_id > 0:
            w_exec_order = part_sch_before.get_execution_orders()[w_id - 1]
            if len(w_exec_order) >= 1 and w_exec_order[-1] == j_id:
                w_id_to_last_ct_distribution[w_id - 1] = end_t_distributions[j_id]
                w_id_to_relative_longest_paths[w_id - 1] = (relative_left_longest_path, relative_right_longest_path)

        v_i_attr = [scheduled, last_performed, candidate, w_id, min_dur, max_dur, e_dur, min_ct, max_ct, e_ct, relative_left_longest_path, relative_right_longest_path]
        v_attributes.append(v_i_attr)
    # additional dummy jobs at the end of each workers' schedule
    for w_id in range(problem.n_workers):
        last_ct_distribution = w_id_to_last_ct_distribution[w_id] if w_id in w_id_to_last_ct_distribution.keys() \
            else DiscreteDistribution(np.array([0]), np.array([1.]))
        relative_longest_paths = w_id_to_relative_longest_paths[w_id] if w_id in w_id_to_last_ct_distribution.keys() \
            else (0., 1.)
        v_i_attr = [0, 1, 0, w_id + 1, 0, 0, 0, last_ct_distribution.min_v(), last_ct_distribution.max_v(), last_ct_distribution.e(), relative_longest_paths[0], relative_longest_paths[1]]
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

        edge_predict_index=torch.LongTensor([edges_to_predict_fr_ids + edges_not_to_predict_fr_ids,
                                             edges_to_predict_to_ids + edges_not_to_predict_to_ids]),  # [2, P]
        edge_predict_label=torch.FloatTensor([1.]*len(edges_to_predict_fr_ids) + [0.]*len(edges_not_to_predict_fr_ids)),  # [P] float(1.0 = positive, 0.0 = negative)
        edge_predict_value=torch.FloatTensor(lengths_to_predict + [0.]*len(edges_not_to_predict_fr_ids))  # [P]
    )
    return data


def partial_sch_to_cropped_data(part_sch_before: Schedule, part_sch_after: Schedule) -> Data:
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
    problem = part_sch_after.get_problem()

    # get longest paths:
    left_longest_paths = get_longest_paths(problem)  # dict[job_id -> longest path]
    right_longest_paths = get_longest_paths(problem, reverse=True)
    the_longest_path = max(left_longest_paths.values())

    # Candidates on the previous step:
    prev_scheduled = part_sch_before.get_scheduled()
    prev_candidates = part_sch_before.get_candidates()
    last_executed = set()
    for seq in part_sch_before.get_execution_orders():
        if len(seq) > 0:
            last_executed.add(seq[-1])

    # Calculate exact end time distributions
    end_t_distributions = part_sch_before.calculate_exact_end_time_distributions()

    j_id_to_gnn_node_id = dict()
    gnn_node_id_to_j_id = dict()
    n_not_sch_jobs = problem.n_jobs - len(prev_scheduled)
    free_id = 0
    for j_id in range(problem.n_jobs):
        if j_id in last_executed:
            # assign to the last executed jobs max_id + w_id
            # for workers that doesn't perform any jobs we will create dummy jobs
            w_id = part_sch_before.get_scheduled_worker(j_id)
            node_id = n_not_sch_jobs + w_id
            j_id_to_gnn_node_id[j_id] = node_id
            gnn_node_id_to_j_id[node_id] = j_id
        elif j_id in prev_scheduled:
            continue
        else:
            j_id_to_gnn_node_id[j_id] = free_id
            gnn_node_id_to_j_id[free_id] = j_id
            free_id += 1

    # Precedence relation edges
    precedence_fr_ids = []
    precedence_to_ids = []
    for fr_id, to_ids in problem.graph.get_copy_of_all_edges().items():
        for to_id in to_ids:
            if fr_id in j_id_to_gnn_node_id.keys() and to_id in j_id_to_gnn_node_id.keys():
                precedence_fr_ids.append(j_id_to_gnn_node_id[fr_id])
                precedence_to_ids.append(j_id_to_gnn_node_id[to_id])

    # Predict edges from dummy job to the last scheduled jobs
    prev_candidates_gnn_nodes = [j_id_to_gnn_node_id[j_id] for j_id in prev_candidates]
    set_of_all_possible_edges_to_predict = {(fr_id, to_id) for fr_id in [n_not_sch_jobs + w_id for w_id in range(problem.n_workers)]
                                            for to_id in prev_candidates_gnn_nodes}
    edges_to_predict_fr_ids = []
    edges_to_predict_to_ids = []
    lengths_to_predict = []
    edges_not_to_predict_fr_ids = []
    edges_not_to_predict_to_ids = []
    for w_id, w_exec_order in enumerate(part_sch_after.get_execution_orders()):
        if len(w_exec_order) >= 1 and w_exec_order[-1] in prev_candidates:
            fr_id = n_not_sch_jobs + w_id
            to_id = j_id_to_gnn_node_id[w_exec_order[-1]]
            edges_to_predict_fr_ids.append(fr_id)
            edges_to_predict_to_ids.append(to_id)
            set_of_all_possible_edges_to_predict.remove((fr_id, to_id))
            # length to predict = buffer time after previous scheduled job on this machine or 0 time point
            # prev_end_t = 0 if len(w_exec_order) == 1 else part_sch_after.get_scheduled_end_time(w_exec_order[-2])
            # stert_time - first possible start time
            first_possible_st_t = part_sch_before.get_first_possible_st_t(w_id, w_exec_order[-1])
            lengths_to_predict.append(part_sch_after.get_scheduled_start_time(w_exec_order[-1]) - first_possible_st_t)

    # the rest of edges that are not in to_predict add to not_to_predict
    for fr_id, to_id in set_of_all_possible_edges_to_predict:
        edges_not_to_predict_fr_ids.append(fr_id)
        edges_not_to_predict_to_ids.append(to_id)

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

        edge_predict_index=torch.LongTensor([edges_to_predict_fr_ids + edges_not_to_predict_fr_ids,
                                             edges_to_predict_to_ids + edges_not_to_predict_to_ids]),  # [2, P]
        edge_predict_label=torch.FloatTensor([1.]*len(edges_to_predict_fr_ids) + [0.]*len(edges_not_to_predict_fr_ids)),  # [P] float(1.0 = positive, 0.0 = negative)
        edge_predict_value=torch.FloatTensor(lengths_to_predict + [0.]*len(edges_not_to_predict_fr_ids))  # [P]
    )
    return data


def partial_sch_to_cropped_data_with_com_node(part_sch_before: Schedule, part_sch_after: Schedule) -> Data:
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
    problem = part_sch_after.get_problem()

    # get longest paths:
    left_longest_paths = get_longest_paths(problem)  # dict[job_id -> longest path]
    right_longest_paths = get_longest_paths(problem, reverse=True)
    the_longest_path = max(left_longest_paths.values())

    # Candidates on the previous step:
    prev_scheduled = part_sch_before.get_scheduled()
    prev_candidates = part_sch_before.get_candidates()
    last_executed = set()
    for seq in part_sch_before.get_execution_orders():
        if len(seq) > 0:
            last_executed.add(seq[-1])

    # Calculate exact end time distributions
    end_t_distributions = part_sch_before.calculate_exact_end_time_distributions()

    j_id_to_gnn_node_id = dict()
    gnn_node_id_to_j_id = dict()
    n_not_sch_jobs = problem.n_jobs - len(prev_scheduled)
    free_id = 0
    for j_id in range(problem.n_jobs):
        if j_id in last_executed:
            # assign to the last executed jobs max_id + w_id
            # for workers that doesn't perform any jobs we will create dummy jobs
            w_id = part_sch_before.get_scheduled_worker(j_id)
            node_id = n_not_sch_jobs + w_id
            j_id_to_gnn_node_id[j_id] = node_id
            gnn_node_id_to_j_id[node_id] = j_id
        elif j_id in prev_scheduled:
            continue
        else:
            j_id_to_gnn_node_id[j_id] = free_id
            gnn_node_id_to_j_id[free_id] = j_id
            free_id += 1
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
    com_candidates_fr_ids = [j_id_to_gnn_node_id[j_id] for j_id in prev_candidates]
    com_candidates_to_ids = [communication_node_id] * len(com_candidates_fr_ids)

    # Predict edges from dummy job to the last scheduled jobs
    prev_candidates_gnn_nodes = [j_id_to_gnn_node_id[j_id] for j_id in prev_candidates]
    set_of_all_possible_edges_to_predict = {(fr_id, to_id) for fr_id in [n_not_sch_jobs + w_id for w_id in range(problem.n_workers)]
                                            for to_id in prev_candidates_gnn_nodes}
    edges_to_predict_fr_ids = []
    edges_to_predict_to_ids = []
    lengths_to_predict = []
    edges_not_to_predict_fr_ids = []
    edges_not_to_predict_to_ids = []
    for w_id, w_exec_order in enumerate(part_sch_after.get_execution_orders()):
        if len(w_exec_order) >= 1 and w_exec_order[-1] in prev_candidates:
            fr_id = n_not_sch_jobs + w_id
            to_id = j_id_to_gnn_node_id[w_exec_order[-1]]
            edges_to_predict_fr_ids.append(fr_id)
            edges_to_predict_to_ids.append(to_id)
            set_of_all_possible_edges_to_predict.remove((fr_id, to_id))
            first_possible_st_t = part_sch_before.get_first_possible_st_t(w_id, w_exec_order[-1])
            lengths_to_predict.append(part_sch_after.get_scheduled_start_time(w_exec_order[-1]) - first_possible_st_t)

    # the rest of edges that are not in to_predict add to not_to_predict
    for fr_id, to_id in set_of_all_possible_edges_to_predict:
        edges_not_to_predict_fr_ids.append(fr_id)
        edges_not_to_predict_to_ids.append(to_id)

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

        v_i_attr = [last_performed, candidate, 0, min_dur, max_dur, e_dur, min_ct, max_ct, e_ct, d_ct, rel_l_longest_path, rel_r_longest_path]
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

        edge_predict_index=torch.LongTensor([edges_to_predict_fr_ids + edges_not_to_predict_fr_ids,
                                             edges_to_predict_to_ids + edges_not_to_predict_to_ids]),  # [2, P]
        edge_predict_label=torch.FloatTensor([1.]*len(edges_to_predict_fr_ids) + [0.]*len(edges_not_to_predict_fr_ids)),  # [P] float(1.0 = positive, 0.0 = negative)
        edge_predict_value=torch.FloatTensor(lengths_to_predict + [0.]*len(edges_not_to_predict_fr_ids))  # [P]
    )
    return data


def main():
    solutions_dir = '../data/solutions/'
    datasets_dir = '../data/datasets_with_length/'
    problems_num = 30
    init_dur_range = (5, 10)
    delta_dur_range = (1, 4)

    jobs_nums = [30, 60, 90]
    workers_nums = [4, 6, 7]

    for jobs_num, workers_num in zip(jobs_nums, workers_nums):
        for problem_id in range(problems_num):
            solution_name = (f'BBr_rand_p_{jobs_num}_j_{workers_num}_w_discrD_'
                            f'{init_dur_range[0]}_{init_dur_range[1]}_'
                            f'{delta_dur_range[0]}_{delta_dur_range[1]}-{problem_id}.json')
            solution_path = os.path.join(solutions_dir, solution_name)

            solution = Schedule.read_from_file(solution_path)
            slices = solution_to_slices(solution)
            print(f'jobs_num: {jobs_num}, problem_id: {problem_id}, slices len: {len(slices)}')
            for i in range(len(slices) - 1):
                data = partial_sch_to_data(slices[i], slices[i + 1])
                torch.save(data, datasets_dir + solution_name[:-5] + f'data_slice_{i}.pt')


def main_occidata():
    solutions_dir = '../data/occidata/solutions/'
    datasets_dir = '../data/occidata/datasets/'
    all_solution_files = glob.glob(os.path.join(solutions_dir, "*.json"))
    for solution_file in all_solution_files:
        solution_name = solution_file.split('\\')[-1].split('.')[0]
        solution = Schedule.read_from_file(solution_file)
        slices = solution_to_slices(solution)
        print(f'solution_name: {solution_name}, slices len: {len(slices)}')
        for i in range(len(slices) - 1):
            data = partial_sch_to_data(slices[i], slices[i + 1])
            torch.save(data, datasets_dir + solution_name + f'_data_slice_{i}.pt')


def main_occidata_cropped():
    solutions_dir = '../data/occidata/solutions/'
    datasets_dir = '../data/occidata/datasets_cropped/'
    all_solution_files = glob.glob(os.path.join(solutions_dir, "*.json"))
    for solution_file in all_solution_files:
        solution_name = solution_file.split('\\')[-1].split('.')[0]
        solution = Schedule.read_from_file(solution_file)
        slices = solution_to_slices(solution)
        print(f'solution_name: {solution_name}, slices len: {len(slices)}')
        for i in range(len(slices) - 1):
            data = partial_sch_to_cropped_data(slices[i], slices[i + 1])
            torch.save(data, datasets_dir + solution_name + f'_data_slice_{i}.pt')


def main_f(solutions_dir: str, datasets_dir: str, part_sch_to_data_f):
    all_solution_files = glob.glob(os.path.join(solutions_dir, "*.json"))
    for solution_file in all_solution_files:
        solution_name = solution_file.split('\\')[-1].split('.')[0]
        solution = Schedule.read_from_file(solution_file)
        slices = solution_to_slices(solution)
        print(f'solution_name: {solution_name}, slices len: {len(slices)}')
        for i in range(len(slices) - 1):
            data = part_sch_to_data_f(slices[i], slices[i + 1])
            torch.save(data, datasets_dir + solution_name + f'_data_slice_{i}.pt')


if __name__ == '__main__':
    # main_occidata()
    # main_occidata_cropped()
    main_f(solutions_dir='../data/occidata/solutions/',
           datasets_dir='../data/occidata/datasets_cropped_com_node/',
           part_sch_to_data_f=partial_sch_to_cropped_data_with_com_node)
    # schedule = Schedule.read_from_file('../data/toy_solutions/sol_toy1.json')
    # folder_to_save = '../data/toy_datasets/'
    # # draw_schedule(schedule)
    # slices = solution_to_slices(schedule)
    # print(f'slices len: {len(slices)}')
    # for i in range(len(slices) - 1):
    #     print(f'######################## {i} ########################')
    #     # draw_schedule(slices[i+1])
    #     data = partial_sch_to_data(slices[i], slices[i+1])
    #     torch.save(data, folder_to_save+f'toy_data_slice{i}.pt')
    #     data = torch.load(folder_to_save+f'toy_data_slice{i}.pt', weights_only=False)
    #     print(data)

