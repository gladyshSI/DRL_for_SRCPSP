import copy
import glob
import os
from itertools import product

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
from tqdm import tqdm

from lib.distribution import DiscreteDistribution
from lib.graph import PrecedenceGraph
from lib.job import Job
from lib.problem import get_longest_paths, Problem
from lib.schedule import Schedule, draw_schedule


def solution_to_round_slices(sol: Schedule) -> list[tuple[Schedule, Schedule]]:
    """
    Return list of partial schedules for each  round slice.
    Round slice in this case means partial schedule created simultaneously from left and right sides.
    The first round slice consists of two dummy jobs scheduled at 0 and makespan time points correspondingly.

    Returns list of tuples (left_partial_schedule, right_partial_schedule),
    where right_partial_schedule is a reversed schedule
    """
    makespan = sol.get_makespan()
    problem = sol.get_problem()
    sch = Schedule(problem)
    rev_problem = problem.reverse()
    rev_sch = Schedule(rev_problem)
    ###############################
    # draw_schedule(sol)
    ###############################
    res = []

    first_job_ids = problem.graph.get_start_ids()
    last_job_ids = problem.graph.get_end_ids()

    # print(f'first round job ids: {first_job_ids}; {last_job_ids}')
    for job_id in first_job_ids:
        w_id, st_t = sol.get_scheduled_worker(job_id), sol.get_scheduled_start_time(job_id)
        sch.schedule_job(w_id, job_id, st_t)
    for job_id in last_job_ids:
        w_id, st_t = sol.get_scheduled_worker(job_id), sol.get_scheduled_start_time(job_id)
        # New starting time in the reverse schedule = makespan - end_t:
        st_t = makespan - (st_t + problem.jobs[job_id].get_duration())
        rev_sch.schedule_job(w_id, job_id, st_t)
    first_round_slice = (copy.deepcopy(sch), copy.deepcopy(rev_sch))
    ###############################
    # draw_schedule(first_round_slice[0])
    # draw_schedule(first_round_slice[1])
    ###############################
    res.append(first_round_slice)

    # All candidates of sch are scheduled in the rev_sch <=> all jobs are scheduled either in sch or rev_sch
    while sch.get_candidates() - rev_sch.get_scheduled():
        last_scheduled_jobs = sch.get_last_jobs()
        last_scheduled_jobs_rev = rev_sch.get_last_jobs()
        exec_seq = sol.get_execution_orders()

        # print(f'last_scheduled_jobs: {last_scheduled_jobs}')
        # print(f'last_scheduled_jobs_rev: {last_scheduled_jobs_rev}')

        next_jobs_in_sol = set()
        next_jobs_in_sol_rev = set()
        for job_id in last_scheduled_jobs:
            w_id, st_t = sol.get_scheduled_worker(job_id), sol.get_scheduled_start_time(job_id)
            idx = exec_seq[w_id].index(job_id) + 1
            if idx < len(exec_seq[w_id]) and exec_seq[w_id][idx] not in rev_sch.get_scheduled():
                next_jobs_in_sol.add(exec_seq[w_id][idx])
        for job_id in last_scheduled_jobs_rev:
            w_id, st_t = sol.get_scheduled_worker(job_id), sol.get_scheduled_start_time(job_id)
            idx = exec_seq[w_id].index(job_id) - 1
            if idx >= 0 and exec_seq[w_id][idx] not in sch.get_scheduled():
                next_jobs_in_sol_rev.add(exec_seq[w_id][idx])
        # Put all duplicates only to the left partial schedule and delete from the right partial schedule
        next_jobs_in_sol_rev = next_jobs_in_sol_rev - next_jobs_in_sol

        # print(f'next_jobs_in_sol: {next_jobs_in_sol}')
        # print(f'next_jobs_in_sol_rev: {next_jobs_in_sol_rev}')

        # intersect with candidates:
        first_scheduled_job_ids = sol.get_first_jobs()
        jobs_to_schedule = (next_jobs_in_sol | first_scheduled_job_ids) & sch.get_candidates()
        last_scheduled_jobs_ids = sol.get_last_jobs()
        jobs_to_schedule_rev = (next_jobs_in_sol_rev | last_scheduled_jobs_ids) & rev_sch.get_candidates()

        # print(f'jobs_to_schedule: {jobs_to_schedule}')
        # print(f'jobs_to_schedule_rev: {jobs_to_schedule_rev}')

        # schedule these jobs:
        for job_id in jobs_to_schedule:
            w_id, st_t = sol.get_scheduled_worker(job_id), sol.get_scheduled_start_time(job_id)
            sch.schedule_job(w_id, job_id, st_t)
        for job_id in jobs_to_schedule_rev:
            w_id, st_t = sol.get_scheduled_worker(job_id), sol.get_scheduled_start_time(job_id)
            st_t = makespan - (st_t + problem.jobs[job_id].get_duration())
            rev_sch.schedule_job(w_id, job_id, st_t)
        res.append((copy.deepcopy(sch), copy.deepcopy(rev_sch)))

        #######################
        # draw_schedule(res[-1][0])
        # draw_schedule(res[-1][1])
        #######################
    return res


def round_slice_to_node_ids_mapping(round_slice: tuple[Schedule, Schedule]) -> tuple[dict[int, (str, int)], dict[str, dict[int, int]]]:
    """
    Returns two mappings: problem job id -> (node type; GNN node id) and reverse:
    (node type; GNN node id) -> problem job id:

    node types:
        + "left_dummy"
        + "right_dummy"
        - "not_scheduled"
        - "com"
    """
    pr_id_to_gnn_id = dict()
    gnn_id_to_pr_id = dict()
    types = ['left_dummy', 'right_dummy', 'not_scheduled', 'com']
    for type in types:
        gnn_id_to_pr_id[type] = dict()


    left_sch, right_sch = round_slice
    # Dummy jobs:
    last_left_scheduled = [(w_id, order[-1]) for w_id, order in enumerate(left_sch.get_execution_orders()) if len(order) > 0]
    left_empty_workers = [w_id for w_id, order in enumerate(left_sch.get_execution_orders()) if len(order) == 0]
    last_right_scheduled = [(w_id, order[-1]) for w_id, order in enumerate(right_sch.get_execution_orders()) if len(order) > 0]
    right_empty_workers = [w_id for w_id, order in enumerate(right_sch.get_execution_orders()) if len(order) == 0]
    for w_id, pr_j_id in last_left_scheduled:
        pr_id_to_gnn_id[pr_j_id] = ("left_dummy", w_id)
        gnn_id_to_pr_id["left_dummy"][w_id] = pr_j_id
    for w_id in left_empty_workers:
        gnn_id_to_pr_id["left_dummy"][w_id] = None
    for w_id, pr_j_id in last_right_scheduled:
        pr_id_to_gnn_id[pr_j_id] = ("right_dummy", w_id)
        gnn_id_to_pr_id["right_dummy"][w_id] = pr_j_id
    for w_id in right_empty_workers:
        gnn_id_to_pr_id["right_dummy"][w_id] = None

    # Not scheduled:
    all_jobs_ids = set(range(left_sch.get_problem().n_jobs))
    not_scheduled = all_jobs_ids - left_sch.get_scheduled() - right_sch.get_scheduled()
    for node_id, job_id in enumerate(not_scheduled):
        pr_id_to_gnn_id[job_id] = ("not_scheduled", node_id)
        gnn_id_to_pr_id["not_scheduled"][node_id] = job_id

    # communication nodes:
    gnn_id_to_pr_id["com"][0] = None  # left
    gnn_id_to_pr_id["com"][1] = None  # right
    gnn_id_to_pr_id["com"][2] = None  # common

    return pr_id_to_gnn_id, gnn_id_to_pr_id


def round_partial_sch_to_cropped_data_with_com_nodes(round_slice_before: tuple[Schedule, Schedule],
                                                     round_slice_after: tuple[Schedule, Schedule]):
    """
    ***
    Schedule jobs from slices from left and right sides simultaneously till all the jobs will be scheduled in any part.

    Here we DO NOT keep the information about scheduled jobs (in the left or the right partial schedules)
    (only dummy jobs at the end/start of workers schedule and others)
    Additionally we create one COMMUNICATION NODE that connected to each dummy job and to each candidate on this step
    And one main communication node that connects two communication nodes for the left and right parties.
    Corresponding dummy jobs are also connected

    We hope that this allows to do a better choice: schedule some jobs to the end if they are good candidates,
     and they will not be candidates for the left part on the next steps.
    ***

    Here we will use HeteroData()

    Node features:
    data['left_dummy'].x → [n, F] float
        F = [Completion time distribution approximation: (min, max, avg, Disp)]
    data['right_dummy'].x → [n, F] float
        F = [Completion time distribution approximation: (min, max, avg, Disp)]
    data['left_candidate'].x → [n, F] float
        F = [Duration distribution: (min, max, avg, Disp);
        left longest path (relative): float; right longest path (relative): float]
    data['right_candidate'].x → [n, F] float
        F = [-//-]
    data['left_right_candidate'].x → [n, F] float
        F = [-//-]
    data['not_candidate'].x → [n, F] float
        F = [-//-]
    data['left_com'].x → [n, F] float
        F = [1, 0, 0]
    data['right_com'].x → [n, F] float
        F = [0, 0, 1]
    data['main_com'].x → [n, F] float
        F = [0, 1, 0]

    Graph structure (observed edges used by the encoder):
    data.edge_index → [2, M] long

    Pairs for link classification (positive + negatives):
    data.edge_predict_index → [2, P] long containing both positives and negatives from last scheduled to candidates
    data.edge_predict_label → [P] float (1.0 = positive, 0.0 = negative)
    data.edge_predict_value → [P_pos] float (values for positives only).
    """
    data = HeteroData()

    problem = round_slice_before[0].get_problem()
    n_workers = problem.n_workers
    pr_id_to_gnn_id, gnn_id_to_pr_id = round_slice_to_node_ids_mapping(round_slice_before)
    # get longest paths:
    left_longest_paths = get_longest_paths(problem)  # dict[job_id -> longest path]
    right_longest_paths = get_longest_paths(problem, reverse=True)
    the_longest_path = max(left_longest_paths.values())

    # Dummy jobs:
    for type in ['left_dummy', 'right_dummy']:
        attrs = []
        for node_id in gnn_id_to_pr_id[type].keys():
            j_id = gnn_id_to_pr_id[type].get(node_id)
            min_ct, max_ct, e_ct, d_ct = 0., 0., 0., 0.
            if j_id:
                sch = round_slice_before[0]
                if type == 'right_dummy':
                    sch = round_slice_before[1]
                ct_d = sch.calculate_exact_end_time_distributions()

                min_ct = ct_d[j_id].min_v()
                max_ct = ct_d[j_id].max_v()
                e_ct = ct_d[j_id].e()
                d_ct = ct_d[j_id].d()
            attrs.append([min_ct, max_ct, e_ct, d_ct])
        data[type].x = torch.FloatTensor(attrs)

    # Not scheduled jobs:
    for type in ['not_scheduled']:
        attrs = []
        for node_id in range(len(gnn_id_to_pr_id[type].keys())):
            j_id = gnn_id_to_pr_id[type].get(node_id)
            dur_d = problem.jobs[j_id].get_distribution()
            min_dur, max_dur, e_dur, d_dur = dur_d.min_v(), dur_d.max_v(), dur_d.e(), dur_d.d()
            rel_l_longest_path = left_longest_paths[j_id] / the_longest_path
            rel_r_longest_path = right_longest_paths[j_id] / the_longest_path
            attrs.append([min_dur, max_dur, e_dur, d_dur, rel_l_longest_path, rel_r_longest_path])
        data[type].x = torch.FloatTensor(attrs)

    # Communication nodes
    data['com'].x = torch.FloatTensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    ########################################################

    # Edges:
    # Precedence relations:
    edges_dict = dict()
    for fr_id, to_ids in problem.graph.get_copy_of_all_edges().items():
        if fr_id not in pr_id_to_gnn_id.keys():
            continue
        fr_node_type, fr_node_id = pr_id_to_gnn_id[fr_id]
        for to_id in to_ids:
            if to_id not in pr_id_to_gnn_id.keys():
                continue
            to_node_type, to_node_id = pr_id_to_gnn_id[to_id]
            key = (fr_node_type, 'precedes', to_node_type)
            if key not in edges_dict.keys():
                edges_dict[key] = [[], []]
            edges_dict[key][0].append(fr_node_id)
            edges_dict[key][1].append(to_node_id)

    # Get candidate gnn ids:
    left_candidates_gnn_ids = [pr_id_to_gnn_id[i][1] for i in round_slice_before[0].get_candidates() & pr_id_to_gnn_id.keys()]
    right_candidates_gnn_ids = [pr_id_to_gnn_id[i][1] for i in round_slice_before[1].get_candidates() & pr_id_to_gnn_id.keys()]

    # Communication edges:
    # [(fr_gnn_node, [(to_type, to_gnn_ids),...])]
    com_edges = [(('com', 0), [('left_dummy', gnn_id_to_pr_id['left_dummy'].keys()),
                               ('not_scheduled', left_candidates_gnn_ids),
                               ('com', [2])]),
                 (('com', 1), [('right_dummy', gnn_id_to_pr_id['right_dummy'].keys()),
                               ('not_scheduled', right_candidates_gnn_ids),
                               ('com', [2])])]
    for fr_gnn_node, to_gnn_nodes in com_edges:
        fr_type, fr_node_id = fr_gnn_node
        for to_type, to_gnn_ids in to_gnn_nodes:
            key = (fr_type, 'com', to_type)
            if key not in edges_dict.keys():
                edges_dict[key] = [[], []]
            edges_dict[key][0].extend([fr_node_id]*len(to_gnn_ids))
            edges_dict[key][1].extend(to_gnn_ids)

    # Edge of correspondence between dummy jobs
    key = ('left_dummy', 'correspond', 'right_dummy')
    edges_dict[key] = [list(range(n_workers)), list(range(n_workers))]

    # Initialize:
    for rel, edges in edges_dict.items():
        data[rel].edge_index = torch.LongTensor(edges)  # shape [2, num_edges]

    # Edges to predict
    edges_predict_index = dict()
    edges_predict_values = dict()

    just_scheduled_ids_l = round_slice_after[0].get_scheduled() - round_slice_before[0].get_scheduled()
    just_scheduled_ids_r = round_slice_after[1].get_scheduled() - round_slice_before[1].get_scheduled()
    just_scheduled_l_gnn_dict = {pr_id_to_gnn_id[i][1]: round_slice_after[0].get_scheduled_worker(i) for i in just_scheduled_ids_l}
    just_scheduled_r_gnn_dict = {pr_id_to_gnn_id[i][1]: round_slice_after[1].get_scheduled_worker(i) for i in just_scheduled_ids_r}
    dummy_gnn_ids = list(range(n_workers))

    # [(predict_from_type, p_fr_ids, p_to_t, p_to_ids, true_assignment)]
    q = [('left_dummy', dummy_gnn_ids, 'not_scheduled', left_candidates_gnn_ids, just_scheduled_l_gnn_dict),
         ('right_dummy', dummy_gnn_ids,'not_scheduled', right_candidates_gnn_ids, just_scheduled_r_gnn_dict)]
    for fr_t, fr_ids, to_t, to_ids, true_assignment in q:
        all_possible_edges = product(fr_ids, to_ids)
        key = (fr_t, 'predict', to_t)
        edges_predict_index[key] = [[], []]
        edges_predict_values[key] = []
        for fr_id, to_id in all_possible_edges:
            edges_predict_index[key][0].append(fr_id)
            edges_predict_index[key][1].append(to_id)
            v = 1 if to_id in true_assignment.keys() and fr_id == true_assignment[to_id] else 0
            edges_predict_values[key].append(v)

    # Initialize:
    for rel, edges in edges_predict_index.items():
        data[rel].edge_label_index = torch.LongTensor(edges)  # shape [2, num_edges possible to predict]
    for rel, values in edges_predict_values.items():
        data[rel].edge_label = torch.IntTensor(values)  # shape [num_edges possible to predict]

    # Add reverse edges
    data = T.ToUndirected(merge=False)(data)
    return data


def test_data_to_round_slices(sol_dir: str | None):
    if not sol_dir:
        type = 2
        if type == 1:
            d = DiscreteDistribution(np.array([1]), np.array([1]))
            jobs = [Job(i, 1, d) for i in range(10)]
            edges = [(0, 1), (0, 2), (1, 3), (1, 5), (1, 6), (2, 4), (3, 7), (4, 5), (4, 6), (5, 9), (6, 7), (6, 8), (7, 9), (8, 9)]
            gr = PrecedenceGraph.set_from_edges_list(edges)
            problem = Problem(n_workers=3, n_jobs=len(jobs), graph=gr, jobs=jobs)
            sol = Schedule(problem)
            j_id_w_id_sol = [(0, 0), (1, 0), (2, 1), (3, 0), (4, 2), (5, 1), (6, 2), (7, 0), (8, 2), (9, 1)]
            for j_id, w_id in j_id_w_id_sol:
                sol.schedule_job(worker_id=w_id, job_id=j_id)
            ##################
            # draw_schedule(sol)
            ##################
        elif type == 2:
            d = DiscreteDistribution(np.array([1]), np.array([1]))
            jobs = [Job(i, 1, d) for i in range(12)]
            edges = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (2, 6), (3, 11), (4, 8),
                     (5, 7), (6, 8), (7, 8), (8, 9), (8, 10),(9, 11), (10, 11)]
            gr = PrecedenceGraph.set_from_edges_list(edges)
            problem = Problem(n_workers=2, n_jobs=len(jobs), graph=gr, jobs=jobs)
            sol = Schedule(problem)
            j_id_w_id_sol = [(0, 0), (1, 0), (2, 1), (4, 0), (5, 1), (6, 0),
                             (7, 1), (3, 0), (8, 1), (9, 0), (10, 1), (11, 1)]
            for j_id, w_id in j_id_w_id_sol:
                sol.schedule_job(worker_id=w_id, job_id=j_id)
            ##################
            # draw_schedule(sol)
            ##################

    else:
        all_solution_files = glob.glob(os.path.join(sol_dir, "*.json"))
        sol = Schedule.read_from_file(all_solution_files[0])

    return solution_to_round_slices(sol)


def main_test():
    # round_slices = test_data_to_round_slices('../data/occidata/solutions/')
    round_slices = test_data_to_round_slices(None)
    #################################
    # draw_schedule(round_slices[1][0])
    # draw_schedule(round_slices[1][1])
    # draw_schedule(round_slices[2][0])
    # draw_schedule(round_slices[2][1])
    #################################
    pr_id_to_gnn_id, gnn_id_to_pr_id = round_slice_to_node_ids_mapping(round_slices[1])
    ####################
    print("pr_id_to_gnn_id = ")
    print(pr_id_to_gnn_id)
    print("gnn_id_to_pr_id = ")
    print(gnn_id_to_pr_id)
    ####################
    data = round_partial_sch_to_cropped_data_with_com_nodes(round_slices[1], round_slices[2])
    print(data)
    print("EDGES:")
    for edge_type in data.edge_types:
        print(f"{edge_type} ->")
        if 'edge_index' in data[edge_type]:
            edge_index = data[edge_type].edge_index
            print(f"{edge_index}\n")

    print("EDGES to predict")
    print("left:")
    print(data[('left_dummy', 'predict', 'not_scheduled')].edge_label_index)
    print(data[('left_dummy', 'predict', 'not_scheduled')].edge_label)
    print("right:")
    print(data[('right_dummy', 'predict', 'not_scheduled')].edge_label_index)
    print(data[('right_dummy', 'predict', 'not_scheduled')].edge_label)


def main_f(solutions_dir: str, datasets_dir: str, part_sch_to_data_f):
    all_solution_files = glob.glob(os.path.join(solutions_dir, "*.json"))
    for solution_file in tqdm(all_solution_files):
        solution_name = solution_file.split('\\')[-1].split('.')[0]
        solution = Schedule.read_from_file(solution_file)
        slices = solution_to_round_slices(solution)
        # print(f'solution_name: {solution_name}, slices len: {len(slices)}')
        for i in range(len(slices) - 1):
            data = part_sch_to_data_f(slices[i], slices[i + 1])
            torch.save(data, datasets_dir + solution_name + f'_round_data_slice_{i}.pt')


if __name__ == '__main__':
    # main_test()
    print("CREATING HETEROGENEOUS DATASETS FROM OCCIDATA SOLUTIONS:")
    main_f(solutions_dir='../data/occidata/solutions/',
           datasets_dir='../data/occidata/datasets_hetero_new/',
           part_sch_to_data_f=round_partial_sch_to_cropped_data_with_com_nodes)
