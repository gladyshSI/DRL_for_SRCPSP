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
from lib.schedule import Schedule, draw_schedule, sensitivity_test, get_exact_left_shift_overlap_distributions


def solution_to_round_slices(sol: Schedule) -> list[tuple[Schedule, Schedule]]:
    """
    Return list of partial schedules for each round slice.
    Round slice in this case means partial schedule created simultaneously from left and right sides.
    The first round slice consists of two dummy jobs scheduled at 0 and makespan time points correspondingly.

    Returns list of tuples (left_partial_schedule, right_partial_schedule),
    where right_partial_schedule is a reversed schedule
    """
    makespan = sol.get_makespan()
    problem = sol.get_problem()
    sch = Schedule(problem)
    rev_sch = Schedule(problem.reverse())

    def get_rev_st(job_id: int, st_t: int) -> int:
        return makespan - (st_t + problem.jobs[job_id].get_duration())

    def schedule_job(schedule: Schedule, job_id: int, reverse: bool = False):
        w_id = sol.get_scheduled_worker(job_id)
        st_t = sol.get_scheduled_start_time(job_id)
        schedule.schedule_job(w_id, job_id, get_rev_st(job_id, st_t) if reverse else st_t)

    first_job_ids = problem.graph.get_start_ids()
    last_job_ids = problem.graph.get_end_ids() - first_job_ids

    for job_id in first_job_ids:
        schedule_job(sch, job_id)
    for job_id in last_job_ids:
        schedule_job(rev_sch, job_id, reverse=True)

    res = [(copy.deepcopy(sch), copy.deepcopy(rev_sch))]

    while sch.get_candidates() - rev_sch.get_scheduled():
        exec_seq = sol.get_execution_orders()

        next_jobs_in_sol = set()
        for job_id in sch.get_last_jobs():
            w_id = sol.get_scheduled_worker(job_id)
            idx = exec_seq[w_id].index(job_id) + 1
            if idx < len(exec_seq[w_id]) and exec_seq[w_id][idx] not in rev_sch.get_scheduled():
                next_jobs_in_sol.add(exec_seq[w_id][idx])

        next_jobs_in_sol_rev = set()
        for job_id in rev_sch.get_last_jobs():
            w_id = sol.get_scheduled_worker(job_id)
            idx = exec_seq[w_id].index(job_id) - 1
            if idx >= 0 and exec_seq[w_id][idx] not in sch.get_scheduled():
                next_jobs_in_sol_rev.add(exec_seq[w_id][idx])

        jobs_to_schedule = (next_jobs_in_sol | sol.get_first_jobs()) & sch.get_candidates()
        jobs_to_schedule_rev = ((next_jobs_in_sol_rev | sol.get_last_jobs()) & rev_sch.get_candidates()) - jobs_to_schedule

        for job_id in jobs_to_schedule:
            schedule_job(sch, job_id)
        for job_id in jobs_to_schedule_rev:
            schedule_job(rev_sch, job_id, reverse=True)

        res.append((copy.deepcopy(sch), copy.deepcopy(rev_sch)))

    return res


def round_slice_to_node_ids_mapping(round_slice: tuple[Schedule, Schedule]) -> tuple[dict[int, (str, int)], dict[str, dict[int, int]]]:
    """
    Returns two mappings:
    pr_id_to_gnn_id: [problem job id] -> (node type; GNN node id)
    gnn_id_to_pr_id: [node type][GNN node id] -> problem job id:

    node types:
        - "left_scheduled" (without left dummy)
        - "left_dummy"
        - "not_scheduled"
        - "right_dummy"
        - "right_scheduled" (without right dummy)
        - "com"
    """
    pr_id_to_gnn_id = dict()
    gnn_id_to_pr_id = dict()
    types = ['left_scheduled', 'left_dummy', 'right_dummy', 'right_scheduled', 'not_scheduled', 'com']
    for type in types:
        gnn_id_to_pr_id[type] = dict()


    left_sch, right_sch = round_slice
    # scheduled + dummy jobs
    def scheduled_and_dummy(sch: Schedule, scheduled_t: str, dummy_t: str):
        scheduled_job_counter = 0
        for w_id, order in enumerate(sch.get_execution_orders()):
            if len(order) == 0:
                gnn_id_to_pr_id[dummy_t][w_id] = None
            else:
                pr_id_to_gnn_id[order[-1]] = (dummy_t, w_id)
                gnn_id_to_pr_id[dummy_t][w_id] = order[-1]
                for job_id in order[:-1]:
                    pr_id_to_gnn_id[job_id] = (scheduled_t, scheduled_job_counter)
                    gnn_id_to_pr_id[scheduled_t][scheduled_job_counter] = job_id
                    scheduled_job_counter += 1
    scheduled_and_dummy(left_sch, "left_scheduled", "left_dummy")
    scheduled_and_dummy(right_sch, "right_scheduled", "right_dummy")

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


def round_partial_sch_to_data(round_slice_before: tuple[Schedule, Schedule],
                              round_slice_after: tuple[Schedule, Schedule] | None = None):
    include_left_scheduled: bool = False
    include_right_scheduled: bool = True
    include_exec_seq_edges: bool = True

    # Feature dimensions for each node type
    NODE_FEAT_DIMS = {
        'left_scheduled':  4,
        'left_dummy':      4,
        'not_scheduled':   6,
        'right_dummy':     4,
        'right_scheduled': 4,
        'com':             3,
    }

    data = HeteroData()

    problem = round_slice_before[0].get_problem()
    n_workers = problem.n_workers
    pr_id_to_gnn_id, gnn_id_to_pr_id = round_slice_to_node_ids_mapping(round_slice_before)

    left_longest_paths = get_longest_paths(problem)
    right_longest_paths = get_longest_paths(problem, reverse=True)
    the_longest_path = max(left_longest_paths.values())

    def empty_x(node_type: str) -> torch.Tensor:
        """Return a (0, F) float tensor for a node type with no instances."""
        return torch.empty((0, NODE_FEAT_DIMS[node_type]), dtype=torch.float)

    #################### ATTRIBUTES ########################

    # left_scheduled
    if include_left_scheduled:
        ntype = 'left_scheduled'
        sch = round_slice_before[0]
        ct_d = sch.calculate_exact_end_time_distributions()
        attrs = []
        for node_id in range(len(gnn_id_to_pr_id[ntype])):
            job_id = gnn_id_to_pr_id[ntype].get(node_id)
            attrs.append([ct_d[job_id].min_v(), ct_d[job_id].max_v(),
                          ct_d[job_id].e(),     ct_d[job_id].d()])
        x = torch.FloatTensor(attrs) if attrs else empty_x(ntype)
        data[ntype].x = x
        data[ntype].num_nodes = x.size(0)
    else:
        ntype = 'left_scheduled'
        data[ntype].x = empty_x(ntype)
        data[ntype].num_nodes = 0

    # left_dummy
    ntype = 'left_dummy'
    sch = round_slice_before[0]
    ct_d = sch.calculate_exact_end_time_distributions()
    attrs = []
    for node_id in range(len(gnn_id_to_pr_id[ntype])):
        job_id = gnn_id_to_pr_id[ntype].get(node_id)
        if job_id:
            attrs.append([ct_d[job_id].min_v(), ct_d[job_id].max_v(),
                          ct_d[job_id].e(),     ct_d[job_id].d()])
        else:
            attrs.append([0., 0., 0., 0.])
    x = torch.FloatTensor(attrs) if attrs else empty_x(ntype)
    data[ntype].x = x
    data[ntype].num_nodes = x.size(0)

    # not_scheduled
    ntype = 'not_scheduled'
    attrs = []
    for node_id in range(len(gnn_id_to_pr_id[ntype])):
        job_id = gnn_id_to_pr_id[ntype].get(node_id)
        dur_d = problem.jobs[job_id].get_distribution()
        attrs.append([
            dur_d.min_v(), dur_d.max_v(), dur_d.e(), dur_d.d(),
            left_longest_paths[job_id]  / the_longest_path,
            right_longest_paths[job_id] / the_longest_path,
        ])
    x = torch.FloatTensor(attrs) if attrs else empty_x(ntype)
    data[ntype].x = x
    data[ntype].num_nodes = x.size(0)

    # right_dummy
    ntype = 'right_dummy'
    sch = round_slice_before[1]
    scheduled_num = len(sch.get_scheduled())
    attrs = []
    for node_id in range(len(gnn_id_to_pr_id[ntype])):
        job_id = gnn_id_to_pr_id[ntype].get(node_id)
        if job_id is not None:
            expected_st_t = -sch.get_scheduled_end_time(job_id)
            s1 = sum(map(abs, sensitivity_test(sch=sch, job_id=job_id, shift=-1.))) / scheduled_num
            s2 = sum(map(abs, sensitivity_test(sch=sch, job_id=job_id, shift=-2.))) / scheduled_num
            s4 = sum(map(abs, sensitivity_test(sch=sch, job_id=job_id, shift=-4.))) / scheduled_num
            attrs.append([expected_st_t, s1, s2, s4])
        else:
            attrs.append([0., 0., 0., 0.])
    x = torch.FloatTensor(attrs) if attrs else empty_x(ntype)
    data[ntype].x = x
    data[ntype].num_nodes = x.size(0)

    # right_scheduled
    if include_right_scheduled:
        ntype = 'right_scheduled'
        sch = round_slice_before[1]
        ct_d = get_exact_left_shift_overlap_distributions(sch=sch)
        attrs = []
        for node_id in range(len(gnn_id_to_pr_id[ntype])):
            job_id = gnn_id_to_pr_id[ntype].get(node_id)
            attrs.append([ct_d[job_id].min_v(), ct_d[job_id].max_v(),
                          ct_d[job_id].e(),     ct_d[job_id].d()])
        x = torch.FloatTensor(attrs) if attrs else empty_x(ntype)
        data[ntype].x = x
        data[ntype].num_nodes = x.size(0)
    else:
        ntype = 'right_scheduled'
        data[ntype].x = empty_x(ntype)
        data[ntype].num_nodes = 0

    # com — always exactly 3 nodes
    ntype = 'com'
    data[ntype].x = torch.FloatTensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    data[ntype].num_nodes = 3

    ########################################################

    ####################### EDGES ##########################

    def safe_add_edge(edges_dict, key, fr_id, to_id):
        if key not in edges_dict:
            edges_dict[key] = [[], []]
        edges_dict[key][0].append(fr_id)
        edges_dict[key][1].append(to_id)

    edges_dict = {}

    # Precedence relations
    for fr_id, to_ids in problem.graph.get_copy_of_all_edges().items():
        if fr_id not in pr_id_to_gnn_id:
            continue
        fr_node_type, fr_node_id = pr_id_to_gnn_id[fr_id]
        if (fr_node_type == 'left_scheduled'  and not include_left_scheduled or
            fr_node_type == 'right_scheduled' and not include_right_scheduled):
            continue
        for to_id in to_ids:
            if to_id not in pr_id_to_gnn_id:
                continue
            to_node_type, to_node_id = pr_id_to_gnn_id[to_id]
            if (to_node_type == 'left_scheduled'  and not include_left_scheduled or
                to_node_type == 'right_scheduled' and not include_right_scheduled):
                continue
            safe_add_edge(edges_dict, (fr_node_type, 'precedes', to_node_type), fr_node_id, to_node_id)

    # Execution sequence relations
    def add_exec_edges(orders, pairs_fn):
        for order in orders:
            for fr_id, to_id in pairs_fn(order):
                if fr_id not in pr_id_to_gnn_id or to_id not in pr_id_to_gnn_id:
                    continue
                fr_node_type, fr_node_id = pr_id_to_gnn_id[fr_id]
                to_node_type, to_node_id = pr_id_to_gnn_id[to_id]
                safe_add_edge(edges_dict, (fr_node_type, 'exec_before', to_node_type), fr_node_id, to_node_id)

    if include_exec_seq_edges:
        if include_left_scheduled:
            add_exec_edges(
                round_slice_before[0].get_execution_orders(),
                lambda order: zip(order, order[1:])
            )
        if include_right_scheduled:
            add_exec_edges(
                round_slice_before[1].get_execution_orders(),
                lambda order: zip(reversed(order), order[-2::-1])
            )

    # Candidate gnn ids
    left_candidates_gnn_ids = [
        pr_id_to_gnn_id[i][1]
        for i in round_slice_before[0].get_candidates() & pr_id_to_gnn_id.keys()
        if pr_id_to_gnn_id[i][0] == "not_scheduled"
    ]
    right_candidates_gnn_ids = [
        pr_id_to_gnn_id[i][1]
        for i in round_slice_before[1].get_candidates() & pr_id_to_gnn_id.keys()
        if pr_id_to_gnn_id[i][0] == "not_scheduled"
    ]

    # Communication edges
    com_edges = [
        (('com', 0), [('left_dummy',     list(gnn_id_to_pr_id['left_dummy'].keys())),
                      ('not_scheduled',  left_candidates_gnn_ids),
                      ('com',            [2])]),
        (('com', 1), [('right_dummy',    list(gnn_id_to_pr_id['right_dummy'].keys())),
                      ('not_scheduled',  right_candidates_gnn_ids),
                      ('com',            [2])]),
    ]
    for (fr_type, fr_node_id), to_gnn_nodes in com_edges:
        for to_type, to_gnn_ids in to_gnn_nodes:
            key = (fr_type, 'com', to_type)
            if key not in edges_dict:
                edges_dict[key] = [[], []]
            to_gnn_ids = list(to_gnn_ids)
            edges_dict[key][0].extend([fr_node_id] * len(to_gnn_ids))
            edges_dict[key][1].extend(to_gnn_ids)

    # Correspondence edges between dummy jobs
    edges_dict[('left_dummy', 'correspond', 'right_dummy')] = [
        list(range(n_workers)), list(range(n_workers))
    ]

    # Assign all edge_index tensors; ensure every edge type is a valid LongTensor
    for rel, edges in edges_dict.items():
        src, dst = edges
        if len(src) == 0:
            data[rel].edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            data[rel].edge_index = torch.LongTensor([src, dst])

    #################### PREDICTION EDGES ##################

    edges_predict_index = {}
    edges_predict_mask  = {}
    edges_predict_buf   = {}

    dummy_gnn_ids = list(range(n_workers))

    if round_slice_after is not None:
        just_scheduled_ids_l = round_slice_after[0].get_scheduled() - round_slice_before[0].get_scheduled()
        just_scheduled_ids_r = round_slice_after[1].get_scheduled() - round_slice_before[1].get_scheduled()
        just_scheduled_l_gnn_dict = {
            pr_id_to_gnn_id[i][1]: round_slice_after[0].get_scheduled_worker(i)
            for i in just_scheduled_ids_l
        }
        just_scheduled_r_gnn_dict = {
            pr_id_to_gnn_id[i][1]: round_slice_after[1].get_scheduled_worker(i)
            for i in just_scheduled_ids_r
        }
    else:
        just_scheduled_l_gnn_dict = {}
        just_scheduled_r_gnn_dict = {}

    q = [
        ('left_dummy',  dummy_gnn_ids, 'not_scheduled', left_candidates_gnn_ids,  just_scheduled_l_gnn_dict),
        ('right_dummy', dummy_gnn_ids, 'not_scheduled', right_candidates_gnn_ids, just_scheduled_r_gnn_dict),
    ]
    for fr_t, fr_ids, to_t, to_ids, true_assignment in q:
        key = (fr_t, 'predict', to_t)
        src_list, dst_list, mask_list, buf_list = [], [], [], []

        for fr_id, to_id in product(fr_ids, to_ids):
            src_list.append(fr_id)
            dst_list.append(to_id)

            v = 1 if (to_id in true_assignment and fr_id == true_assignment[to_id]) else 0
            mask_list.append(v)

            buf = 0.
            if v == 1 and round_slice_after is not None:
                sch_before = round_slice_before[0] if fr_t == 'left_dummy' else round_slice_before[1]
                sch_after  = round_slice_after[0]  if fr_t == 'left_dummy' else round_slice_after[1]
                first_possible_st_t = sch_before.get_first_possible_st_t(fr_id, to_id)
                pr_to_id = gnn_id_to_pr_id['not_scheduled'][to_id]
                buf = sch_after.get_scheduled_start_time(pr_to_id) - first_possible_st_t
            buf_list.append(buf)

        if src_list:
            data[key].edge_label_index = torch.LongTensor([src_list, dst_list])
        else:
            data[key].edge_label_index = torch.empty((2, 0), dtype=torch.long)

        if round_slice_after is not None:
            data[key].edge_value  = torch.LongTensor(mask_list)
            data[key].edge_length = torch.LongTensor(buf_list)

    # Add reverse edges
    data = T.ToUndirected(merge=False)(data)
    return data


def gen_test_small_sch():
    g = PrecedenceGraph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(1, 4)
    g.add_edge(2, 4)
    g.add_edge(3, 5)
    g.add_edge(4, 5)

    dist0 = DiscreteDistribution(values=np.array([1]), probs=np.array([1.]))
    dist3 = DiscreteDistribution(values=np.array([2, 3, 4]), probs=np.array([1. / 3] * 3))
    jobs = ([Job(0, 1, dist0)] +
            [Job(i, 3, dist3) for i in range(1, 5)] +
            [Job(5, 1, dist0)])

    problem = Problem(n_workers=2, n_jobs=len(jobs), graph=g, jobs=jobs)
    sch = Schedule(problem=problem)
    to_schedule = [(0, 0, 0), (0, 2, 1), (1, 3, 2), (0, 1, 5), (1, 4, 8), (1, 5, 11)]
    for w_id, j_id, st_t in to_schedule:
        sch.schedule_job(worker_id=w_id, job_id=j_id, start_time=st_t)
    ##################
    # draw_schedule(sch)
    ##################
    return sch

def test_data_to_round_slices(sol_dir: str | None):
    if not sol_dir:
        sol = gen_test_small_sch()
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
    print("SCHEDULED LEFT:", round_slices[1][0].get_scheduled(), round_slices[2][0].get_scheduled())
    print("SCHEDULED RIGHT:", round_slices[1][1].get_scheduled(), round_slices[2][1].get_scheduled())
    data = round_partial_sch_to_data(round_slices[1], round_slices[2])
    print(data)
    print("FEATURES:")
    for node_type in data.node_types:
        print(f"{node_type}: {data[node_type].x}")
    print("EDGES:")
    for edge_type in data.edge_types:
        print(f"{edge_type} ->")
        if 'edge_index' in data[edge_type]:
            edge_index = data[edge_type].edge_index
            print(f"{edge_index}\n")

    print("EDGES to predict")
    print("left:")
    print(data[('left_dummy', 'predict', 'not_scheduled')].edge_label_index)
    print(data[('left_dummy', 'predict', 'not_scheduled')].edge_value)
    print(data[('left_dummy', 'predict', 'not_scheduled')].edge_length)
    print("right:")
    print(data[('right_dummy', 'predict', 'not_scheduled')].edge_label_index)
    print(data[('right_dummy', 'predict', 'not_scheduled')].edge_value)
    print(data[('right_dummy', 'predict', 'not_scheduled')].edge_length)

    print("DATA FOR PROBLEM SOLVING")
    data = round_partial_sch_to_data(round_slices[1])
    print(data)



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
           part_sch_to_data_f=round_partial_sch_to_data)
