import copy
import os

import numpy as np
import torch
from torch_geometric.data import Data

from lib.distribution import DiscreteDistribution
from lib.schedule import Schedule, draw_schedule


def solution_to_slices(sol: Schedule) -> list[Schedule]:
    """Return list of partial schedules for each slice"""
    problem = sol.get_problem()
    sch = Schedule(problem)
    res = []

    first_job_ids = problem.graph.get_start_ids()
    first_scheduled_job_ids = sol.get_first_jobs()
    # print(f'first_job_ids: {first_job_ids}')
    for job_id in first_job_ids:
        w_id, st_t = sol.get_scheduled_worker(job_id), sol.get_scheduled_start_time(job_id)
        sch.schedule_job(w_id, job_id, st_t)
    first_slice = copy.deepcopy(sch)
    res.append(first_slice)

    # i = 0
    while sch.get_candidates():
        # print(f'i: {i}; candidates: {sch._candidates}')
        # i += 1

        last_scheduled_jobs = sch.get_last_jobs()
        exec_seq = sol.get_execution_order()

        # print(f'last_scheduled_jobs: {last_scheduled_jobs}')
        next_jobs_in_sol = set()
        # fill next_jobs in solution
        for job_id in last_scheduled_jobs:
            w_id, st_t = sol.get_scheduled_worker(job_id), sol.get_scheduled_start_time(job_id)
            idx = exec_seq[w_id].index(job_id) + 1
            if idx < len(exec_seq[w_id]):
                next_jobs_in_sol.add(exec_seq[w_id][idx])
        # print(f'next_jobs_in_sol: {next_jobs_in_sol}')
        # intersect with candidates:
        jobs_to_schedule = (next_jobs_in_sol | first_scheduled_job_ids) & sch.get_candidates()
        # print(f'jobs_to_schedule: {jobs_to_schedule}')
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
        Duration distribution: (min, max, avg); Competition time distribution approximation: (min, max, avg)]

    Graph structure (observed edges used by the encoder):
    data.edge_index → [2, M] long
    data.edge_attr → [M, Fe] float (optional)
        Fe = [precedence: bool, exec_order: bool, reverse: precedence: bool, reverse_exec_order: bool]

    Pairs for link classification (positive + negatives):
    data.edge_predict_index → [2, P] long
    data.edge_predict_label → [P] float (1.0 = positive, 0.0 = negative)
    data.edge_predict_value → [P_pos] float (values for positives only).
    """
    problem = part_sch_after.get_problem()
    # Candidates on the previous step:
    prev_scheduled = part_sch_before.get_scheduled()
    prev_candidates = part_sch_before.get_candidates()

    # Calculate exact end time distributions
    overlaps = part_sch_before.calculate_exact_overlap_distributions()
    end_t_distributions = dict()
    for i, overlap_distribution in overlaps.items():
        scheduled_st_t = part_sch_before.get_scheduled_start_time(i)
        dur_distribution = problem.jobs[i].get_distribution()
        end_t_distributions[i] = scheduled_st_t + overlap_distribution + dur_distribution

    # Precedence relation edges
    precedence_fr_ids = []
    precedence_to_ids = []
    for fr_id, to_ids in problem.graph.get_copy_of_all_edges().items():
        for to_id in to_ids:
            precedence_fr_ids.append(fr_id)
            precedence_to_ids.append(to_id)

    # Execution order edges
    execution_order_fr_ids = []
    execution_order_to_ids = []
    for w_exec_order in part_sch_before.get_execution_order():
        for i in range(len(w_exec_order) - 1):
            execution_order_fr_ids.append(w_exec_order[i])
            execution_order_to_ids.append(w_exec_order[i + 1])

    # Make dummy jobs at the end of each workers' schedule.
    # dummy job for worker w_i has id = max_job_id + 1 + w_i
    for w_id, w_exec_order in enumerate(part_sch_before.get_execution_order()):
        if len(w_exec_order) >= 1:
            execution_order_fr_ids.append(w_exec_order[-1])
            execution_order_to_ids.append(problem.n_jobs + w_id)

    # Predict edges from dummy job to the last scheduled jobs
    set_of_all_possible_edges_to_predict = {(fr_id, to_ids) for fr_id in [problem.n_jobs + w_id for w_id in range(problem.n_workers)]
                                            for to_ids in prev_candidates}
    edges_to_predict_fr_ids = []
    edges_to_predict_to_ids = []
    lengths_to_predict = []
    edges_not_to_predict_fr_ids = []
    edges_not_to_predict_to_ids = []
    for w_id, w_exec_order in enumerate(part_sch_after.get_execution_order()):
        if len(w_exec_order) >= 1 and w_exec_order[-1] in prev_candidates:
            fr_id = problem.n_jobs + w_id
            to_id = w_exec_order[-1]
            edges_to_predict_fr_ids.append(fr_id)
            edges_to_predict_to_ids.append(to_id)
            set_of_all_possible_edges_to_predict.remove((fr_id, to_id))

            sch_st_t_to_id = part_sch_after.get_scheduled_start_time(to_id)
            pred_end_ts = [part_sch_after.get_scheduled_end_time(pred_id) for pred_id in problem.graph.get_predecessors(to_id)]
            if len(w_exec_order) >= 2:
                pred_end_ts.append(part_sch_after.get_scheduled_end_time(w_exec_order[-2]))
            max_pred_end_t = 0 if len(pred_end_ts) == 0 else max(pred_end_ts)
            lengths_to_predict.append(sch_st_t_to_id - max_pred_end_t)
    for fr_id, to_id in set_of_all_possible_edges_to_predict:
        edges_not_to_predict_fr_ids.append(fr_id)
        edges_not_to_predict_to_ids.append(to_id)


    w_id_to_last_ct_distribution = dict()
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

        # save ct distribution if it is prev last performed
        if w_id > 0:
            w_exec_order = part_sch_before.get_execution_order()[w_id - 1]
            if len(w_exec_order) >= 1 and w_exec_order[-1] == j_id:
                w_id_to_last_ct_distribution[w_id - 1] = end_t_distributions[j_id]

        v_i_attr = [scheduled, last_performed, candidate, w_id, min_dur, max_dur, e_dur, min_ct, max_ct, e_ct]
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

        edge_predict_index=torch.LongTensor([edges_to_predict_fr_ids + edges_not_to_predict_fr_ids,
                                             edges_to_predict_to_ids + edges_not_to_predict_to_ids]),  # [2, P]
        edge_predict_label=torch.FloatTensor([1.]*len(edges_to_predict_fr_ids) + [0.]*len(edges_not_to_predict_fr_ids)),  # [P] float(1.0 = positive, 0.0 = negative)
        edge_predict_value=torch.FloatTensor(lengths_to_predict)
    )
    return data


def main():
    solutions_dir = '../data/solutions/'
    datasets_dir = '../data/datasets/'
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
            print(f'slices len: {len(slices)}')
            for i in range(len(slices) - 1):
                data = partial_sch_to_data(slices[i], slices[i + 1])
                torch.save(data, datasets_dir + solution_name[:-5] + f'data_slice_{i}.pt')


if __name__ == '__main__':
    main()
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

