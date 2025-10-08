import os
import time

from cp_models.cp_stochastic_bounded_buffer_num import cplex_stochastic_multi_mode_buf
from lib.problem import Problem
from lib.schedule import Schedule, draw_schedule


def run_cp_stochastic_multi_mode_buf(problem: Problem, time_limit: int, parameters: dict) -> (Schedule, float, float):
    sum_of_buf = parameters['sum_of_buf']
    scenarios_num = parameters['N']
    max_b = parameters['max_b']
    obj = parameters['obj']
    start_time = time.time()
    sch, gap = cplex_stochastic_multi_mode_buf(problem,
                                               num_of_buf_modes=max_b+1,  # including buf=0
                                               sum_of_buf=sum_of_buf,
                                               scenarios_num=scenarios_num,
                                               obj=obj,
                                               time_limit=time_limit,
                                               log_output=True)
    end_time = time.time()

    return sch, gap, (end_time - start_time)

def main():
    problems_dir = '../data/problems/'
    solutions_dir = '../data/solutions/'
    problems_num = 30
    start_n_nodes_range = (1, 3)
    init_dur_range = (5, 10)
    delta_dur_range = (1, 4)

    jobs_nums = [30, 60, 90]
    workers_nums = [4, 6, 7]

    for jobs_num, workers_num in zip(jobs_nums, workers_nums):
        for problem_id in range(1, problems_num):
            problem_name = (f'rand_p_{jobs_num}_j_{workers_num}_w_discrD_'
                            f'{init_dur_range[0]}_{init_dur_range[1]}_'
                            f'{delta_dur_range[0]}_{delta_dur_range[1]}-{problem_id}.json')
            problem_path = os.path.join(problems_dir, problem_name)

            problem = Problem.read_from_file(problem_path)
            print(f' PROBLEM: \n {problem}')

            sch, gap, dt = run_cp_stochastic_multi_mode_buf(problem, 300,
                                                            {'sum_of_buf': round(0.2 * jobs_num),
                                                             'N': 30, 'max_b': 1, 'obj': 'avg_rm'})
            path_to_save = os.path.join(solutions_dir, 'BBr_' + problem_name)
            sch.save_to_file(path_to_save)


if __name__ == '__main__':
    main()
    # problems_dir = '../data/toy_problems/'
    # solutions_dir = '../data/toy_solutions/'
    # problem_name = 'toy1.json'
    # problem_path = os.path.join(problems_dir, problem_name)
    # problem = Problem.read_from_file(problem_path)
    # print(f' PROBLEM: \n {problem}')
    #
    # sch, gap, dt = run_cp_stochastic_multi_mode_buf(problem, 300,
    #                                                 {'sum_of_buf': round(0.2 * problem.n_jobs),
    #                                                  'N': 30, 'max_b': 1, 'obj': 'avg_rm'})
    # path_to_save = os.path.join(solutions_dir, 'BBr_' + problem_name)
    # sch.save_to_file(path_to_save)





