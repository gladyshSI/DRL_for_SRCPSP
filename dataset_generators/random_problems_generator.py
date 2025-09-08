import numpy as np

from lib.distribution import DiscreteDistribution
from lib.graph import PrecedenceGraph
from lib.job import Job
from lib.problem import Problem


def generate_random_problems(problems_num, jobs_num, workers_num,
                             save_dir='../data/problems/',
                             start_n_nodes_range=(1, 3),
                             end_n_nodes_range=(1, 3),
                             init_dur_range=(5, 10),
                             delta_dur_range=(1, 4),
                             seed=0):
    for _ in range(problems_num):
        g = PrecedenceGraph()
        g.random_network(number_of_nodes=jobs_num,
                         start_n_node_range=start_n_nodes_range,
                         end_n_node_range=end_n_nodes_range,
                         seed=seed)
        print(f'GRAPH:\n{g}\n ===================')

        init_durations = np.random.randint(low=init_dur_range[0], high=init_dur_range[1], size=jobs_num)
        print(f'INIT DURATIONS: {init_durations}')
        delta_durations = np.random.randint(low=delta_dur_range[0], high=delta_dur_range[1], size=jobs_num)
        print(f'DELTA DURATIONS: {delta_durations}')
        duration_ranges = zip(init_durations - delta_durations, init_durations + delta_durations)
        dur_distributions = [DiscreteDistribution.set_uniform(a, b) for a, b in duration_ranges]
        jobs = [Job(job_id=i, init_duration=int(init_durations[i]), dur_dist=d) for i, d in enumerate(dur_distributions)]

        p = Problem(graph=g, jobs=jobs, n_workers=workers_num, n_jobs=len(jobs))
        file_name = f'rand_p_{jobs_num}_j_{workers_num}_w_discrD_{init_dur_range[0]}_{init_dur_range[1]}_{delta_dur_range[0]}_{delta_dur_range[1]}-{_}.json'
        p.save_to_file(save_dir + file_name)


if __name__ == '__main__':
    save_dir = '../data/problems/'
    problems_num = 30
    start_n_nodes_range = (1, 3)
    init_dur_range = (5, 10)
    delta_dur_range = (1, 4)

    jobs_nums = [30, 60, 90]
    workers_nums = [4, 6, 7]

    for jobs_num, workers_num in zip(jobs_nums, workers_nums):
        generate_random_problems(problems_num=problems_num,
                                 jobs_num=jobs_num,
                                 workers_num=workers_num,
                                 save_dir=save_dir,
                                 start_n_nodes_range=start_n_nodes_range,
                                 init_dur_range=init_dur_range,
                                 delta_dur_range=delta_dur_range)
