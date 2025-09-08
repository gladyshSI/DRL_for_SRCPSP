import copy

import numpy as np

from lib.problem import Problem
from lib.graph import PrecedenceGraph
from lib.job import Job
from lib.distribution import DiscreteDistribution

# V_NUM = 5
# g = PrecedenceGraph()
# g.random_network(number_of_nodes=V_NUM,
#                  start_n_node_range=(1, 2),
#                  end_n_node_range=(1, 2),
#                  seed=1)
# g_empty = PrecedenceGraph()
#
#
# jobs = [Job(job_id=i, init_duration=d.e(), dur_dist=d) for i, d in
#         enumerate([DiscreteDistribution.set_uniform(a, b) for a, b in [(1, 5), (1, 3), (3, 3), (3, 3)]])]


# p = Problem(graph=g, jobs=jobs, n_workers=1, n_jobs=len(jobs))
# p.graph.print_itself()
# print(len(p.jobs))
# print(p.jobs[0].print_itself())

################ MIPT BLOCK ################
ds_good = [DiscreteDistribution.set_uniform(a, b) for a, b in [(1, 3), (1, 5), (1, 3), (3, 3)]]
ds_bad = [DiscreteDistribution.set_uniform(a, b) for a, b in [(1, 5), (1, 3), (1, 3), (3, 3)]]
for ds, text in [(ds_good, 'good'), (ds_bad, 'bad')]:
    print()
    print('############### ', text, ' ###############')
    init_sts = [0] + [int(d.e()) for d in ds]
    overlap_ds = []
    prev_ovp_d = DiscreteDistribution.set_uniform(0, 0)
    for i in range(1, len(ds)):
        prev_ovp_d = (prev_ovp_d + ds[i-1]).max_with(init_sts[i]) - init_sts[i]
        print(f'{i}: prev_ovp_d: {prev_ovp_d}')
        print(f'{i}: exp ovp: {prev_ovp_d.e():.4f}')
        overlap_ds.append(copy.deepcopy(prev_ovp_d))

##############################################



