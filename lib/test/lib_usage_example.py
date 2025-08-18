import numpy as np

from lib.problem import Problem
from lib.graph import PrecedenceGraph
from lib.job import Job
from lib.distribution import DiscreteDistribution

V_NUM = 5
g = PrecedenceGraph()
g.random_network(number_of_nodes=V_NUM,
                 start_n_node_range=(1, 2),
                 end_n_node_range=(1, 2),
                 seed=1)

jobs = [Job(i, d.e(), d) for i, d in enumerate([DiscreteDistribution.set_uniform(a, b) for a, b in
                                                zip(np.random.randint(1, 5, size=V_NUM),
                                                    np.random.randint(5, 10, size=V_NUM))])]

p = Problem(g, jobs)
p.graph.print_itself()
print(len(p.jobs))
print(p.jobs[0].print_itself())
