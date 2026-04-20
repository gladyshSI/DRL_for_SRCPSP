from unittest import TestCase

import numpy as np

from lib.distribution import DiscreteDistribution
from lib.graph import PrecedenceGraph
from lib.job import Job
from lib.problem import Problem, get_longest_paths


class Test(TestCase):

    def setUp(self) -> None:
        g = PrecedenceGraph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 4)
        g.add_edge(2, 4)
        g.add_edge(3, 5)
        g.add_edge(4, 5)

        dist0 = DiscreteDistribution(values=np.array([0]), probs=np.array([1.]))
        dist1 = DiscreteDistribution(values=np.array([1, 2, 3]), probs=np.array([1. / 3] * 3))
        dist2 = DiscreteDistribution(values=np.array([1, 2, 3]), probs=np.array([1. / 3] * 3))
        dist3 = DiscreteDistribution(values=np.array([2, 3, 4]), probs=np.array([1. / 3] * 3))
        dist4 = DiscreteDistribution(values=np.array([3, 4, 5]), probs=np.array([1. / 3] * 3))
        jobs = ([Job(0, 0, dist0)] +
                [Job(1, 2, dist1)] +
                [Job(2, 2, dist2)] +
                [Job(3, 4, dist3)] +
                [Job(4, 1, dist4)] +
                [Job(5, 0, dist0)])

        self.problem = Problem(n_workers=2, n_jobs=len(jobs), graph=g, jobs=jobs)
    def test_get_longest_paths(self):
        l_pths = get_longest_paths(self.problem)
        self.assertEqual({0: 0, 1: 0, 2: 0, 3: 0, 4: 2, 5: 4}, l_pths)
        l_pths_rev = get_longest_paths(self.problem, reverse=True)
        self.assertEqual({0: 4, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0}, l_pths_rev)

    def test_reverse(self):
        rev_problem = self.problem.reverse()
        self.assertEqual(self.problem.n_jobs, rev_problem.n_jobs)
        self.assertEqual(self.problem.n_workers, rev_problem.n_workers)
        self.assertEqual(self.problem.jobs, rev_problem.jobs)
        self.assertEqual(self.problem.graph.get_copy_of_all_edges(), rev_problem.graph.get_copy_of_all_reversed_edges())
        self.assertEqual(self.problem.graph.get_copy_of_all_reversed_edges(), rev_problem.graph.get_copy_of_all_edges())

