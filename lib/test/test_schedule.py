from unittest import TestCase

import numpy as np
import time

from lib.distribution import DiscreteDistribution
from lib.graph import PrecedenceGraph
from lib.job import Job
from lib.problem import Problem
from lib.schedule import Schedule


class TestSchedule(TestCase):
    def setUp(self):
        g = PrecedenceGraph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 4)
        g.add_edge(2, 4)
        g.add_edge(3, 5)
        g.add_edge(4, 5)

        dist0 = DiscreteDistribution(values=np.array([0]), probs=np.array([1.]))
        dist3 = DiscreteDistribution(values=np.array([2, 3, 4]), probs=np.array([1./3]*3))
        jobs = ([Job(0, 0, dist0)] +
                [Job(i, 3, dist3) for i in range(1, 5)] +
                [Job(5, 0, dist0)])

        problem = Problem(n_workers=2, n_jobs=len(jobs), graph=g, jobs=jobs)
        self.schedule = Schedule(problem=problem)

    def generate_schedule(self):
        self.schedule.schedule_job(worker_id=0, job_id=0)
        self.schedule.schedule_job(worker_id=0, job_id=2)
        self.schedule.schedule_job(worker_id=0, job_id=1)
        self.schedule.schedule_job(worker_id=1, job_id=3)
        self.schedule.schedule_job(worker_id=1, job_id=4)
        self.schedule.schedule_job(worker_id=0, job_id=5)

    def test_schedule_job(self):
        self.schedule.schedule_job(worker_id=0, job_id=0, start_time=1)
        self.schedule.schedule_job(worker_id=0, job_id=2, start_time=6)
        self.schedule.schedule_job(worker_id=0, job_id=1, start_time=3)

        self.assertEqual({0, 1, 2}, self.schedule._scheduled)
        self.assertEqual({3, 4}, self.schedule._candidates)
        self.assertRaises(ValueError, self.schedule.schedule_job, worker_id=0, job_id=3, start_time=1)  # Not enough space
        self.assertRaises(ValueError, self.schedule.schedule_job, worker_id=1, job_id=1, start_time=1)  # Already scheduled
        self.assertRaises(ValueError, self.schedule.schedule_job, worker_id=1, job_id=3, start_time=0)  # Precedence constraints
        self.assertRaises(ValueError, self.schedule.schedule_job, worker_id=1, job_id=5, start_time=1)  # Precedence constarints
        self.assertRaises(ValueError, self.schedule.schedule_job, worker_id=1, job_id=4, start_time=8)  # Precedence constarints
        self.assertRaises(ValueError, self.schedule.schedule_job, worker_id=2, job_id=5, start_time=1)  # There is no such worker
        self.assertRaises(ValueError, self.schedule.schedule_job, worker_id=-1, job_id=5, start_time=1)  # There is no such worker
        self.assertRaises(ValueError, self.schedule.schedule_job, worker_id=1, job_id=6, start_time=1)  # There is no such job
        self.assertRaises(ValueError, self.schedule.schedule_job, worker_id=1, job_id=-1, start_time=1)  # There is no such job
        self.schedule.schedule_job(worker_id=1, job_id=3, start_time=1)
        self.schedule.schedule_job(worker_id=1, job_id=4, start_time=9)
        self.schedule.schedule_job(worker_id=0, job_id=5, start_time=12)

        self.assertEqual([[0, 1, 2, 5], [3, 4]], self.schedule._w_exec_seq)
        self.assertEqual([[1, 3, 6, 12], [1, 9]], self.schedule._w_st_times)
        self.assertEqual(set(), self.schedule._candidates)
        self.assertEqual({
            0: (0, 1),
            1: (0, 3),
            2: (0, 6),
            3: (1, 1),
            4: (1, 9),
            5: (0, 12)}, dict(self.schedule._j_schedule))
        self.assertEqual({
            0: {1: 2, 2: 5, 3: 0},
            1: {2: 0, 4: 3},
            2: {4: 0, 5: 3},
            3: {4: 5, 5: 8},
            4: {5: 0}
        }, {i: dict(val) for i, val in self.schedule._edges.items()})
        self.assertEqual({
            1: {0: 2},
            2: {1: 0, 0: 5},
            3: {0: 0},
            4: {1: 3, 2: 0, 3: 5},
            5: {2: 3, 3: 8, 4: 0}
        }, {i: dict(val) for i, val in self.schedule._reverse_edges.items()})

    def test_schedule_job_on_top(self):
        self.generate_schedule()

        self.assertEqual([[0, 2, 1, 5], [3, 4]], self.schedule._w_exec_seq)
        self.assertEqual([[0, 0, 3, 9], [0, 6]], self.schedule._w_st_times)
        self.assertEqual(set(), self.schedule._candidates)
        self.assertEqual({
            0: (0, 0),
            1: (0, 3),
            2: (0, 0),
            3: (1, 0),
            4: (1, 6),
            5: (0, 9)}, dict(self.schedule._j_schedule))
        self.assertEqual({
            0: {1: 3, 2: 0, 3: 0},
            1: {4: 0, 5: 3},
            2: {1: 0, 4: 3},
            3: {4: 3, 5: 6},
            4: {5: 0}
        }, {i: dict(val) for i, val in self.schedule._edges.items()})
        self.assertEqual({
            1: {0: 3, 2: 0},
            2: {0: 0},
            3: {0: 0},
            4: {1: 0, 2: 3, 3: 3},
            5: {1: 3, 3: 6, 4: 0}
        }, {i: dict(val) for i, val in self.schedule._reverse_edges.items()})

    def test_get_first_tasks(self):
        self.assertEqual(set(), self.schedule.get_first_jobs())
        self.generate_schedule()
        self.assertEqual({0, 3}, self.schedule.get_first_jobs())

    def test_get_last_tasks(self):
        self.assertEqual(set(), self.schedule.get_last_jobs())
        self.generate_schedule()
        self.assertEqual({5, 4}, self.schedule.get_last_jobs())

    def test_get_makespan(self):
        self.assertEqual(0, self.schedule.get_makespan())
        self.generate_schedule()
        self.assertEqual(9, self.schedule.get_makespan())

    def test_topological_sort(self):
        self.assertEqual([], self.schedule.topological_sort())
        self.assertEqual([], self.schedule.topological_sort(reverse=True))
        self.schedule.schedule_job(worker_id=0, job_id=0, start_time=0)
        self.schedule.schedule_job(worker_id=0, job_id=2, start_time=0)
        self.schedule.schedule_job(worker_id=0, job_id=1, start_time=3)
        self.schedule.schedule_job(worker_id=1, job_id=4, start_time=6)

        self.assertEqual([0, 2, 1, 4], self.schedule.topological_sort())
        self.assertEqual([4, 1, 2, 0], self.schedule.topological_sort(reverse=True))

        self.schedule.schedule_job(worker_id=1, job_id=3, start_time=0)
        self.schedule.schedule_job(worker_id=0, job_id=5, start_time=9)

        order = self.schedule.topological_sort()
        rev_order = self.schedule.topological_sort(reverse=True)
        for fr_id, to_ids in self.schedule._edges.items():
            for to_id, _ in to_ids.items():
                self.assertTrue(order.index(fr_id) < order.index(to_id))
                self.assertTrue(rev_order.index(fr_id) > rev_order.index(to_id))

    def test_calculate_new_starting_times_after_right_shift(self):
        self.generate_schedule()
        new_durations = np.array([[0, 4, 2, 4, 3, 0]])
        order = self.schedule.topological_sort()
        print(f'order: {order}')
        self.assertEqual([0, 3, 0, 0, 7, 10],
                         list(self.schedule.calculate_new_starting_times_after_right_shift(order,
                                                                                           self.schedule._j_schedule,
                                                                                           self.schedule._reverse_edges,
                                                                                           new_durations)[0]))

    def test_calculate_exact_overlap_distributions(self):
        self.schedule.schedule_job(worker_id=0, job_id=0)
        self.schedule.schedule_job(worker_id=0, job_id=2)
        self.schedule.schedule_job(worker_id=0, job_id=1)
        self.schedule.schedule_job(worker_id=1, job_id=3)
        overlaps = {
            0: {0: 1.},
            1: {0: round(2. / 3, 8), 1: round(1. / 3, 8)},
            2: {0: 1.},
            3: {0: 1.}
        }
        print("Check partial schedule:")
        ovps = self.schedule.calculate_exact_overlap_distributions()
        # print("ovps = ", {i: {int(v): float(p.round(8)) for v, p in zip(val.values, val.probs)} for i, val in ovps.items()})
        self.assertEqual(overlaps, {i: {int(v): float(p.round(8)) for v, p in zip(val.values, val.probs)} for i, val in
                                    ovps.items()})

        self.schedule.schedule_job(worker_id=1, job_id=4)
        self.schedule.schedule_job(worker_id=0, job_id=5)
        overlaps = {
            0: {0: 1.},
            1: {0: round(2./3, 8), 1: round(1./3, 8)},
            2: {0: 1.},
            3: {0: 1.},
            4: {0: round(5./9, 8), 1: round(3./9, 8), 2: round(1./9, 8)},
            5: {0: round(13./27, 8), 1: round(9./27, 8), 2: round(4./27, 8), 3: round(1./27, 8)}
        }
        t1 = time.time()
        print("Check full schedule:")
        ovps = self.schedule.calculate_exact_overlap_distributions()
        t2 = time.time()
        print(t2 - t1)
        # print("ovps = ", {i: {int(v): float(p.round(8)) for v, p in zip(val.values, val.probs)} for i, val in ovps.items()})
        self.assertEqual(overlaps, {i: {int(v): float(p.round(8)) for v, p in zip(val.values, val.probs)} for i, val in ovps.items()})

    def test_estimate_discr_overlap_distributions_by_monte_carlo(self):
        self.schedule.schedule_job(worker_id=0, job_id=0)
        self.schedule.schedule_job(worker_id=0, job_id=2)
        self.schedule.schedule_job(worker_id=0, job_id=1)
        self.schedule.schedule_job(worker_id=1, job_id=3)
        overlaps = {
            0: {0: 1.},
            1: {0: round(2. / 3, 2), 1: round(1. / 3, 2)},
            2: {0: 1.},
            3: {0: 1.}
        }
        print("Check partial schedule ~ 3sec:")
        ovps = self.schedule.estimate_discr_overlap_distributions_by_monte_carlo(num_points=10 ** 7)
        self.assertEqual(overlaps, {i: {int(v): float(p.round(2)) for v, p in zip(val.values, val.probs)} for i, val in
                                    ovps.items()})

        self.schedule.schedule_job(worker_id=1, job_id=4)
        self.schedule.schedule_job(worker_id=0, job_id=5)
        overlaps = {
            0: {0: 1.},
            1: {0: round(2. / 3, 2), 1: round(1. / 3, 2)},
            2: {0: 1.},
            3: {0: 1.},
            4: {0: round(5. / 9, 2), 1: round(3. / 9, 2), 2: round(1. / 9, 2)},
            5: {0: round(13. / 27, 2), 1: round(9. / 27, 2), 2: round(4. / 27, 2), 3: round(1. / 27, 2)}
        }
        print("Check full schedule ~ 3sec:")
        t1 = time.time()
        ovps = self.schedule.estimate_discr_overlap_distributions_by_monte_carlo(num_points=10**7)
        t2 = time.time()
        print("Total time:", t2-t1)
        # print("ovps = ",
        #       {i: {int(v): float(p.round(3)) for v, p in zip(val.values, val.probs)} for i, val in ovps.items()})
        self.assertEqual(overlaps, {i: {int(v): float(p.round(2)) for v, p in zip(val.values, val.probs)} for i, val in ovps.items()})
