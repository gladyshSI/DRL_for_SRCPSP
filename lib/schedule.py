import copy
from collections import defaultdict, deque
from itertools import repeat, starmap

from lib.graph import PrecedenceGraph
from lib.job import Job
from lib.problem import Problem
from lib.distribution import Distribution, DiscreteDistribution, max_of_discr_distributions

import numpy as np
import numpy.typing as npt
import typing as tt
from tqdm import tqdm
import multiprocessing as mp
import time


class Schedule:
    def __init__(self, problem: Problem) -> None:
        self._problem = problem
        self._w_exec_seq: tt.List[tt.List[int]] = [[] for _ in
                                                   range(self._problem.n_workers)]  # worker_id: [job_id] SORTED BY TIME
        self._w_st_times: tt.List[tt.List[int]] = [[] for _ in range(
            self._problem.n_workers)]  # worker_id: [starting time] SORTED BY TIME
        self._scheduled: tt.Set[int] = set()  # {ids of scheduled job}
        self._candidates: tt.Set[
            int] = self._problem.graph.get_start_ids()  # jobs that can be scheduled on the next step
        self._j_schedule = defaultdict(tt.Tuple[int, int])  # job_id -> (worker_id, start_time)
        self._edges = dict()  # job_id -> successor_id -> time_lag
        self._reverse_edges = dict()  # job_id -> predecessor_id -> time_lag

        for fr_id, to_ids in self._problem.graph.get_copy_of_all_edges().items():
            self._edges[fr_id] = dict()
            for to_id in to_ids:
                self._edges[fr_id][to_id] = 0
                self._reverse_edges[to_id] = dict()
                self._reverse_edges[to_id][fr_id] = 0

    def clear(self) -> None:
        self._w_exec_seq = [[] for _ in range(self._problem.n_workers)]
        self._w_st_times = [[] for _ in range(self._problem.n_workers)]
        self._scheduled.clear()
        self._candidates = self._problem.graph.get_start_ids()
        self._j_schedule.clear()
        self._edges.clear()
        self._reverse_edges.clear()

        for fr_id, to_ids in self._problem.graph.get_copy_of_all_edges().items():
            self._edges[fr_id] = dict()
            for to_id in to_ids:
                self._edges[fr_id][to_id] = 0
                self._reverse_edges[to_id] = dict()
                self._reverse_edges[to_id][fr_id] = 0

    def nearest_left_and_right_job_ids(self, worker_id: int, time_to_sch: int) -> (int | None, int | None):
        idx = np.searchsorted(self._w_st_times[worker_id], time_to_sch)
        left_job_id = self._w_exec_seq[worker_id][idx - 1] if idx - 1 >= 0 else None
        if idx < len(self._w_exec_seq[worker_id]):
            if self._w_st_times[worker_id][idx] == time_to_sch:
                idx_right = np.searchsorted(self._w_st_times[worker_id], time_to_sch, side='right')
                right_job_id = self._w_exec_seq[worker_id][idx_right] if idx_right < len(
                    self._w_exec_seq[worker_id]) else None
            else:
                right_job_id = self._w_exec_seq[worker_id][idx]
        else:
            right_job_id = None
        return left_job_id, right_job_id

    def check_is_there_space(self, worker_id: int, time_to_sch: int, duration: int) -> bool:
        """Returns False if there is free space < duration in the schedule starting from time_to_sch."""
        # _w_st_times[worker_id][idx-1] <= time_to_sch < _w_st_times[worker_id][idx-1]
        idx = np.searchsorted(self._w_st_times[worker_id], time_to_sch, side='right')
        left_job_id = self._w_exec_seq[worker_id][idx - 1] if idx - 1 >= 0 else None
        left_st_time = self._w_st_times[worker_id][idx - 1] if left_job_id else 0
        left_duration = self._problem.jobs[left_job_id].get_duration() if left_job_id else 0
        right_job_id = self._w_exec_seq[worker_id][idx] if idx < len(self._w_exec_seq[worker_id]) else None
        right_st_time = self._w_st_times[worker_id][idx] if right_job_id else None
        return (left_st_time + left_duration <= time_to_sch and
                (not right_st_time or time_to_sch + duration <= right_st_time))

    def check_precedence_relationships(self, job_id: int, time_to_sch: int) -> bool:
        duration = self._problem.jobs[job_id].get_duration()
        end_time = time_to_sch + duration
        for suc_id in self._problem.graph.get_all_successors(job_id):
            if suc_id in self._scheduled and self._j_schedule[suc_id][1] < end_time:
                return False
        for pred_id in self._problem.graph.get_all_predecessors(job_id):
            pred_duration = self._problem.jobs[pred_id].get_duration()
            if pred_id in self._scheduled and self._j_schedule[pred_id][1] + pred_duration > time_to_sch:
                return False
        return True

    def schedule_job(self, worker_id: int, job_id: int, start_time: int | None = None):
        """Schedules job, if start_time = None, we schedule job at the end to the first possible time + time_lag"""
        if worker_id >= self._problem.n_workers or worker_id < 0:
            raise ValueError(f'Worker id {worker_id} out of range [0 - {self._problem.n_workers}]')
        if job_id >= self._problem.n_jobs or job_id < 0:
            raise ValueError(f'Job id {job_id} out of range [0 - {self._problem.n_jobs}]')
        if job_id not in self._candidates:
            raise ValueError(f'Task id {job_id} not in the candidates set: {self._candidates}')
        if start_time is not None and not self.check_is_there_space(worker_id, start_time,
                                                                    self._problem.jobs[job_id].get_duration()):
            raise ValueError(
                f'There is no space between tasks {self.nearest_left_and_right_job_ids(worker_id, start_time)}')
        if start_time is not None and not self.check_precedence_relationships(job_id, start_time):
            raise ValueError(f'There is precedence relationship error while scheduling {job_id}')

        # UPDATE: self._scheduled:
        self._scheduled.add(job_id)
        # UPDATE: self._w_exec_seq:    # worker_id: [job_id] SORTED BY TIME
        # UPDATE: self._w_st_times:    # worker_id: [starting time] SORTED BY TIME
        if start_time is not None:
            idx_to_insert = np.searchsorted(self._w_st_times[worker_id], start_time, side='right')
            self._w_st_times[worker_id].insert(idx_to_insert, start_time)
            self._w_exec_seq[worker_id].insert(idx_to_insert, job_id)
        else:
            left_neighbor_id = self._w_exec_seq[worker_id][-1] if len(self._w_exec_seq[worker_id]) > 0 else None
            predecessors = self._problem.graph.get_predecessors(job_id)
            if left_neighbor_id:
                predecessors.add(left_neighbor_id)
            max_end_time = 0
            for pred_id in predecessors:
                if pred_id in self._scheduled:
                    pred_st_time = self._j_schedule[pred_id][1]
                    pred_duration = self._problem.jobs[pred_id].get_duration()
                    max_end_time = max(pred_st_time + pred_duration, max_end_time)
            start_time = max_end_time
            self._w_st_times[worker_id].append(max_end_time)
            self._w_exec_seq[worker_id].append(job_id)
        # UPDATE: self._candidates:    # jobs that can be scheduled on the next step
        self._candidates.remove(job_id)
        for suc_id in self._problem.graph.get_successors(job_id):
            if self._problem.graph.get_predecessors(suc_id) <= self._scheduled:
                self._candidates.add(suc_id)
        # UPDATE: self._j_schedule:    # job_id -> (worker_id, start_time)
        self._j_schedule[job_id] = (worker_id, start_time)
        # UPDATE: self._edges:         # job_id -> successor_id -> time_lag
        # UPDATE: self._reverse_edges  # job_id -> predecessor_id -> time_lag
        # - update time lags for graph predecessors and successors:
        if start_time is not None:
            left_neighbor_id, right_neighbor_id = self.nearest_left_and_right_job_ids(worker_id, start_time)
        else:
            left_neighbor_id = self._w_exec_seq[worker_id][-2] if len(self._w_exec_seq[worker_id]) > 1 else None
            right_neighbor_id = None
        predecessors = self._problem.graph.get_predecessors(job_id)
        if left_neighbor_id:
            predecessors.add(left_neighbor_id)
        for pred_id in predecessors:
            if pred_id in self._scheduled:
                pred_start_time = self._j_schedule[pred_id][1]
                pred_duration = self._problem.jobs[pred_id].get_duration()
                time_lag = start_time - (pred_start_time + pred_duration)
                if pred_id not in self._edges.keys():
                    self._edges[pred_id] = dict()
                self._edges[pred_id][job_id] = time_lag
                if job_id not in self._reverse_edges.keys():
                    self._reverse_edges[job_id] = dict()
                self._reverse_edges[job_id][pred_id] = time_lag
        # - add right neighbor_id:
        if start_time and right_neighbor_id:
            duration = self._problem.jobs[job_id].get_duration()
            right_st_time = self._j_schedule[right_neighbor_id][1]
            time_lag = right_st_time - (start_time + duration)
            if job_id not in self._edges.keys():
                self._edges[job_id] = dict()
            self._edges[job_id][right_neighbor_id] = time_lag
            if right_neighbor_id not in self._reverse_edges.keys():
                self._reverse_edges[right_neighbor_id] = dict()
            self._reverse_edges[right_neighbor_id][job_id] = time_lag

    def get_first_jobs(self) -> tt.Set[int]:
        return set([w_sch[0] for w_sch in self._w_exec_seq if len(w_sch) > 0])

    def get_last_jobs(self) -> tt.Set[int]:
        return set([w_sch[-1] for w_sch in self._w_exec_seq if len(w_sch) > 0])

    def get_makespan(self):
        return np.max([w_st_t[-1] for w_st_t in self._w_st_times if len(w_st_t) > 0] + [0])

    def topological_sort(self, reverse: bool = False) -> tt.List[int]:
        graph = self._problem.graph
        start_nodes = graph.get_start_ids()
        edges = self._edges
        reverse_edges = self._reverse_edges

        order = list(start_nodes.intersection(self._scheduled))
        considered = start_nodes
        candidates = set()
        for i in order:
            candidates = candidates.union(set(edges[i].keys()).intersection(self._scheduled))
        while candidates:
            for c in candidates:
                if set(reverse_edges[c].keys()) <= considered:
                    order.append(c)
                    considered.add(c)
                    candidates.remove(c)
                    new_candidates = set() if c not in edges.keys() else set(edges[c].keys())
                    candidates = candidates.union(new_candidates.intersection(self._scheduled))
                    break
        return order if not reverse else list(reversed(order))

    def calculate_exact_overlap_distributions(self, error_value=1e-6) -> tt.Dict[int, Distribution]:
        end_times_distribution = dict()  # task_id -> Distribution
        overlap_distributions = dict()  # task_id -> Distribution

        order = self.topological_sort()
        for j in order:
            scheduled_start_time = self._j_schedule[j][1]
            predecessors_dict = dict() if j not in self._reverse_edges.keys() else self._reverse_edges[j]
            pred_delta_distributions = [DiscreteDistribution(values=np.array([0]), probs=np.array([1.]))]
            for pred, time_lag in predecessors_dict.items():
                # scheduled_pred_end_time = self._j_schedule[pred][1] + self._problem.jobs[pred].get_duration()
                delta_dist = end_times_distribution[pred].shift(-scheduled_start_time)
                pred_delta_distributions.append(delta_dist)

            overlap_distributions[j] = max_of_discr_distributions(pred_delta_distributions)
            overlap_distributions[j].normalize(epsilon=error_value)

            duration_distribution = self._problem.jobs[j].get_distribution()
            end_times_distribution[j] = scheduled_start_time + overlap_distributions[j] + duration_distribution


        return overlap_distributions

    @staticmethod
    def calculate_new_starting_times_after_right_shift(order: tt.List[int],
                                                       sch_j_to_id: tt.Dict[int, int],
                                                       j_schedule: tt.DefaultDict,
                                                       reverse_edges: tt.Dict[int, tt.Dict[int, int]],
                                                       new_durations: npt.NDArray) -> npt.NDArray:
        new_sts = np.empty(new_durations.shape, dtype=int)
        for j_id in order:
            # Do not intersect with self._scheduled since we suppose that job can be scheduled
            # only if all its predecessors are already scheduled
            predecessors_list = [] if j_id not in reverse_edges.keys() else (
                [sch_j_to_id[p] for p in reverse_edges[j_id].keys()])
            pred_ets = new_sts[:, predecessors_list] + new_durations[:, predecessors_list]
            new_sts[:, j_id] = np.max(pred_ets, axis=1, initial=j_schedule[j_id][1])  # We Don't move to the left
        return new_sts

    @staticmethod
    def calculate_ovp_distr_from_samples(ovp_s: npt.NDArray, j_id: int, error_value: float = 1e-6) -> tt.Tuple[int, DiscreteDistribution]:
        num_points = ovp_s.size
        values, counts = np.unique(ovp_s, return_counts=True)
        ovp_distribution = DiscreteDistribution(values=values, probs=counts / num_points)
        ovp_distribution.normalize(epsilon=error_value)
        return j_id, ovp_distribution

    def estimate_discr_overlap_distributions_by_monte_carlo(self, num_points: int, error_value=1e-6) -> tt.Dict[
        int, DiscreteDistribution]:
        t0 = time.time()

        id_to_sch_j = []
        sch_j_to_id = {}
        for i, j_id in enumerate(self._scheduled):
            id_to_sch_j.append(j_id)
            sch_j_to_id[j_id] = i

        scheduled_start_times = np.array([self._j_schedule[j_id][1] for j_id in id_to_sch_j])
        new_durations = np.empty(shape=(len(id_to_sch_j), num_points), dtype=np.uint8)
        for i, j_id in enumerate(id_to_sch_j):
            new_durations[i] = self._problem.jobs[j_id]._distribution.generate(num_points)
        order = self.topological_sort()

        t1 = time.time()
        print(f'new durations generation: {t1 - t0}')
        overlaps = self.calculate_new_starting_times_after_right_shift(order,
                                                                       sch_j_to_id,
                                                                       self._j_schedule,
                                                                       self._reverse_edges,
                                                                       new_durations.T)
        overlaps = np.maximum(0, overlaps - scheduled_start_times)  # Calculate overlaps
        t2 = time.time()
        print(f"overlaps computation: {t2 - t1}")

        overlap_distributions: tt.Dict[int, DiscreteDistribution] = dict()

        args = [(overlaps[:, sch_j_to_id[j_id]], j_id) for j_id in self._scheduled]
        for j_id, d in starmap(self.calculate_ovp_distr_from_samples, args):
            overlap_distributions[j_id] = d

        return overlap_distributions

    def is_complete(self) -> bool:
        return True if len(self._candidates) == 0 else False
