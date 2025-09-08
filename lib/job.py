import copy

import numpy as np

from lib.distribution import Distribution, DiscreteDistribution


class Job:
    def __init__(self, job_id: int, init_duration: int | float, dur_dist: Distribution):
        self._id: int = job_id
        self._duration: int | float = init_duration
        self._distribution: Distribution = dur_dist
        self._min_possible_dur: int | float = dur_dist.min_v()
        self._max_possible_dur: int | float = dur_dist.max_v()

    def get_id(self) -> int:
        return self._id

    def get_duration(self) -> int | float:
        return self._duration

    def get_distribution(self) -> Distribution:
        return copy.deepcopy(self._distribution)

    def get_min_possible_dur(self) -> int | float:
        return self._min_possible_dur

    def get_max_possible_dur(self) -> int | float:
        return self._max_possible_dur

    def set_id(self, job_id: int) -> None:
        self._id = job_id

    def set_duration(self, duration: int | float) -> None:
        if duration < 0:
            raise ValueError(f"Duration of the job {self._id} must not negative, but it is {duration}")
        if self._distribution[duration] != 0:
            raise ValueError(f"Duration of the job {self._id} must be contained in a distribution {self._distribution}")
        self._duration = duration

    def set_distribution(self, dur_dist: Distribution) -> None:
        self._distribution = dur_dist.normalize()

    def to_dict(self) -> dict:
        return {'id': self._id,
                'duration': self._duration,
                'distribution': self._distribution.to_dict()}

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data.get('id'), data.get('duration'), DiscreteDistribution.from_dict(data.get('distribution')))

    def __repr__(self):
        return f'id: {self._id}, dur: {self._duration}, distribution: {self._distribution}'

#
#
# def discrete_normal_dist(mean: int, std: float, tail: int) -> dict:
#     if tail >= mean or tail < 0:
#         raise ValueError(
#             "The tail must be smaller than the expectation and the tail must be greater than or equal to 0.")
#     N = 10000
#     norm_array = np.random.normal(mean, std, N)
#     dist = {i: 0. for i in range(mean - tail, mean + tail + 1, 1)}
#     for n in norm_array:
#         i = np.round(n)
#         i = min([mean + tail, i])
#         i = max([mean - tail, i])
#         dist[i] += 1
#     for i in range(mean - tail, mean + tail + 1, 1):
#         dist[i] = dist[i] / N
#     return dist
#
#
# def discrete_exponential_dist(mean: int, lb: int, tail: int) -> dict:
#     if tail >= mean + lb or tail < 0:
#         raise ValueError(
#             "The tail must be smaller than the expectation and the tail must be greater than or equal to 0.")
#     N = 10000
#     norm_array = np.random.exponential(mean, N)
#     ub = lb + mean + tail
#     dist = {i: 0. for i in range(lb, ub + 1, 1)}
#     for n in norm_array:
#         i = np.round(n + lb)
#         i = min([ub, i])
#         i = max([lb, i])
#         dist[i] += 1
#     for i in range(lb, ub + 1, 1):
#         dist[i] = dist[i] / N
#     return dist
