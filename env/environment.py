import typing as tt
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import Box, Dict, Discrete

from lib.schedule import Schedule

MAX_WORKERS = 5  # less than max value of  D_TYPE
MAX_JOBS = 100  # less than max value of  D_TYPE
D_TYPE = np.uint16


class ScheduleEnv(gym.Env):
    spec = EnvSpec("ScheduleEnv-v0")

    def __init__(self, schedule: Schedule):
        self.schedule = schedule
        """Action is a pair of (worker_id, job_id)"""
        self.action_space = Discrete(MAX_WORKERS * MAX_JOBS)
        self.observation_space = Dict(
            {"n_workers": Discrete(n=MAX_WORKERS),
             "n_jobs": Discrete(n=MAX_JOBS),
             "n_edges": Discrete(n=MAX_JOBS ** 2),
             "edge_index": Box(
                 low=0,
                 high=MAX_JOBS,
                 shape=(2, MAX_JOBS ** 2),
                 dtype=D_TYPE
             )}
        )
        self.reset()

    def reset(self, *, seed: int | None = None, options: dict[str, tt.Any] | None = None):
        super().reset(seed=seed, options=options)
        self.schedule.clear()

    def step(self, action_idx: int) -> tt.Tuple[dict, float, bool, bool, dict]:
        worker_id, job_id = self._convert_action_idx_to_worker_and_job(action_idx)
        self.schedule.schedule_job(worker_id, job_id)
        gym_obs = self._gym_obs_from_sch()
        reward = self._get_reward()
        done = self.schedule.is_complete()
        info = {}
        return gym_obs, reward, done, False, info

    def _convert_action_idx_to_worker_and_job(self, action_idx: int) -> tt.Tuple[int, int]:
        return action_idx // MAX_JOBS, action_idx % MAX_JOBS

    def _get_reward(self) -> float:
        overlaps_dict = self.schedule.calculate_exact_overlap_distributions()
        sum_of_avg_overlaps = sum((i.e() for j_id, i in overlaps_dict.items()))
        n = len(overlaps_dict)
        return sum_of_avg_overlaps / n

    def _gym_obs_from_sch(self) -> tt.Dict[str, any]:
        edge_index = np.zeros((2, MAX_JOBS ** 2), dtype=D_TYPE)
        idx = 0
        for fr_id, to_id_dict in self.schedule._edges.items():
            for to_id, _ in to_id_dict.items():
                edge_index[0, idx] = fr_id
                edge_index[1, idx] = to_id
                idx += 1
        return {
            "n_workers": self.schedule._problem.n_workers,
            "n_jobs": len(self.schedule._scheduled),
            "n_edges": idx,
            "edge_index": edge_index
        }
