import typing as tt
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import Box, Dict, Discrete

MAX_WORKERS = 5  # less than max value of  np.uint16
MAX_JOBS = 100  # less than max value of  np.uint16
D_TYPE = np.uint16


class State:
    def __init__(self):
        self.n_workers = 2
        self.n_jobs = 10
        self.edge_index = np.zeros((2, self.n_workers), dtype=D_TYPE)


class ScheduleEnv(gym.Env):
    spec = EnvSpec("ScheduleEnv-v0")

    def __init__(self):
        """Action is a pair of (worker_id, job_id)"""
        self.action_space = Box(low=np.array([0, 0]),
                                high=np.array([MAX_WORKERS, MAX_JOBS]),
                                shape=(2,), dtype=D_TYPE)
        self.observation_space = Dict(
            {"n_workers": Discrete(n=MAX_WORKERS),
             "n_jobs": Discrete(n=MAX_JOBS),
             "edge_index": Box(
                 low=0,
                 high=MAX_JOBS,
                 shape=(2, MAX_JOBS**2),
                 dtype=D_TYPE
             )}
        )

    def reset(self, *, seed: int | None = None, options: dict[str, tt.Any] | None = None):
        super().reset(seed=seed, options=options)

    def step(self, action_idx: int) -> tt.Tuple[np.ndarray, float, bool, bool, dict]:
        pass