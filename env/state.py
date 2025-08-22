from lib.problem import Problem
import typing as tt
import numpy.typing as npt
import numpy as np


class State:
    def __init__(self, problem: Problem, schedule):
        self.problem = problem
        self.partial_sch = schedule

    def reset(self):
        pass

    def step(self, action):
        pass

    def encode(self):
        pass