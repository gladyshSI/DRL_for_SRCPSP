import multiprocessing
import os
import time

import numpy as np
from lib.distribution import DiscreteDistribution
from lib.problem import Problem
import multiprocessing as mp

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[8, 9, 10], [11, 12, 13]])
mask = [0, 2]
c = a[:, mask] + b[:, mask]
c = np.column_stack((c, np.array([1, 1])))
print(c.T)
print(np.max(c.T, axis=0))
