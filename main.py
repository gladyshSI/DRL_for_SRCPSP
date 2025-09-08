import multiprocessing
import os
import time

import numpy as np
from lib.distribution import DiscreteDistribution
from lib.problem import Problem
import multiprocessing as mp

a = np.array([[1, 2, 3], [4, 5, 6]])
a *= -1
print(a)
