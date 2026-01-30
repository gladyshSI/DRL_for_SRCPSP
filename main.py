import multiprocessing
import os
import random
import time

import numpy as np
import torch

from lib.distribution import DiscreteDistribution
from lib.problem import Problem
import multiprocessing as mp

result = ",".join(str(random.randint(4, 15)) for _ in range(52))
print(result)
