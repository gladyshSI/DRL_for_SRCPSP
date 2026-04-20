import multiprocessing
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from lib.distribution import DiscreteDistribution
from lib.problem import Problem
import multiprocessing as mp

# def e1(x: list):
#     return sum(x)
#
# def e2(x: list):
#     if len(x) == 2:
#         return x[0] * x[1]
#     elif len(x) > 2:
#         return e2(x[:-1]) + e1(x[:-1]) * x[-1]
#     else:
#         return 0
# def e3(x: list):
#     if len(x) == 3:
#         return x[0] * x[1] * x[2]
#     elif len(x) > 3: return e3(x[:-1]) + e2(x[:-1]) * x[-1]
#     else: return 0
#
# def e4(x: list):
#     if len(x) == 4:
#         return x[0] * x[1] * x[2] * x[3]
#     elif len(x) > 4: return e4(x[:-1]) + e3(x[:-1]) * x[-1]
#     else: return 0
#
# def obj(x: list):
#     return e1(x) - e2(x) + 2*e3(x) - 5*e4(x)
#
# if __name__ == '__main__':
#     n_points = 20
#     x = [0]+[(i+1)/(2*n_points) for i in range(n_points)]
#     for i1 in tqdm(range(n_points)):
#         for i2 in range(i1, n_points):
#             for i3 in range(i2, n_points):
#                 for i4 in range(i3, n_points):
#                     for i5 in range(i4, n_points):
#                         for i6 in range(i5, n_points):
#                             for i7 in range(i6, n_points):
#                                 for i8 in range(i7, n_points):
#                                     lx = [x[i1], x[i3], x[i5], x[i7]]
#                                     l = obj(lx)
#                                     r1x = [x[i1], x[i2], x[i7], x[i8]]
#                                     r1 = obj(r1x)
#                                     r2x = [x[i3], x[i4], x[i5], x[i6]]
#                                     r2 = obj(r2x)
#
#                                     if l > r1 and l > r2:
#                                         print(f'x: {x[i1], x[i2], x[i3], x[i4], x[i5], x[i6], x[i7], x[i8]}')
#                                         print(f'lx: {lx}, r1x: {r1x}, r2x: {r2x}')
#                                         print("SUCCESS:", l, r1, r2)

# x: (0.025, 0.025, 0.225, 0.225, 0.275, 0.275, 0.45, 0.45)
# lx: [0.025, 0.225, 0.275, 0.45], r1x: [0.025, 0.025, 0.45, 0.45], r2x: [0.225, 0.225, 0.275, 0.275]
# SUCCESS: 0.7309257812500001 0.7226171874999999 0.730857421875

# x: (0, 0, 0.25, 0.25, 0.25, 0.25, 0.475, 0.475)
# lx: [0, 0.25, 0.25, 0.475], r1x: [0, 0, 0.475, 0.475], r2x: [0.25, 0.25, 0.25, 0.25]
# SUCCESS: 0.734375 0.724375 0.73046875

values = np.array(range(-1, 3 + 1))
probs = np.exp(-(values - 1)**2 / 2.)
print(values, probs)
