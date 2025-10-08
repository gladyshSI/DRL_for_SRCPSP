import copy

import numpy as np

from lib.problem import Problem
from lib.graph import PrecedenceGraph
from lib.job import Job
from lib.distribution import DiscreteDistribution
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt installed
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import Axes3D

# V_NUM = 5
# g = PrecedenceGraph()
# g.random_network(number_of_nodes=V_NUM,
#                  start_n_node_range=(1, 2),
#                  end_n_node_range=(1, 2),
#                  seed=1)
# g_empty = PrecedenceGraph()
#
#
# jobs = [Job(job_id=i, init_duration=d.e(), dur_dist=d) for i, d in
#         enumerate([DiscreteDistribution.set_uniform(a, b) for a, b in [(1, 5), (1, 3), (3, 3), (3, 3)]])]


# p = Problem(graph=g, jobs=jobs, n_workers=1, n_jobs=len(jobs))
# p.graph.print_itself()
# print(len(p.jobs))
# print(p.jobs[0].print_itself())

################ MIPT BLOCK ################
first_job_range = [(1, 121)]  # , (1, 5), (1, 7)]
second_job_range = [(1, 363)]  # 7 + k) for k in range(0, 11, 2)]
third_job_range = [(1, 3)]  # + k) for k in range(0, 11, 2)]

a = dict()
e = dict()
for r1 in first_job_range:
    for r2 in second_job_range:
        for r3 in third_job_range:
            print('~~~~~~~~~~~~~~~~', r1, r2, r3, '~~~~~~~~~~~~~~~~')
            experiment_label = str(r1) + str(r2) + str(r3)
            a[experiment_label] = dict()
            e[experiment_label] = dict()
            ds_good = [DiscreteDistribution.set_uniform(a, b) for a, b in [r1, r2, r3, (3, 3)]]
            ds_bad = [DiscreteDistribution.set_uniform(a, b) for a, b in [r2, r1, r3, (3, 3)]]
            for ds, text in [(ds_good, 'good'), (ds_bad, 'bad')]:
                print()
                print('############### ', text, ' ###############')
                init_sts = [0] + [round(d.e()) for d in ds]
                overlap_ds = []
                prev_ovp_d = DiscreteDistribution.set_uniform(0, 0)

                a[experiment_label][text] = dict()
                e[experiment_label][text] = dict()
                for i in range(1, len(ds)):
                    # print(f'{i}: old prev_ovp_d: {prev_ovp_d}')
                    # print(f'{i}: ds[i-1]: {ds[i-1]}')
                    # print(f'{i}: init_sts[i]: {init_sts[i]}')
                    # print(f'{i}: sum: {prev_ovp_d + ds[i-1]}')
                    # print(f'{i}: max: {(prev_ovp_d + ds[i-1]).max_with(init_sts[i])}')
                    prev_ovp_d = (prev_ovp_d + ds[i - 1]).max_with(init_sts[i]) - init_sts[i]

                    a[experiment_label][text][i] = copy.deepcopy(prev_ovp_d)
                    e[experiment_label][text][i] = prev_ovp_d.e()
                    print(f'{i}: -- RES prev_ovp_d: {prev_ovp_d}')
                    print(f'{i}: -- RES exp ovp: {prev_ovp_d.e():.4f}')
                    overlap_ds.append(copy.deepcopy(prev_ovp_d))
print('############### ###############')
print(a)
print(e)
considered = set()
for label, inner in a.items():
    r1, r2, r3 = re.findall(r'\((\d+),\s*(\d+)\)', label)
    if r1 + r2 not in considered:
        considered.add(r1 + r2)
    else:
        continue
    a_good = inner['good'][2]
    a_bad = inner['bad'][2]
    da = [(int(v), float(a_good[v] - a_bad[v])) for v in (a_good.values)]
    print('LABEL: ', label, da)
    cum_sums = []
    for i in range(len(da)):
        s = 0
        for j in range(1, i+2):
            c = sum(range(j + 1))
            s += c * da[i + 1 - j][1]
            # print(f'i: {i}, j: {j}, c: {c}, da[i+1-j][1]: {da[i + 1 - j][1]}, now s = {s}')
        cum_sums.append(s)
    print(f'CUM SUMS: {cum_sums}')


# range_dif_1_2, range_3, x_vals, colors = [], [], [], []
#
# for label, inner in e.items():
#     r1, r2, r3 = re.findall(r'\((\d+),\s*(\d+)\)', label)
#     for text, e_vals in inner.items():
#         range_dif_1_2.append(int(r2[1]) - int(r1[1]))
#         range_3.append(int(r3[1]) - int(r3[0]))
#         x_vals.append(e_vals[3] + e_vals[1])
#         colors.append('green' if text == 'good' else 'red')
#         print(f'a: {int(r2[1]) - int(r1[1])}, b: {int(r3[1]) - int(r3[0])},  x: {e_vals[3] + e_vals[1]}')
#
# range_dif_1_2 = np.array(range_dif_1_2)
# range_3 = np.array(range_3)
# x_vals = np.array(x_vals)

# 3D scatter plot with color
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(range_dif_1_2, range_3, x_vals, c=colors, s=50)
#
# ax.set_xlabel("r2 - r1")
# ax.set_ylabel("r3")
# ax.set_zlabel("x")
#
# plt.show()
##############################################
