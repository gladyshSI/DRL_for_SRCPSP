import copy
import itertools
import math
import random

import numpy as np
from tqdm import tqdm

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
def calculate_ovp_e(distributions: list[DiscreteDistribution]):
    return [d.e() for d in distributions]


class SingleMachine:
    def __init__(self, names: list, buffers: dict[int, int], ds: list[DiscreteDistribution]):
        if len(names) != len(ds):
            raise ValueError("names and ds must have the same length")
        self.names = names
        self.num_jobs = len(names)
        self.buffers = buffers
        self.ds = ds

    def set_ds(self, ds: list[DiscreteDistribution]):
        self.ds = ds

    @classmethod
    def with_uniform_distributions(cls, jobs_half_lengths: list[int]):
        names = copy.deepcopy(jobs_half_lengths)
        buffers = dict()
        ds = [DiscreteDistribution.set_uniform(-hl, hl) for hl in jobs_half_lengths]
        return cls(names, buffers, ds)

    @classmethod
    def with_norm_distributions(cls, jobs_half_lengths: list[int], es: list[float], ds: list[float]):
        if len(jobs_half_lengths) != len(es) or len(jobs_half_lengths) != len(ds):
            raise ValueError("jobs_half_lengths, es, and ds must have the same length")
        names = copy.deepcopy(jobs_half_lengths)
        buffers = dict()
        ds = [DiscreteDistribution.set_norm_approx(-jobs_half_lengths[i],
                                                   jobs_half_lengths[i],
                                                   e=es[i],
                                                   d=ds[i]) for i in range(len(jobs_half_lengths))]
        return cls(names, buffers, ds)

    @classmethod
    def with_exp_distributions(cls, jobs_half_lengths: list[int]):
        names = copy.deepcopy(jobs_half_lengths)
        buffers = dict()
        ds = [DiscreteDistribution.set_exp_approx(-hl, hl, lamb=1/hl) for hl in jobs_half_lengths]
        return cls(names, buffers, ds)

    @classmethod
    def with_trimodal_symmetric(cls, p0s: list[float]):
        names = [p0 for p0 in p0s]
        buffers = dict()
        return cls(names, buffers, [DiscreteDistribution.set_trimodal_symmetric(0, p0) for p0 in p0s])

    def permutate(self, permutation: list[int]):
        if set(permutation) != set(range(len(self.names))):
            raise ValueError("permutation must have unique values")
        self.names = [self.names[i] for i in permutation]
        self.ds = [self.ds[i] for i in permutation]

    def calculate_ovp_distributions(self):
        ovp_distributions = []
        prev_ovp_d = DiscreteDistribution.set_uniform(0, 0)
        for i in range(1, self.num_jobs + 1):
            bft = 0 if i - 1 not in self.buffers.keys() else self.buffers[i - 1]
            prev_ovp_d = (prev_ovp_d + self.ds[i - 1]).max_with(max(0, bft)) - bft
            ovp_distributions.append(copy.deepcopy(prev_ovp_d))
        return ovp_distributions

    def calculate_min_bound_ovp_distributions(self, group_by: int):
        if group_by > self.num_jobs or group_by < 1:
            raise ValueError("group_by must be >= 1 and <= num_jobs")
        groupe_by_count = 0
        ovp_distributions_grouped = []
        prev_ovp_d = DiscreteDistribution.set_uniform(0, 0)
        for i in range(1, self.num_jobs + 1):
            bft = 0 if i - 1 not in self.buffers.keys() else self.buffers[i - 1]
            prev_ovp_d = (prev_ovp_d + self.ds[i - 1]).max_with(max(0, bft)) - bft
            ovp_distributions_grouped.append(copy.deepcopy(prev_ovp_d))
            groupe_by_count += 1
            if groupe_by_count % group_by == 0:
                prev_ovp_d = DiscreteDistribution.set_uniform(0, 0)
        return ovp_distributions_grouped

    def calculate_min_bound_sum_ovp_e(self) -> int:
        sum_ovp_e = 0
        cs = [float(1. - self.ds[i].max_with(-1)[-1]) for i in range(1, self.num_jobs)]
        cs_prod = [math.prod(cs[:i]) for i in range(len(cs)+1)]
        for i in range(self.num_jobs):
            bft = 0 if i not in self.buffers.keys() else self.buffers[i]
            # print(f'i = {i}, cs_prod[i:] = {cs_prod[i:]}, sum(cs_prod[i:]) = {sum(cs_prod[i:])}, cs_prod[i-1] = {cs_prod[i]}')
            sum_ovp_e += self.ds[i].max_with(bft).e() * (sum(cs_prod[i:])/cs_prod[i])
        return sum_ovp_e

    def swap(self, i: int, j: int):
        if i < 0 or j < 0 or i >= self.num_jobs or j >= self.num_jobs:
            raise ValueError("Invalid index")
        self.names[i], self.names[j] = self.names[j], self.names[i]
        self.ds[i], self.ds[j] = self.ds[j], self.ds[i]

    def add_bft(self, after_i: int, length: int):
        if after_i not in self.buffers.keys():
            self.buffers[after_i] = 0
        self.buffers[after_i] += length

    def print_all_bft(self):
        print(self.buffers)


def draw_dist_list(dist_list: list[DiscreteDistribution]):
    n = len(dist_list)
    width = 0.9 / n

    fig, ax = plt.subplots()
    for i, d in enumerate(dist_list):
        x = np.array([i for i, _ in d.to_dict().items()])
        y = [p for _, p in d.to_dict().items()]
        ax.bar(x + i * width, y, width, label=f'{i}')
    plt.legend()
    plt.show()


def exp_sort(n: int):
    x = []
    y = []
    jobs_half_lengths: list[int] = [2] + [1 for _ in range(n - 1)]
    sm1 = SingleMachine
    ovp_e = sm1.calculate_ovp_e()
    for i in tqdm(range(1, n)):
        # print(sm1.jobs_half_lengths)
        # print(ovp_e)
        x.append(i)
        y.append(sum(ovp_e))
        sm1.swap(i - 1, i)
        ovp_e = sm1.calculate_ovp_e()

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()


def exp_last_ovp_e(n: int, k: int):
    jobs_half_lengths: list[int] = [n] + [_ for _ in range(n)]
    print(jobs_half_lengths)
    sm1 = SingleMachine
    y = [sm1.calculate_ovp_e()[k]]
    for i in tqdm(range(n - 1)):
        sm1.swap(i, i + 1)
        y.append(sm1.calculate_ovp_e()[k])
    print(sm1.names)
    fig, ax = plt.subplots()
    ax.plot(list(range(n)), y)
    plt.show()


def single_machine_permutator(sm: SingleMachine):
    new_sm = copy.deepcopy(sm)  # or deepcopy() if needed
    for perm in itertools.permutations(list(range(sm.num_jobs))):
        new_sm.permutate(list(perm))
        yield new_sm


def optimizator(sm: SingleMachine):
    n = sm.num_jobs
    min_sum = 10**5
    min_last = 10**5
    best_min_sum_perm = []
    best_min_last_perm = []
    print("n! = ", math.factorial(n))
    last_e = 0
    for sm_perm in tqdm(single_machine_permutator(sm)):
        sum_e = sum([d.e() for d in sm_perm.calculate_ovp_distributions()])
        last_e = sm_perm.calculate_ovp_distributions()[-1].e()
        if sum_e < min_sum:
            min_sum = sum_e
            best_min_sum_perm = sm_perm.names
        if last_e < min_last:
            min_last = last_e
            best_min_last_perm = sm_perm.names
    return min_sum, best_min_sum_perm, last_e, best_min_last_perm


def bft_place_optimizer(sm: SingleMachine, bft_size: int = 1):
    print("names:",  sm.names)
    # print("jobs num: ", sm.num_jobs)
    # print("sum ovp_e: ", sum(sm.calculate_ovp_e()))
    # print("last ovp_e: ", sm.calculate_ovp_distributions()[-1])

    best_last_ovp_e = 10 ** 10
    best_place_for_le = -1
    best_sum_ovp_e = 10 ** 10
    best_place_for_se = -1
    for i in range(sm.num_jobs):
        sm.add_bft(i, bft_size)
        sum_ovp_e = sum(sm.calculate_ovp_e())
        last_ovp_e = sm.calculate_ovp_e()[-1]
        # print(f'=====\nadded bft after job {i}')
        # print("ovp_distributions: ", sm.calculate_ovp_distributions())
        # print("sum ovp_e: ", sum_ovp_e)
        # print("last ovp distr.: ", sm.calculate_ovp_distributions()[-1])
        # print("last ovp e: ", last_ovp_e)
        if sum_ovp_e < best_sum_ovp_e:
            best_sum_ovp_e = sum_ovp_e
            best_place_for_se = i
        if last_ovp_e < best_last_ovp_e:
            best_last_ovp_e = last_ovp_e
            best_place_for_le = i
        sm.add_bft(i, -bft_size)
    print(f'best_last_ovp_e: {best_last_ovp_e}; best_place_for_le: {best_place_for_le}; best_sum_ovp_e: {best_last_ovp_e}: best_place_for_se: {best_place_for_se}')
    return best_last_ovp_e, best_place_for_le, best_sum_ovp_e, best_place_for_se


def exp_mirror(n: int):
    jobs_half_lengths: list[int] = list(np.random.randint(1, 10, size=n - 1)) + [1]
    print(jobs_half_lengths)
    sm1 = SingleMachine
    mirrored = sm1.names[-2::-1] + [sm1.names[-1]]
    print(mirrored)
    sm2 = SingleMachine
    print(sm1.calculate_ovp_e()[-1] - sm2.calculate_ovp_e()[-1])


def sev_machines_exp():
    machines_num = 5
    jobs_per_machine = 19
    jobs_n = jobs_per_machine * machines_num
    # jobs_x_list = [i * 0.5 / jobs_n for i in range(1, jobs_n + 1)]
    # jobs_x_list = [0.499 + 0.001 * i / jobs_n for i in range(1, jobs_n + 1)]
    jobs_x_list = [0.001 + 0.001 * i / jobs_n for i in range(1, jobs_n + 1)]
    jobs_p0s = list(map(lambda x: 1 - 2*x, jobs_x_list))

    # NOT EQUAL CASE:
    print("NOT EQUAL CASE")
    print([jobs_p0s[i:i+jobs_per_machine] for i in range(0, jobs_n, jobs_per_machine)])
    machines = [SingleMachine.with_trimodal_symmetric(jobs_p0s[i:i+jobs_per_machine]) for i in range(0, jobs_n, jobs_per_machine)]
    neq_sum_obj = sum([sum(calculate_ovp_e(machines[i].calculate_ovp_distributions())) for i in range(machines_num)])
    neq_last_obj = max([calculate_ovp_e(machines[i].calculate_ovp_distributions())[-1] for i in range(machines_num)])
    print(f'neq_sum_obj: {neq_sum_obj}, neq_last_obj: {neq_last_obj}')

    # EQUAL CASE:
    print("EQUAL CASE")
    print([jobs_p0s[i::machines_num] for i in range(machines_num)])
    machines = [SingleMachine.with_trimodal_symmetric(jobs_p0s[i::machines_num]) for i in
                range(machines_num)]
    eq_sum_obj = sum([sum(calculate_ovp_e(machines[i].calculate_ovp_distributions())) for i in range(machines_num)])
    eq_last_obj = max([calculate_ovp_e(machines[i].calculate_ovp_distributions())[-1] for i in range(machines_num)])
    print(f'eq_sum_obj: {eq_sum_obj}, eq_last_obj: {eq_last_obj}')

    message_sum = "neq_sum is better" if neq_sum_obj < eq_sum_obj else "eq_sum is better"
    message_last = "neq_last is better" if neq_last_obj < eq_last_obj else "eq_sum is better"
    print(message_sum)
    print(message_last)


def article_experiments(n_exp: int = 5, n_jobs: int = 10):
    hls = np.random.randint(1, 10, n_exp * n_jobs)
    hls = np.sort(hls.reshape(n_exp, n_jobs), axis=1)
    ds = np.random.random(n_exp * n_jobs) * 3
    ds = np.sort(ds.reshape(n_exp, n_jobs), axis=1)
    for exp_i in range(n_exp):
        print(f'exp_i: {exp_i} / {n_exp}')
        hl = hls[exp_i]
        sm_unif = SingleMachine.with_uniform_distributions(hl)
        d = ds[exp_i]
        sm_norm = SingleMachine.with_norm_distributions([10 for _ in range(n_jobs)],
                                                   [0. for _ in range(n_jobs)],
                                                   d)
        sm_exp = SingleMachine.with_exp_distributions(hl)
        distrs = [sm.calculate_ovp_distributions() for sm in [sm_unif, sm_norm, sm_exp]]
        f_sums = [sum([(d.e()) for d in ds]) for ds in distrs]
        f_sum_opts = [optimizator(sm)[0] for sm in [sm_unif, sm_norm, sm_exp]]
        for sm_id in range(len(f_sums)):
            if not math.isclose(f_sums[sm_id], f_sum_opts[sm_id], rel_tol=1e-9):
                print(d, sm_id, f_sums[sm_id], f_sum_opts[sm_id])


if __name__ == "__main__":
    # article_experiments(10, 8)
    v = np.array([-1, 0, 1])
    n_jobs = 4
    probs = [np.array([0.5 - x/(2 * n_jobs), 0.5, x/(2 * n_jobs)]) for x in range(n_jobs)]
    ds = [DiscreteDistribution(values=v, probs=p) for p in probs]
    sm = SingleMachine(names=[d.e() for d in ds], buffers={}, ds=ds)
    print("x = ", [x/(2 * n_jobs) for x in range(n_jobs)])
    print("names = ", sm.names)
    print("ds = ", sm.calculate_ovp_distributions())
    print("es = ", [d.e() for d in sm.calculate_ovp_distributions()])
    print("sum = ", sum([d.e() for d in sm.calculate_ovp_distributions()]))
    print(optimizator(sm)[0])
    # sm = SingleMachine.with_uniform_distributions([1, 5, 1, 5, 1, 5, 1])
    # print([round(float(d.e()), 2) for d in sm.calculate_ovp_distributions()])
    # print(sum([float(d.e()) for d in sm.calculate_ovp_distributions()]))
    # for i in range(sm.num_jobs):
    #     print("Put bft after", i)
    #     sm.add_bft(i, 1)
    #     print([round(float(d.e()), 2) for d in sm.calculate_ovp_distributions()])
    #     print(sum([float(d.e()) for d in sm.calculate_ovp_distributions()]))
    #     sm.add_bft(i, -1)
    #

    # sev_machines_exp()

    # sm = SingleMachine.with_trimodal_symmetric([0.1, 0.1, 0.1, 0.1])
    # print(sm.calculate_ovp_distributions())
    # print([float(x[0]) for x in sm.calculate_ovp_distributions()])

    # sm = SingleMachine.with_uniform_distributions([1, 2, 1])
    # print(sm.names)
    # print(sm.calculate_ovp_distributions()[-1].e() * 45)
    # sm.swap(0, 1)
    # print(sm.names)
    # print(sm.calculate_ovp_distributions()[-1].e() * 45)
    #
    # min_sum, best_min_sum_perm, last_e, best_min_last_perm = optimizator(sm)
    # print(f'best_min_sum_perm: {best_min_sum_perm}, min_sum: {min_sum * 45}')
    # print(f'best_min_last_perm: {best_min_last_perm}, last_e: {last_e * 45}')
    #
    # sm = SingleMachine.with_uniform_distributions([1, 1, 2])
    # l = [d.e() * 45 for d in sm.calculate_ovp_distributions()]
    # print(l, sum(l))
    # sm = SingleMachine.with_uniform_distributions([1, 2, 1])
    # l = [d.e() * 45 for d in sm.calculate_ovp_distributions()]
    # print(l, sum(l))
    # sm = SingleMachine.with_uniform_distributions([2, 1, 1])
    # l = [d.e() * 45 for d in sm.calculate_ovp_distributions()]
    # print(l, sum(l))


    # exp_sort(100)
    # exp_last_ovp_e(30, -1)
    # exp_mirror(10)

    # print(bft_place_optimizer([100, 1, 1, 1, 1, 1, 1], 1))

    # sm = SingleMachine([5, 1, 1, 1, 1, 1, 1])
    # draw_dist_list(sm.calculate_ovp_distributions())

    # N = 40
    # sm = SingleMachine.with_uniform_distributions([i+1 for i in range(N)])
    # print(sm.names)
    # print((calculate_ovp_e(sm.calculate_ovp_distributions())))
    # print(sum(calculate_ovp_e(sm.calculate_ovp_distributions())))
    # print((calculate_ovp_e(sm.calculate_min_bound_ovp_distributions(1))))
    # print(sum(calculate_ovp_e(sm.calculate_min_bound_ovp_distributions(1))))
    # print("lower bound = ", sm.calculate_min_bound_sum_ovp_e())
    # sm.permutate(list(range(N-1, -1, -1)))
    # print(sm.names)
    # print((calculate_ovp_e(sm.calculate_ovp_distributions())))
    # print(sum(calculate_ovp_e(sm.calculate_ovp_distributions())))
    # print((calculate_ovp_e(sm.calculate_min_bound_ovp_distributions(1))))
    # print(sum(calculate_ovp_e(sm.calculate_min_bound_ovp_distributions(1))))
    # print("lower bound = ", sm.calculate_min_bound_sum_ovp_e())
    #
    # N = 5
    # p0s = [(N-i)/(N+1) for i in range(N)]
    # sm = SingleMachine.with_trimodal_symmetric(p0s=p0s)
    # bft_place_optimizer(sm)
    # print(p0s)
    # print(sm.calculate_ovp_e()[-1])
    # sm.swap(3, 4)
    # print(sm.calculate_ovp_e()[-1])
    # print(optimizator(sm))
    # print("min_sum, best_min_sum_perm, last_e, best_min_last_perm")

    # N = 6
    # sm = SingleMachine.with_trimodal_symmetric(p0s=[(i + 1) / (N + 1) for i in range(N)])
    # print(bft_place_optimizer(sm))
    # print("best_last_ovp_e, best_place_for_le, best_sum_ovp_e, best_place_for_se")

    # jobs_half_lengths_1: list[int] = [1, 2, 3, 4, 5, 1]
    # jobs_half_lengths_2: list[int] = [5, 1, 2, 3, 1, 1]
    # sm1 = SingleMachine(jobs_half_lengths_1)
    # sm2 = SingleMachine(jobs_half_lengths_2)
    # print(sm1.calculate_ovp_e()[-1] - sm2.calculate_ovp_e()[-1])

# first_job_range = [(1, 121)]  # , (1, 5), (1, 7)]
# second_job_range = [(1, 363)]  # 7 + k) for k in range(0, 11, 2)]
# third_job_range = [(1, 3)]  # + k) for k in range(0, 11, 2)]
#
# a = dict()
# e = dict()
# for r1 in first_job_range:
#     for r2 in second_job_range:
#         for r3 in third_job_range:
#             print('~~~~~~~~~~~~~~~~', r1, r2, r3, '~~~~~~~~~~~~~~~~')
#             experiment_label = str(r1) + str(r2) + str(r3)
#             a[experiment_label] = dict()
#             e[experiment_label] = dict()
#             ds_good = [DiscreteDistribution.set_uniform(a, b) for a, b in [r1, r2, r3, (3, 3)]]
#             ds_bad = [DiscreteDistribution.set_uniform(a, b) for a, b in [r2, r1, r3, (3, 3)]]
#             for ds, text in [(ds_good, 'good'), (ds_bad, 'bad')]:
#                 print()
#                 print('############### ', text, ' ###############')
#                 init_sts = [0] + [round(d.e()) for d in ds]
#                 overlap_ds = []
#                 prev_ovp_d = DiscreteDistribution.set_uniform(0, 0)
#
#                 a[experiment_label][text] = dict()
#                 e[experiment_label][text] = dict()
#                 for i in range(1, len(ds)):
#                     # print(f'{i}: old prev_ovp_d: {prev_ovp_d}')
#                     # print(f'{i}: ds[i-1]: {ds[i-1]}')
#                     # print(f'{i}: init_sts[i]: {init_sts[i]}')
#                     # print(f'{i}: sum: {prev_ovp_d + ds[i-1]}')
#                     # print(f'{i}: max: {(prev_ovp_d + ds[i-1]).max_with(init_sts[i])}')
#                     prev_ovp_d = (prev_ovp_d + ds[i - 1]).max_with(init_sts[i]) - init_sts[i]
#
#                     a[experiment_label][text][i] = copy.deepcopy(prev_ovp_d)
#                     e[experiment_label][text][i] = prev_ovp_d.e()
#                     print(f'{i}: -- RES prev_ovp_d: {prev_ovp_d}')
#                     print(f'{i}: -- RES exp ovp: {prev_ovp_d.e():.4f}')
#                     overlap_ds.append(copy.deepcopy(prev_ovp_d))
# print('############### ###############')
# print(a)
# print(e)
# considered = set()
# for label, inner in a.items():
#     r1, r2, r3 = re.findall(r'\((\d+),\s*(\d+)\)', label)
#     if r1 + r2 not in considered:
#         considered.add(r1 + r2)
#     else:
#         continue
#     a_good = inner['good'][2]
#     a_bad = inner['bad'][2]
#     da = [(int(v), float(a_good[v] - a_bad[v])) for v in (a_good.values)]
#     print('LABEL: ', label, da)
#     cum_sums = []
#     for i in range(len(da)):
#         s = 0
#         for j in range(1, i+2):
#             c = sum(range(j + 1))
#             s += c * da[i + 1 - j][1]
#             # print(f'i: {i}, j: {j}, c: {c}, da[i+1-j][1]: {da[i + 1 - j][1]}, now s = {s}')
#         cum_sums.append(s)
#     print(f'CUM SUMS: {cum_sums}')


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
