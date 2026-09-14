import numpy as np
from docplex.cp.model import *

from lib.problem import Problem
from lib.schedule import Schedule


def make_schedule_from_cplex_stochastic(problem: Problem, msol, r_iks, fr_scenario=0) -> Schedule:
    sch = Schedule(problem)
    starting_times = dict()  # task_id -> st
    chosen_resources = dict()  # task_id -> res
    for i in range(problem.n_jobs):
        for k in range(problem.n_workers):
            var_sol = msol.get_var_solution(r_iks[(i, k, fr_scenario)])
            if var_sol.is_present():
                starting_times[i] = var_sol.get_start()
                chosen_resources[i] = k

    for i in range(problem.n_jobs):
        sch.schedule_job(chosen_resources[i], i, starting_times[i])

    return sch

def cplex_stochastic_avg_d_makespan_bound(problem: Problem, obj: str, makespan=1000, scenarios_num=200, p=None, time_limit=2, log_output=True, execfile=None):
    tasks = list(problem.get_all_ids())
    last_task = next(iter(problem.get_end_ids()))
    resources = list(range(problem.get_machines_num()))

    edge_list = []
    for i, js in problem.get_copy_of_all_edges().items():
        for j in js:
            edge_list.append((i, j))

    if p == None:
        p = []
        for v in tasks:
            p.append(problem.get_duration(v))

    scenarios = [p]
    for _ in range(scenarios_num - 1):
        new_durations = problem.get_random_durations_from_distributions()
        ps = [new_durations[i] for i in tasks]
        scenarios.append(ps)

    # MODEL
    mdl = CpoModel()

    # VARIABLES:
    x_is = {}  # job i in the scenario s
    r_iks = {}  # job i assigned to the worker k at the scenario s
    for s in range(scenarios_num):
        for i in tasks:
            pis = scenarios[s][i]
            x_is[(i, s)] = mdl.interval_var(size=pis)
            for k in resources:
                r_iks[(i, k, s)] = mdl.interval_var(optional=True)

    # sequence variables:
    seq_ks = {(k, s): mdl.sequence_var([r_iks[(i, k, s)] for i in tasks],
                                       name="resource_" + str(k) + "_scenario_" + str(s))
              for s in range(scenarios_num) for k in resources}

    # CONSTRAINTS:
    # end before start:
    for i, j in edge_list:
        for s in range(scenarios_num):
            mdl.add(mdl.end_before_start(x_is[(i, s)], x_is[(j, s)]))

    # alternative:
    for i in tasks:
        for s in range(scenarios_num):
            mdl.add(mdl.alternative(x_is[(i, s)], [r_iks[(i, k, s)] for k in resources]))

    # no overlap:
    for k in resources:
        for s in range(scenarios_num):
            mdl.add(mdl.no_overlap(seq_ks[(k, s)]))

    # same sequences:
    for k in resources:
        for s in range(1, scenarios_num):
            mdl.add(mdl.same_sequence(seq_ks[(k, 0)], seq_ks[(k, s)]))

    # start before start (Right-Shift constraint)
    for j in tasks:
        for s in range(1, scenarios_num):
            mdl.add(mdl.start_before_start(x_is[(j, 0)], x_is[(j, s)]))

    # makespan bound
    mdl.add(mdl.end_of(x_is[(last_task, 0)]) == makespan)

    # OBJECTIVE:
    obj_s = {}
    if obj == "avg_rm":
        for s in range(1, scenarios_num):
            obj_s[s] = mdl.start_of(x_is[(last_task, s)]) - mdl.start_of(x_is[(last_task, 0)])
        agg_obj = mdl.sum([obj_s[s] for s in range(1, scenarios_num)])
    if obj == "max_rm":
        for s in range(1, scenarios_num):
            obj_s[s] = mdl.start_of(x_is[(last_task, s)]) - mdl.start_of(x_is[(last_task, 0)])
        agg_obj = mdl.max([obj_s[s] for s in range(1, scenarios_num)])
    if obj == "avg_exp_ovp":
        for s in range(1, scenarios_num):
            obj_s[s] = mdl.sum([(mdl.start_of(x_is[(i, s)]) - mdl.start_of(x_is[(i, 0)])) for i in tasks])
        agg_obj = mdl.sum([obj_s[s] for s in range(1, scenarios_num)])
    if obj == "max_exp_ovp":
        obj_i = {i: 0 for i in tasks}
        for i in tasks:
            for s in range(1, scenarios_num):
                obj_i[i] += mdl.start_of(x_is[(i, s)]) - mdl.start_of(x_is[(i, 0)])
        agg_obj = mdl.max([obj_i[i] for i in tasks])
    if obj == "max_max_ovp":
        for s in range(1, scenarios_num):
            obj_s[s] = mdl.max([(mdl.start_of(x_is[(i, s)]) - mdl.start_of(x_is[(i, 0)])) for i in tasks])
        agg_obj = mdl.max([obj_s[s] for s in range(1, scenarios_num)])

    mdl.add(mdl.minimize(agg_obj))

    # Solve the model
    if execfile is not None:
        msol = mdl.solve(execfile=execfile, TimeLimit=time_limit, log_output=log_output)
    else:
        msol = mdl.solve(TimeLimit=time_limit, log_output=log_output)
    gap = msol.get_objective_gap()

    return make_schedule_from_cplex_stochastic(problem, msol, r_iks, fr_scenario=0), gap