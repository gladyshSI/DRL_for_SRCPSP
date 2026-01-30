import numpy as np
from docplex.cp.model import *

from lib.problem import Problem
from lib.schedule import Schedule


def make_schedule_from_cplex_simple(problem: Problem, msol, rik) -> Schedule:
    sch = Schedule(problem)
    starting_times = dict()  # task_id -> st
    chosen_resources = dict()  # task_id -> res
    for i in range(problem.n_jobs):
        for k in range(problem.n_workers):
            var_sol = msol.get_var_solution(rik[(i, k)])
            if var_sol.is_present():
                starting_times[i] = var_sol.get_start()
                chosen_resources[i] = k

    for t in range(problem.n_jobs):
        sch.schedule_job(worker_id=chosen_resources[t], job_id=t, start_time=starting_times[t])
    return sch


def cplex_simple(problem: Problem, p=None, time_limit=2, log_output=True) -> (Schedule, float):

    tasks = list(range(problem.n_jobs))
    task_num = problem.n_jobs
    last_task = next(iter(problem.graph.get_end_ids()))
    resources = list(range(problem.n_workers))

    edge_list = []
    for i, js in problem.graph.get_copy_of_all_edges().items():
        for j in js:
            edge_list.append((i, j))

    if p == None:
        p = []
        for v in range(task_num):
            p.append(problem.jobs[v].get_duration())

    # MODEL
    mdl = CpoModel()

    # VARIABLES:
    xi = {}
    rik = {}
    for i in tasks:
        p_i = p[i]
        # est_i = rl_passes_list[i][0]
        # lst_i = makespan - rl_passes_list[i][1] + 1
        xi[i] = mdl.interval_var(size=p_i)  # start=[est_i, lst_i + p_i],
        for k in resources:
            rik[(i, k)] = mdl.interval_var(optional=True)

    # CONSTRAINTS:
    # end before start:
    for i, j in edge_list:
        mdl.add(mdl.end_before_start(xi[i], xi[j]))

    # alternative:
    for i in tasks:
        mdl.add(mdl.alternative(xi[i], [rik[(i, k)] for k in resources]))

    # no overlap:
    for k in resources:
        mdl.add(mdl.no_overlap([rik[(i, k)] for i in tasks]))

    # OBJECTIVE:
    # original:
    mdl.add(mdl.minimize(mdl.end_of(xi[last_task])))

    # Solve the model
    msol = mdl.solve(TimeLimit=time_limit, log_output=log_output)
    gap = msol.get_objective_gap()

    return make_schedule_from_cplex_simple(problem, msol, rik), gap