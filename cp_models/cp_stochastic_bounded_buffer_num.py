from docplex.cp.model import *

from lib.problem import Problem
from lib.schedule import Schedule


def make_schedule_from_cplex_stochastic_multi_mode(problem: Problem, msol, r_ik, x_im) -> Schedule:
    for im, xim in x_im.items():
        var_sol = msol.get_var_solution(xim)
        if var_sol.is_present():
            print("PRESENT (i, m)", im)

    sch = Schedule(problem)
    starting_times = dict()  # task_id -> st
    chosen_resources = dict()  # task_id -> res
    for i in problem.get_all_ids():
        for k in range(problem.get_machines_num()):
            var_sol = msol.get_var_solution(r_ik[(i, k)])
            if var_sol.is_present():
                starting_times[i] = var_sol.get_start()
                chosen_resources[i] = k

    for i in problem.get_all_ids():
        sch.schedule_task(chosen_resources[i], i, starting_times[i])

    return sch

def cplex_stochastic_multi_mode_buf(problem: Problem, obj: str, num_of_buf_modes, sum_of_buf=100, scenarios_num=200, p=None, time_limit=2, log_output=True) -> (Schedule, float):
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

    if num_of_buf_modes < 1:
        raise ValueError("num_of_buf_modes includes buf=0, so it should be at least 1")
    m = num_of_buf_modes  # Number of buffer modes (including 0)
    p_buf_modes = [[pi + buf for buf in range(m)] for pi in p]  # initial durations with different buffer times

    scenarios = [p]
    for _ in range(1, scenarios_num):
        new_durations = problem.get_random_durations_from_distributions()
        ps = [new_durations[i] for i in tasks]
        scenarios.append(ps)

    # MODEL
    mdl = CpoModel()
    # VARIABLES:
    # ==== for initial scenario ====
    x_i = {}  # job i in the initial scenario
    x_im = {}  # Chosen mode in the initial scenario
    r_ik = {}  # Chosen machine in the initial scenario
    for i in tasks:
        x_i[i] = mdl.interval_var()
        for mode in range(m):
            p_i_mode = p_buf_modes[i][mode]
            x_im[(i, mode)] = mdl.interval_var(size=p_i_mode, optional=True)
        for k in resources:
            r_ik[(i, k)] = mdl.interval_var(optional=True)
    seq_k = {k: mdl.sequence_var([r_ik[(i, k)] for i in tasks],
                                 name="init_resource_" + str(k) + "_scenario_" + str(0))
             for k in resources}

    # ==== for other scenarios ====
    x_is = {}  # job i in the scenario s
    r_iks = {}  # job i assigned to the worker k at the scenario s
    for s in range(1, scenarios_num):
        for i in tasks:
            pis = scenarios[s][i]
            x_is[(i, s)] = mdl.interval_var(size=pis)
            for k in resources:
                r_iks[(i, k, s)] = mdl.interval_var(optional=True)
    # sequence variables:
    seq_ks = {(k, s): mdl.sequence_var([r_iks[(i, k, s)] for i in tasks],
                                       name="resource_" + str(k) + "_scenario_" + str(s))
              for s in range(1, scenarios_num) for k in resources}

    # CONSTRAINTS:
    # ==== for initial scenario ====
    # alternative
    for i in tasks:
        mdl.add(mdl.alternative(x_i[i], [x_im[(i, mode)] for mode in range(m)]))  # Choose mode
        mdl.add(mdl.alternative(x_i[i], [r_ik[(i, k)] for k in resources]))  # Choose machine

    # end before start:
    for i, j in edge_list:
        mdl.add(mdl.end_before_start(x_i[i], x_i[j]))

    # no overlap:
    for k in resources:
        mdl.add(mdl.no_overlap(seq_k[k]))

    # ==== for other scenarios ====
    # alternative:
    for i in tasks:
        for s in range(1, scenarios_num):
            mdl.add(mdl.alternative(x_is[(i, s)], [r_iks[(i, k, s)] for k in resources]))

    # end before start:
    for i, j in edge_list:
        for s in range(1, scenarios_num):
            mdl.add(mdl.end_before_start(x_is[(i, s)], x_is[(j, s)]))

    # no overlap:
    for k in resources:
        for s in range(1, scenarios_num):
            mdl.add(mdl.no_overlap(seq_ks[(k, s)]))

    # same sequences:
    for k in resources:
        for s in range(1, scenarios_num):
            mdl.add(mdl.same_sequence(seq_k[k], seq_ks[(k, s)]))

    # start before start (Right-Shift constraint)
    for j in tasks:
        for s in range(1, scenarios_num):
            mdl.add(mdl.start_before_start(x_i[j], x_is[(j, s)]))

    # bound num of buffer times:
    max_buf_num = sum_of_buf
    mdl.add(mdl.sum(bj * mdl.presence_of(x_im[(i, bj)]) for i in tasks for bj in range(m)) == max_buf_num)

    # OBJECTIVE:
    obj_s = {}
    if obj == "avg_rm":
        for s in range(1, scenarios_num):
            obj_s[s] = mdl.start_of(x_is[(last_task, s)]) - mdl.start_of(x_i[last_task])
        agg_obj = mdl.sum([obj_s[s] for s in range(1, scenarios_num)])
    if obj == "max_rm":
        for s in range(1, scenarios_num):
            obj_s[s] = mdl.start_of(x_is[(last_task, s)]) - mdl.start_of(x_i[last_task])
        agg_obj = mdl.max([obj_s[s] for s in range(1, scenarios_num)])
    if obj == "avg_exp_ovp":
        for s in range(1, scenarios_num):
            obj_s[s] = mdl.sum([(mdl.start_of(x_is[(i, s)]) - mdl.start_of(x_i[i])) for i in tasks])
        agg_obj = mdl.sum([obj_s[s] for s in range(1, scenarios_num)])
    if obj == "max_exp_ovp":
        obj_i = {i: 0 for i in tasks}
        for i in tasks:
            for s in range(1, scenarios_num):
                obj_i[i] += mdl.start_of(x_is[(i, s)]) - mdl.start_of(x_i[i])
        agg_obj = mdl.max([obj_i[i] for i in tasks])
    if obj == "max_max_ovp":
        for s in range(1, scenarios_num):
            obj_s[s] = mdl.max([(mdl.start_of(x_is[(i, s)]) - mdl.start_of(x_i[i])) for i in tasks])
        agg_obj = mdl.max([obj_s[s] for s in range(1, scenarios_num)])

    mdl.add(mdl.minimize_static_lex([mdl.end_of(x_i[last_task]), agg_obj]))

    # Solve the model
    msol = mdl.solve(TimeLimit=time_limit, log_output=log_output)
    gap = msol.get_objective_gap()

    return make_schedule_from_cplex_stochastic_multi_mode(problem, msol, r_ik, x_im), gap