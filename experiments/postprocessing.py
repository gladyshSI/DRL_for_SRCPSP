# VERSION: checkpoints + initial schedule recovery + size-based time limits
import logging
import os
import pickle
import shutil
import sys
import tempfile
import time
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from docplex.cp.model import CpoModel
from docplex.cp.solution import CpoModelSolution
from docplex.cp.solver.solver_listener import CpoSolverListener

from cp_models.cp_stochastic_avg_d_makespan_bound import (
    make_schedule_from_cplex_stochastic,
)
from cp_models.cp_stochastic_bounded_buffer_num import (
    make_schedule_from_cplex_stochastic_multi_mode,
)
from lib.schedule import (
    Schedule,
    draw_schedule,
    get_max_possible_right_shift,
)


ScheduleFactory = Callable[[Any], Schedule]
CHECKPOINT_DIR_PREFIX = "docplex_cp_checkpoint_"


# ---------------------------------------------------------------------------
# Checkpoint support
# ---------------------------------------------------------------------------


def _write_checkpoint(
    checkpoint_path: Path,
    schedule: Schedule,
    gap: float | None = None,
    objective: Any = None,
) -> None:
    """Atomically save a schedule and optional solver information."""
    payload = {
        "schedule": schedule,
        "gap": gap,
        "objective": objective,
    }
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

    with temporary_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_path, checkpoint_path)


def _read_checkpoint(checkpoint_path: Path) -> tuple[Schedule, float | None]:
    """Load a previously saved schedule checkpoint."""
    with checkpoint_path.open("rb") as file:
        payload = pickle.load(file)

    schedule = payload.get("schedule")
    if schedule is None:
        raise ValueError(f"Checkpoint {checkpoint_path} contains no schedule")

    return schedule, payload.get("gap")


class SaveBestScheduleListener(CpoSolverListener):
    """Save intermediate feasible solutions while CP Optimizer is running."""

    def __init__(
        self,
        schedule_factory: ScheduleFactory,
        checkpoint_path: Path,
        min_interval: float = 2.0,
    ) -> None:
        if not callable(schedule_factory):
            raise TypeError("schedule_factory must be callable")
        if min_interval < 0:
            raise ValueError("min_interval cannot be negative")

        self.schedule_factory = schedule_factory
        self.checkpoint_path = checkpoint_path
        self.min_interval = min_interval
        self._last_write = 0.0

    def result_found(self, solver: Any, msol: Any) -> None:
        if not msol.is_solution():
            return

        now = time.monotonic()
        if now - self._last_write < self.min_interval:
            return

        try:
            schedule = self.schedule_factory(msol)
            _write_checkpoint(
                checkpoint_path=self.checkpoint_path,
                schedule=schedule,
                gap=msol.get_objective_gap(),
                objective=msol.get_objective_values(),
            )
            self._last_write = now
        except Exception:
            # Checkpoint errors must never terminate the optimization itself.
            logging.exception(
                "Could not save intermediate CP solution to %s",
                self.checkpoint_path,
            )


def _solve_with_checkpoint(
    *,
    mdl: CpoModel,
    schedule_factory: ScheduleFactory,
    initial_schedule: Schedule,
    time_limit: int,
    log_output: bool | None,
    execfile: str | None = None,
    checkpoint_interval: float = 2.0,
) -> tuple[Schedule, float | None]:
    """
    Solve a model and return ``(schedule, gap)``.

    The initial schedule is checkpointed before CP Optimizer starts. Each
    reported intermediate solution replaces it. If solving fails or finishes
    without a feasible solution, the latest valid checkpoint is returned.
    All checkpoint files created by this call are deleted before it exits.
    """
    checkpoint_directory = Path(
        tempfile.mkdtemp(prefix=CHECKPOINT_DIR_PREFIX)
    )
    checkpoint_path = checkpoint_directory / "best_solution.pkl"

    try:
        # This guarantees a recoverable schedule even when the solver does not
        # improve or report the supplied starting point.
        _write_checkpoint(
            checkpoint_path=checkpoint_path,
            schedule=initial_schedule,
        )

        listener = SaveBestScheduleListener(
            schedule_factory=schedule_factory,
            checkpoint_path=checkpoint_path,
            min_interval=checkpoint_interval,
        )
        mdl.add_solver_listener(listener)

        solve_kwargs: dict[str, Any] = {
            "TimeLimit": time_limit,
            "log_output": log_output,
            # Passing the leaf context name makes solve() use search_next(),
            # which enables intermediate-result callbacks.
            "solve_with_search_next": True,
        }
        if execfile is not None:
            solve_kwargs["execfile"] = execfile

        try:
            msol = mdl.solve(**solve_kwargs)

            if msol is None or not msol.is_solution():
                raise RuntimeError(
                    "CP Optimizer finished without a feasible solution"
                )

            return schedule_factory(msol), msol.get_objective_gap()

        except (Exception, KeyboardInterrupt):
            logging.exception(
                "CP Optimizer failed; returning the latest checkpoint"
            )
            return _read_checkpoint(checkpoint_path)

    finally:
        # Removes the checkpoint, a possible .tmp file, and the directory.
        shutil.rmtree(checkpoint_directory, ignore_errors=True)


# ---------------------------------------------------------------------------
# Shared model helpers
# ---------------------------------------------------------------------------


def _get_edge_list(problem: Any) -> list[tuple[int, int]]:
    return [
        (source, target)
        for source, targets in problem.graph.get_copy_of_all_edges().items()
        for target in targets
    ]


def _get_processing_times(
    problem: Any,
    tasks: Sequence[int],
    processing_times: Sequence[int] | None,
) -> list[int]:
    if processing_times is not None:
        return list(processing_times)
    return [problem.jobs[task].get_duration() for task in tasks]


def _generate_scenarios(
    problem: Any,
    tasks: Sequence[int],
    processing_times: Sequence[int],
    scenarios_num: int,
) -> np.ndarray:
    if scenarios_num < 2:
        raise ValueError("scenarios_num must be at least 2")

    scenarios = np.empty((scenarios_num, problem.n_jobs), dtype=int)
    scenarios[0, :] = np.asarray(processing_times, dtype=int)

    for task in tasks:
        scenarios[1:, task] = (
            problem.jobs[task]
            ._distribution.generate(scenarios_num - 1)
            .astype(int)
        )

    return scenarios


def _build_aggregate_objective(
    *,
    mdl: CpoModel,
    obj: str,
    tasks: Sequence[int],
    last_task: int,
    scenarios_num: int,
    initial_interval: Callable[[int], Any],
    scenario_interval: Callable[[int, int], Any],
) -> Any:
    """Build the common robustness objective used by both models."""
    supported = {
        "avg_rm",
        "max_rm",
        "avg_exp_ovp",
        "max_exp_ovp",
        "max_max_ovp",
    }
    if obj not in supported:
        raise ValueError(
            f"Unknown objective {obj!r}. Expected one of {sorted(supported)}"
        )

    scenario_ids = range(1, scenarios_num)

    if obj in {"avg_rm", "max_rm"}:
        values = [
            mdl.start_of(scenario_interval(last_task, scenario))
            - mdl.start_of(initial_interval(last_task))
            for scenario in scenario_ids
        ]
        return mdl.sum(values) if obj == "avg_rm" else mdl.max(values)

    if obj == "avg_exp_ovp":
        values = [
            mdl.sum(
                mdl.start_of(scenario_interval(task, scenario))
                - mdl.start_of(initial_interval(task))
                for task in tasks
            )
            for scenario in scenario_ids
        ]
        return mdl.sum(values)

    if obj == "max_exp_ovp":
        task_values = [
            mdl.sum(
                mdl.start_of(scenario_interval(task, scenario))
                - mdl.start_of(initial_interval(task))
                for scenario in scenario_ids
            )
            for task in tasks
        ]
        return mdl.max(task_values)

    scenario_values = [
        mdl.max(
            mdl.start_of(scenario_interval(task, scenario))
            - mdl.start_of(initial_interval(task))
            for task in tasks
        )
        for scenario in scenario_ids
    ]
    return mdl.max(scenario_values)


# ---------------------------------------------------------------------------
# Multi-mode bounded-buffer post-processing model
# ---------------------------------------------------------------------------


def cp_multi_mode_buf_postprocessing(
    start_sol: Schedule,
    obj: str,
    num_of_buf_modes: int,
    search_window: int,
    scenarios_num: int = 200,
    p: Sequence[int] | None = None,
    time_limit: int = 2,
    log_output: bool | None = True,
    execfile: str | None = None,
    checkpoint_interval: float = 2.0,
) -> tuple[Schedule, float | None]:
    problem = start_sol.get_problem()
    tasks = list(range(problem.n_jobs))
    resources = list(range(problem.n_workers))
    last_task = next(iter(problem.graph.get_end_ids()))
    edge_list = _get_edge_list(problem)
    processing_times = _get_processing_times(problem, tasks, p)

    if num_of_buf_modes < 1:
        raise ValueError(
            "num_of_buf_modes includes buffer mode 0, so it must be at least 1"
        )

    modes_num = num_of_buf_modes
    processing_times_by_mode = [
        [duration + buffer for buffer in range(modes_num)]
        for duration in processing_times
    ]
    scenarios = _generate_scenarios(
        problem,
        tasks,
        processing_times,
        scenarios_num,
    )

    mdl = CpoModel()
    start_solution = CpoModelSolution()

    # Initial-scenario variables.
    x_i: dict[int, Any] = {}
    x_im: dict[tuple[int, int], Any] = {}
    r_ik: dict[tuple[int, int], Any] = {}
    real_buf_sum = 0

    for task in tasks:
        start_time = start_sol.get_scheduled_start_time(task)
        duration = problem.jobs[task].get_duration()
        worker_id = start_sol.get_scheduled_worker(task)

        buffer = min(
            get_max_possible_right_shift(start_sol, task),
            modes_num - 1,
        )
        real_buf_sum += buffer

        lower_bound = max(0, start_time - search_window)
        upper_bound = min(
            start_sol.get_makespan() - duration,
            start_time + search_window,
        )
        x_i[task] = mdl.interval_var(start=(lower_bound, upper_bound))
        start_solution.add_interval_var_solution(
            x_i[task],
            start=start_time,
        )

        for mode in range(modes_num):
            x_im[(task, mode)] = mdl.interval_var(
                size=processing_times_by_mode[task][mode],
                optional=True,
            )

        for resource in resources:
            r_ik[(task, resource)] = mdl.interval_var(optional=True)
            if resource == worker_id:
                start_solution.add_interval_var_solution(
                    r_ik[(task, resource)],
                    start=start_time,
                )
            else:
                start_solution.add_interval_var_solution(
                    r_ik[(task, resource)],
                    presence=False,
                )

    initial_sequences = {
        resource: mdl.sequence_var(
            [r_ik[(task, resource)] for task in tasks],
            name=f"init_resource_{resource}_scenario_0",
        )
        for resource in resources
    }

    # The original model excludes the final job's mandatory largest buffer.
    real_buf_sum -= modes_num - 1

    # Other-scenario variables.
    x_is: dict[tuple[int, int], Any] = {}
    r_iks: dict[tuple[int, int, int], Any] = {}

    for scenario in range(1, scenarios_num):
        for task in tasks:
            x_is[(task, scenario)] = mdl.interval_var(
                size=int(scenarios[scenario, task])
            )
            for resource in resources:
                r_iks[(task, resource, scenario)] = mdl.interval_var(
                    optional=True
                )

    scenario_sequences = {
        (resource, scenario): mdl.sequence_var(
            [r_iks[(task, resource, scenario)] for task in tasks],
            name=f"resource_{resource}_scenario_{scenario}",
        )
        for scenario in range(1, scenarios_num)
        for resource in resources
    }

    # Initial-scenario constraints.
    for task in tasks:
        mdl.add(
            mdl.alternative(
                x_i[task],
                [x_im[(task, mode)] for mode in range(modes_num)],
            )
        )
        mdl.add(
            mdl.alternative(
                x_i[task],
                [r_ik[(task, resource)] for resource in resources],
            )
        )

    for source, target in edge_list:
        mdl.add(mdl.end_before_start(x_i[source], x_i[target]))

    for resource in resources:
        mdl.add(mdl.no_overlap(initial_sequences[resource]))

    # Other-scenario constraints.
    for scenario in range(1, scenarios_num):
        for task in tasks:
            mdl.add(
                mdl.alternative(
                    x_is[(task, scenario)],
                    [
                        r_iks[(task, resource, scenario)]
                        for resource in resources
                    ],
                )
            )

        for source, target in edge_list:
            mdl.add(
                mdl.end_before_start(
                    x_is[(source, scenario)],
                    x_is[(target, scenario)],
                )
            )

        for resource in resources:
            mdl.add(
                mdl.no_overlap(scenario_sequences[(resource, scenario)])
            )
            mdl.add(
                mdl.same_sequence(
                    initial_sequences[resource],
                    scenario_sequences[(resource, scenario)],
                )
            )

        for task in tasks:
            mdl.add(
                mdl.start_before_start(
                    x_i[task],
                    x_is[(task, scenario)],
                )
            )

    if modes_num > 1:
        mdl.add(
            mdl.sum(
                mode * mdl.presence_of(x_im[(task, mode)])
                for task in tasks
                for mode in range(modes_num)
            )
            >= real_buf_sum
        )

    for task in tasks:
        mdl.add(mdl.end_of(x_i[task]) <= start_sol.get_makespan())

    aggregate_objective = _build_aggregate_objective(
        mdl=mdl,
        obj=obj,
        tasks=tasks,
        last_task=last_task,
        scenarios_num=scenarios_num,
        initial_interval=lambda task: x_i[task],
        scenario_interval=lambda task, scenario: x_is[(task, scenario)],
    )
    mdl.add(
        mdl.minimize_static_lex(
            [mdl.end_of(x_i[last_task]), aggregate_objective]
        )
    )

    mdl.set_starting_point(start_solution)

    schedule_factory = lambda msol: (
        make_schedule_from_cplex_stochastic_multi_mode(
            problem,
            msol,
            r_ik,
            x_im,
        )
    )

    return _solve_with_checkpoint(
        mdl=mdl,
        schedule_factory=schedule_factory,
        initial_schedule=start_sol,
        time_limit=time_limit,
        log_output=log_output,
        execfile=execfile,
        checkpoint_interval=checkpoint_interval,
    )


# ---------------------------------------------------------------------------
# Makespan-bound post-processing model
# ---------------------------------------------------------------------------


def cp_makespan_bound_postprocessing(
    start_sol: Schedule,
    search_window: int,
    scenarios_num: int,
    obj: str,
    time_limit: int,
    p: Sequence[int] | None = None,
    log_output: bool | None = True,
    execfile: str | None = None,
    checkpoint_interval: float = 2.0,
) -> tuple[Schedule, float | None]:
    # search_window is retained for API compatibility. The original active
    # model did not apply a start-time window in this function.
    del search_window

    problem = start_sol.get_problem()
    tasks = list(range(problem.n_jobs))
    resources = list(range(problem.n_workers))
    last_task = next(iter(problem.graph.get_end_ids()))
    edge_list = _get_edge_list(problem)
    processing_times = _get_processing_times(problem, tasks, p)
    scenarios = _generate_scenarios(
        problem,
        tasks,
        processing_times,
        scenarios_num,
    )

    mdl = CpoModel()
    start_solution = CpoModelSolution()

    x_is: dict[tuple[int, int], Any] = {}
    r_iks: dict[tuple[int, int, int], Any] = {}

    for scenario in range(scenarios_num):
        for task in tasks:
            start_time = start_sol.get_scheduled_start_time(task)
            worker_id = start_sol.get_scheduled_worker(task)

            x_is[(task, scenario)] = mdl.interval_var(
                size=int(scenarios[scenario, task])
            )

            if scenario == 0:
                start_solution.add_interval_var_solution(
                    x_is[(task, scenario)],
                    start=start_time,
                )

            for resource in resources:
                r_iks[(task, resource, scenario)] = mdl.interval_var(
                    optional=True
                )

                if scenario == 0 and resource == worker_id:
                    start_solution.add_interval_var_solution(
                        r_iks[(task, resource, scenario)],
                        start=start_time,
                    )
                elif scenario == 0:
                    start_solution.add_interval_var_solution(
                        r_iks[(task, resource, scenario)],
                        presence=False,
                    )

    sequences = {
        (resource, scenario): mdl.sequence_var(
            [r_iks[(task, resource, scenario)] for task in tasks],
            name=f"resource_{resource}_scenario_{scenario}",
        )
        for scenario in range(scenarios_num)
        for resource in resources
    }

    for scenario in range(scenarios_num):
        for source, target in edge_list:
            mdl.add(
                mdl.end_before_start(
                    x_is[(source, scenario)],
                    x_is[(target, scenario)],
                )
            )

        for task in tasks:
            mdl.add(
                mdl.alternative(
                    x_is[(task, scenario)],
                    [
                        r_iks[(task, resource, scenario)]
                        for resource in resources
                    ],
                )
            )

        for resource in resources:
            mdl.add(mdl.no_overlap(sequences[(resource, scenario)]))

    for scenario in range(1, scenarios_num):
        for resource in resources:
            mdl.add(
                mdl.same_sequence(
                    sequences[(resource, 0)],
                    sequences[(resource, scenario)],
                )
            )

        for task in tasks:
            mdl.add(
                mdl.start_before_start(
                    x_is[(task, 0)],
                    x_is[(task, scenario)],
                )
            )

    mdl.add(
        mdl.end_of(x_is[(last_task, 0)]) == start_sol.get_makespan()
    )

    aggregate_objective = _build_aggregate_objective(
        mdl=mdl,
        obj=obj,
        tasks=tasks,
        last_task=last_task,
        scenarios_num=scenarios_num,
        initial_interval=lambda task: x_is[(task, 0)],
        scenario_interval=lambda task, scenario: x_is[(task, scenario)],
    )
    mdl.add(mdl.minimize(aggregate_objective))

    mdl.set_starting_point(start_solution)

    schedule_factory = lambda msol: make_schedule_from_cplex_stochastic(
        problem,
        msol,
        r_iks,
        fr_scenario=0,
    )

    return _solve_with_checkpoint(
        mdl=mdl,
        schedule_factory=schedule_factory,
        initial_schedule=start_sol,
        time_limit=time_limit,
        log_output=log_output,
        execfile=execfile,
        checkpoint_interval=checkpoint_interval,
    )


# ---------------------------------------------------------------------------
# Example cluster entry point
# ---------------------------------------------------------------------------
import argparse
import json
import logging
import time
from pathlib import Path


def list_solution_files(raw_solutions_path: str) -> list[Path]:
    raw_solutions_dir = Path(raw_solutions_path)
    if not raw_solutions_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {raw_solutions_dir}"
        )
    # Sorting makes the mapping deterministic.
    return sorted(p for p in raw_solutions_dir.iterdir() if p.is_file())


def build_job_manifest(
    raw_solutions_path: str,
    time_limits: dict[int, list[int]],
) -> list[tuple[Path, int]]:
    """Flattened list mapping job_id -> (file_path, time_limit).

    For each solution file (sorted order), every time limit configured
    for its problem size becomes one manifest entry. Files whose problem
    size isn't in `time_limits` are skipped.
    """
    manifest: list[tuple[Path, int]] = []
    for path in list_solution_files(raw_solutions_path):
        schedule = Schedule.read_from_file(str(path))
        jobs_num = schedule.get_problem().n_jobs
        for t_l in time_limits.get(jobs_num, []):
            manifest.append((path, t_l))
    return manifest


def save_manifest(manifest: list[tuple[Path, int]], manifest_path: Path) -> None:
    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump([[str(p), t] for p, t in manifest], f)
    tmp_path.replace(manifest_path)  # atomic rename on POSIX


def load_manifest(manifest_path: Path) -> list[tuple[Path, int]]:
    with open(manifest_path) as f:
        raw = json.load(f)
    return [(Path(p), t) for p, t in raw]


def get_or_build_manifest(
    raw_solutions_path: str,
    time_limits: dict[int, list[int]],
    manifest_path: Path,
) -> list[tuple[Path, int]]:
    if manifest_path.is_file():
        return load_manifest(manifest_path)
    manifest = build_job_manifest(raw_solutions_path, time_limits)
    save_manifest(manifest, manifest_path)
    return manifest


def cluster_post_processing(
    input_path: Path,
    time_limit: int,
    processed_solutions_dir: Path,
    model: str = "multi_mode",
    execfile: str | None = None,
) -> None:
    try:
        schedule = Schedule.read_from_file(str(input_path))
        jobs_num = schedule.get_problem().n_jobs

        print(
            f"Problem size: {jobs_num} jobs; "
            f"time limit: {time_limit} seconds; model: {model}"
        )

        start_time = time.monotonic()

        if model == "multi_mode":
            processed_schedule, gap = cp_multi_mode_buf_postprocessing(
                start_sol=schedule,
                num_of_buf_modes=2,
                search_window=500,
                scenarios_num=30,
                obj="avg_exp_ovp",
                time_limit=time_limit,
                log_output=True,
                execfile=execfile,
            )
            suffix = "_pproc_cp_multi_mode"

        elif model == "makespan_bound":
            processed_schedule, gap = cp_makespan_bound_postprocessing(
                start_sol=schedule,
                search_window=500,
                scenarios_num=30,
                obj="avg_exp_ovp",
                time_limit=time_limit,
                log_output=True,
                execfile=execfile,
            )
            suffix = "_pproc_cp_makespan_bound"

        else:
            raise ValueError(
                f"Unknown model {model!r}. Use 'multi_mode' or "
                "'makespan_bound'."
            )

        solve_time = time.monotonic() - start_time

        processed_solutions_dir.mkdir(parents=True, exist_ok=True)

        if "_no_post_pr" in input_path.stem:
            output_stem = input_path.stem.replace("_no_post_pr", suffix)
        else:
            output_stem = f"{input_path.stem}{suffix}"

        # Encode the chosen time limit in the filename so multiple runs
        # with different limits don't overwrite each other.
        output_stem = f"{output_stem}_tl{time_limit}"

        output_path = processed_solutions_dir / (
            output_stem + input_path.suffix
        )
        processed_schedule.save_to_file(str(output_path))

        # Sidecar file holding metadata not stored in the schedule itself.
        meta_path = processed_solutions_dir / (output_stem + ".meta.json")
        with open(meta_path, "w") as mf:
            json.dump(
                {"gap": gap, "time": solve_time, "time_limit": time_limit},
                mf,
            )

        gap_text = "unknown" if gap is None else str(gap)
        print(
            f"OK: file {input_path.name}; gap={gap_text}; "
            f"time={solve_time:.2f}s; saved to {output_path}"
        )

    except (Exception, KeyboardInterrupt) as error:
        logging.exception(
            "Post-processing failed for file %s, time limit %i",
            input_path.name, time_limit
        )
        print(
            f"Error {error}, file {input_path.name}, time_limit {time_limit}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cluster gives job_id")
    parser.add_argument("job_id", type=int, nargs="?", help="job index")
    parser.add_argument(
        "--build-manifest", action="store_true",
        help="Build (or rebuild) the job manifest and exit.",
    )
    args = parser.parse_args()

    processed_solutions_dir = Path(
        "../data/occidata/sol_best_gnn_model_pproc"
    )
    raw_solutions_path = "../data/occidata/sol_best_gnn_model"
    time_limits = {
        62: [2 * 60, 20 * 60],
        122: [3 * 60, 30 * 60],
        202: [3 * 60, 30 * 60],
        402: [6 * 60, 60 * 60],
    }

    processed_solutions_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_solutions_dir / "job_manifest.json"

    if args.build_manifest:
        manifest = build_job_manifest(raw_solutions_path, time_limits)
        save_manifest(manifest, manifest_path)
        print(f"Manifest built with {len(manifest)} entries: {manifest_path}")
        sys.exit(0)

    if args.job_id is None:
        parser.error("job_id is required unless --build-manifest is given")

    manifest = load_manifest(manifest_path)
    if args.job_id < 0 or args.job_id >= len(manifest):
        print(
            f"Invalid job_id {args.job_id}. Available range: "
            f"0..{len(manifest) - 1}"
        )
    else:
        input_path, time_limit = manifest[args.job_id]
        cluster_post_processing(
            input_path, time_limit, processed_solutions_dir, model="multi_mode"
        )