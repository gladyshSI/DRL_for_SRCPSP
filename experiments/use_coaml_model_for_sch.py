import csv
import time

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from tqdm import tqdm

from dataset_generators.occidata_to_schedule import (
    read_jobs,
    read_graph_from_txt,
    make_schedule_from_str,
    make_df_from_all_csv_files,
    filter_dataframe,
    occidata_graph_file_path_to_ours,
    occidata_jobs_file_path_to_ours,
)
from engines.link_value_train_coaml import assignment_oracle
from gnn_models.link_value_prediction import SimpleLinkValuePredictor
from lib.problem import Problem, get_longest_paths
from lib.schedule import Schedule


CHECKPOINT_PATH = "../checkpoints/SimpleLinkValuePredictor_Coaml.pth"
EXPERIMENT_JOURNAL = "../data/experiments/journal.csv"


def get_j_id_node_id_relationship(part_sch: Schedule):
    """Build the same cropped-graph node mapping used by the GNN."""
    problem = part_sch.get_problem()
    scheduled = part_sch.get_scheduled()

    last_executed = set()
    for seq in part_sch.get_execution_orders():
        if len(seq) > 0:
            last_executed.add(seq[-1])

    j_id_to_gnn_node_id = {}
    gnn_node_id_to_j_id = {}

    n_not_sch_jobs = problem.n_jobs - len(scheduled)
    free_id = 0

    for j_id in range(problem.n_jobs):
        if j_id in last_executed:
            # Machine node IDs are placed after all unscheduled-job nodes.
            w_id = part_sch.get_scheduled_worker(j_id)
            node_id = n_not_sch_jobs + w_id
            j_id_to_gnn_node_id[j_id] = node_id
            gnn_node_id_to_j_id[node_id] = j_id
        elif j_id in scheduled:
            continue
        else:
            j_id_to_gnn_node_id[j_id] = free_id
            gnn_node_id_to_j_id[free_id] = j_id
            free_id += 1

    return j_id_to_gnn_node_id, gnn_node_id_to_j_id


def partial_sch_to_cropped_data_with_com_node(part_sch: Schedule) -> Data:
    """
    Build the cropped graph used by the trained COAML model.

    Nodes:
      - all unscheduled jobs,
      - one machine node per worker (representing the last scheduled job),
      - one communication node.

    Prediction edges:
      - complete machine x current-candidate bipartite graph.

    Important: this representation intentionally follows the existing training
    representation so that inference uses the same features as the checkpoint.
    """
    problem = part_sch.get_problem()

    left_longest_paths = get_longest_paths(problem)
    right_longest_paths = get_longest_paths(problem, reverse=True)
    the_longest_path = max(left_longest_paths.values())

    scheduled = part_sch.get_scheduled()
    candidates = part_sch.get_candidates()
    end_t_distributions = part_sch.calculate_exact_end_time_distributions()

    n_not_sch_jobs = problem.n_jobs - len(scheduled)
    j_id_to_gnn_node_id, gnn_node_id_to_j_id = get_j_id_node_id_relationship(part_sch)

    communication_node_id = n_not_sch_jobs + problem.n_workers

    # Precedence edges among nodes that remain in the cropped graph.
    precedence_fr_ids = []
    precedence_to_ids = []
    for fr_id, to_ids in problem.graph.get_copy_of_all_edges().items():
        for to_id in to_ids:
            if fr_id in j_id_to_gnn_node_id and to_id in j_id_to_gnn_node_id:
                precedence_fr_ids.append(j_id_to_gnn_node_id[fr_id])
                precedence_to_ids.append(j_id_to_gnn_node_id[to_id])

    # Communication-node edges.
    com_workers_fr_ids = [
        n_not_sch_jobs + w_id
        for w_id in range(problem.n_workers)
    ]
    com_workers_to_ids = [communication_node_id] * len(com_workers_fr_ids)

    com_candidates_fr_ids = [j_id_to_gnn_node_id[j_id] for j_id in candidates]
    com_candidates_to_ids = [communication_node_id] * len(com_candidates_fr_ids)

    # Complete machine x candidate prediction graph.
    candidate_gnn_nodes = [j_id_to_gnn_node_id[j_id] for j_id in candidates]
    possible_edges_from_ids = []
    possible_edges_to_ids = []

    for machine_node_id in [
        n_not_sch_jobs + w_id
        for w_id in range(problem.n_workers)
    ]:
        for candidate_node_id in candidate_gnn_nodes:
            possible_edges_from_ids.append(machine_node_id)
            possible_edges_to_ids.append(candidate_node_id)

    # Node features.
    v_attributes = []
    for node_id in range(n_not_sch_jobs + problem.n_workers):
        j_id = gnn_node_id_to_j_id.get(node_id, None)

        last_performed = 1 if node_id >= n_not_sch_jobs else 0
        candidate = 1 if node_id in candidate_gnn_nodes else 0

        min_dur = max_dur = e_dur = 0
        min_ct = max_ct = e_ct = d_ct = 0
        rel_l_longest_path, rel_r_longest_path = 0.0, 1.0

        # Keep the same semantics as the training representation.
        if j_id:
            distribution = problem.jobs[j_id].get_distribution()
            min_dur = distribution.min_v()
            max_dur = distribution.max_v()
            e_dur = distribution.e()

            rel_l_longest_path = left_longest_paths[j_id] / the_longest_path
            rel_r_longest_path = right_longest_paths[j_id] / the_longest_path

        if j_id and last_performed == 1:
            ct_distribution = end_t_distributions[j_id]
            min_ct = ct_distribution.min_v()
            max_ct = ct_distribution.max_v()
            e_ct = ct_distribution.e()
            d_ct = ct_distribution.d()

        v_attributes.append([
            last_performed,
            candidate,
            0,  # communication-node flag
            min_dur,
            max_dur,
            e_dur,
            min_ct,
            max_ct,
            e_ct,
            d_ct,
            rel_l_longest_path,
            rel_r_longest_path,
        ])

    # Communication node.
    v_attributes.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    total_fr_ids = (
        precedence_fr_ids
        + precedence_to_ids
        + com_workers_fr_ids
        + com_workers_to_ids
        + com_candidates_fr_ids
        + com_candidates_to_ids
    )
    total_to_ids = (
        precedence_to_ids
        + precedence_fr_ids
        + com_workers_to_ids
        + com_workers_fr_ids
        + com_candidates_to_ids
        + com_candidates_fr_ids
    )

    prec_edge_attr = [[1, 0, 0, 0, 0, 0] for _ in precedence_fr_ids]
    rev_prec_edge_attr = [[0, 1, 0, 0, 0, 0] for _ in precedence_to_ids]
    com_w_edge_attr = [[0, 0, 1, 0, 0, 0] for _ in com_workers_fr_ids]
    rev_com_w_edge_attr = [[0, 0, 0, 1, 0, 0] for _ in com_workers_to_ids]
    com_c_edge_attr = [[0, 0, 0, 0, 1, 0] for _ in com_candidates_fr_ids]
    rev_com_c_edge_attr = [[0, 0, 0, 0, 0, 1] for _ in com_candidates_to_ids]

    total_edge_attr = (
        prec_edge_attr
        + rev_prec_edge_attr
        + com_w_edge_attr
        + rev_com_w_edge_attr
        + com_c_edge_attr
        + rev_com_c_edge_attr
    )

    return Data(
        x=torch.FloatTensor(v_attributes),
        edge_index=torch.LongTensor([total_fr_ids, total_to_ids]),
        edge_attr=torch.FloatTensor(total_edge_attr),
        edges_eval_index=torch.LongTensor([
            possible_edges_from_ids,
            possible_edges_to_ids,
        ]),
    )


@torch.no_grad()
def get_theta_and_buffers(model, data: Data, device: torch.device):
    """
    Run the GNN once for the current partial schedule.

    Returns raw COAML scores theta. No sigmoid is applied before Hungarian.
    """
    model.eval()
    data = data.to(device)

    z = model(
        data.x,
        data.edge_index,
        edge_attr=getattr(data, "edge_attr", None),
    )

    src = data.edges_eval_index[0]
    dst = data.edges_eval_index[1]

    theta, buffer_pred = model.score_and_value(z[src], z[dst])
    predicted_buffers = (buffer_pred > 0.5).int()

    return src, dst, theta, predicted_buffers


def schedule_edges(
    sch: Schedule,
    edges_to_schedule: list[tuple[int, int, int]],
    machine_node_shift: int,
    use_buffers: bool = True,
):
    """Apply selected machine-job assignments to the partial schedule."""
    for machine_node_id, job_id, buffer_value in edges_to_schedule:
        worker_id = machine_node_id - machine_node_shift

        if not (0 <= worker_id < sch.get_problem().n_workers):
            raise ValueError(
                f"Invalid worker id {worker_id} obtained from machine node "
                f"{machine_node_id} with shift {machine_node_shift}."
            )

        execution_order = sch.get_execution_orders()[worker_id]
        previous_end_time = (
            0
            if len(execution_order) == 0
            else sch.get_scheduled_end_time(execution_order[-1])
        )

        buffer_value = int(buffer_value) if use_buffers else 0
        start_time = max(
            sch.get_first_possible_st_t(worker_id=worker_id, job_id=job_id),
            previous_end_time + buffer_value,
        )

        sch.schedule_job(
            worker_id=worker_id,
            job_id=job_id,
            start_time=start_time,
        )


def choose_coaml_assignment(src, dst, theta):
    """
    Decode raw GNN scores with the same assignment oracle as training.

    The training oracle allows all machines to choose IDLE. During constructive
    inference, however, an all-IDLE decision would leave the state unchanged and
    create an infinite loop. Therefore, if the oracle selects no real edge, force
    exactly the highest-scoring real edge as a progress safeguard.
    """
    y_hat = assignment_oracle(src, dst, theta)

    forced_progress = False
    if int(y_hat.sum().item()) == 0:
        best_edge = int(torch.argmax(theta).item())
        y_hat[best_edge] = 1.0
        forced_progress = True

    return y_hat, forced_progress


def solve_problem_with_coaml_nn(
    model,
    problem: Problem,
    device: torch.device,
    use_buffers: bool = True,
) -> Schedule:
    """Construct a complete baseline schedule using GNN scores + Hungarian."""
    sch = Schedule(problem=problem)

    # Source/dummy job used by the existing scheduling formulation.
    sch.schedule_job(worker_id=0, job_id=0)

    forced_progress_steps = 0

    while len(sch.get_candidates()) > 0:
        _, gnn_node_id_to_j_id = get_j_id_node_id_relationship(sch)
        n_not_sch_jobs = problem.n_jobs - len(sch.get_scheduled())

        data = partial_sch_to_cropped_data_with_com_node(sch)
        src, dst, theta, predicted_buffers = get_theta_and_buffers(
            model,
            data,
            device,
        )

        y_hat, forced_progress = choose_coaml_assignment(src, dst, theta)
        forced_progress_steps += int(forced_progress)

        selected_idx = torch.nonzero(
            y_hat > 0.5,
            as_tuple=False,
        ).flatten()

        edges_to_schedule = []
        for k in selected_idx.tolist():
            machine_node_id = int(src[k].item())
            candidate_node_id = int(dst[k].item())
            job_id = gnn_node_id_to_j_id[candidate_node_id]
            buffer_value = int(predicted_buffers[k].item())

            edges_to_schedule.append(
                (machine_node_id, job_id, buffer_value)
            )

        if not edges_to_schedule:
            raise RuntimeError(
                "COAML decoder produced no real assignments even after the "
                "progress safeguard."
            )

        schedule_edges(
            sch,
            edges_to_schedule,
            machine_node_shift=n_not_sch_jobs,
            use_buffers=use_buffers,
        )

    if forced_progress_steps > 0:
        print(
            f"  progress safeguard used {forced_progress_steps} time(s) "
            f"for problem with {problem.n_jobs} jobs."
        )

    return sch


def calculate_agg_f_ovp_dist_e(agg_f, ovp_dist):
    return agg_f([x.e() for x in ovp_dist.values()])


def schedule_metrics(sch: Schedule):
    dist = sch.calculate_exact_overlap_distributions()
    n_jobs = sch.get_problem().n_jobs

    return {
        "makespan": sch.get_makespan(),
        "avg_delta": calculate_agg_f_ovp_dist_e(sum, dist) / n_jobs,
        "max_delta": calculate_agg_f_ovp_dist_e(max, dist),
    }


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load the trained SimpleLinkValuePredictor COAML checkpoint."""
    print(f"LOAD MODEL: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("model_name") != "SimpleLinkValuePredictor":
        raise ValueError(
            "This evaluation script expects a SimpleLinkValuePredictor "
            f"checkpoint, got {checkpoint.get('model_name')!r}."
        )

    config = checkpoint["config"]
    if config.get("conv_class") != "GATv2Conv":
        raise ValueError(
            f"Unsupported convolution class: {config.get('conv_class')!r}"
        )

    model = SimpleLinkValuePredictor(
        conv_layer=GATv2Conv,
        in_dim=config["in_dim"],
        hid_dim=config["hid_dim"],
        out_dim=config["out_dim"],
        num_layers=config["num_layers"],
        edge_dim=config["edge_dim"],
        heads=config["heads"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("MODEL CONFIG:", config)
    return model


def collect_reference_solutions():
    """Load the CP/reference rows and the unique problem instances to evaluate."""
    sources = [
        ("../data/occidata/outputs_from400", [62, 122]),
        ("../data/occidata/outputs_200_400-80", [202, 402]),
    ]

    dfs = []
    for path, job_nums in sources:
        df = make_df_from_all_csv_files(path)

        for n_jobs in job_nums:
            include = {
                "distribution": ["uniform"],
                "name": ["DET", "STs2", "BBr"],
                "jobs_num": [n_jobs],
            }
            filtered = filter_dataframe(
                df,
                include=include,
                exclude={},
            ).drop_duplicates(
                subset=["name", "jobs_f", "time_limit"]
            )
            dfs.append(filtered)

    if not dfs:
        raise RuntimeError("No reference solution data were loaded.")

    return pd.concat(dfs, ignore_index=True)


def build_problem(row) -> Problem:
    graph_path = occidata_graph_file_path_to_ours(row.graph_f)
    jobs_path = occidata_jobs_file_path_to_ours(row.jobs_f)

    jobs = read_jobs(jobs_path)
    graph = read_graph_from_txt(graph_path)

    return Problem(
        n_workers=int(row.workers_num),
        n_jobs=len(jobs),
        graph=graph,
        jobs=jobs,
    )


def write_reference_rows(writer, df_filtered, problems):
    """Write the existing CP/reference schedules and cache unique problems."""
    for _, row in tqdm(
        df_filtered.iterrows(),
        total=len(df_filtered),
        desc="Reference solutions",
    ):
        problem_key = (row.jobs_f, int(row.workers_num))

        if problem_key not in problems:
            problems[problem_key] = build_problem(row)

        # Validate that the stored reference schedule can still be reconstructed.
        sch = Schedule(problem=problems[problem_key])
        make_schedule_from_str(sch, row.schedule)

        writer.writerow({
            "name": row["name"],
            "jobs_f": row["jobs_f"],
            "jobs_num": row["jobs_num"],
            "workers_num": row["workers_num"],
            "time_limit": row["time_limit"],
            "schedule": row["schedule"],
            "gap": row["gap"],
            "time": row["time"],
            "makespan": row["makespan"],
            "avg_delta": row["avg_delta"],
            "max_delta": row["max_delta"],
        })


def write_coaml_row(writer, jobs_f, problem, sch, solve_time):
    metrics = schedule_metrics(sch)

    writer.writerow({
        "name": "NN_coaml",
        "jobs_f": jobs_f,
        "jobs_num": problem.n_jobs,
        "workers_num": problem.n_workers,
        "time_limit": None,
        "schedule": sch,
        "gap": None,
        "time": solve_time,
        "makespan": metrics["makespan"],
        "avg_delta": metrics["avg_delta"],
        "max_delta": metrics["max_delta"],
    })


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    best_model = load_model_from_checkpoint(
        CHECKPOINT_PATH,
        device=device,
    )

    fieldnames = [
        "name",
        "jobs_f",
        "jobs_num",
        "workers_num",
        "time_limit",
        "schedule",
        "gap",
        "time",
        "makespan",
        "avg_delta",
        "max_delta",
    ]

    print("COLLECT REFERENCE SOLUTIONS")
    df_filtered = collect_reference_solutions()
    print(f"Reference rows: {len(df_filtered)}")

    problems = {}

    # Start a fresh journal and first copy the reference methods.
    with open(EXPERIMENT_JOURNAL, "w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            delimiter=";",
        )
        writer.writeheader()
        write_reference_rows(writer, df_filtered, problems)

    print(f"UNIQUE PROBLEMS TO SOLVE WITH COAML: {len(problems)}")

    # Append one COAML schedule for each unique (jobs file, worker count) problem.
    with open(EXPERIMENT_JOURNAL, "a", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            delimiter=";",
        )

        for (jobs_f, _workers_num), problem in tqdm(
            problems.items(),
            desc="COAML scheduling",
        ):
            t0 = time.time()
            sch = solve_problem_with_coaml_nn(
                best_model,
                problem,
                device=device,
                use_buffers=True,
            )
            solve_time = time.time() - t0

            write_coaml_row(
                writer,
                jobs_f,
                problem,
                sch,
                solve_time,
            )

    print(f"DONE. Results written to {EXPERIMENT_JOURNAL}")


if __name__ == "__main__":
    main()
