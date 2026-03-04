import pandas as pd
import glob
import os
import ast

from tqdm import tqdm

from lib.distribution import Distribution, DiscreteDistribution
from lib.graph import PrecedenceGraph
from lib.job import Job
from lib.problem import Problem
from lib.schedule import Schedule


def make_df_from_all_csv_files(directory: str) -> pd.DataFrame:
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    print(len(all_files))
    return pd.concat((pd.read_csv(f, sep=';') for f in all_files), ignore_index=True)


def filter_dataframe(df: pd.DataFrame, include: dict, exclude: dict) -> pd.DataFrame:
    mask = True
    for col, vals in include.items():
        temp_mask = False
        for val in vals:
            temp_mask |= (df[col] == val)
        mask &= temp_mask
    for col, vals in exclude.items():
        temp_mask = False
        for val in vals:
            temp_mask &= (df[col] != val)
            mask &= temp_mask
    return df[mask].copy().reset_index()


def read_graph_from_txt(path_to_file: str) -> PrecedenceGraph:
    graph = PrecedenceGraph()
    with open(path_to_file, 'r') as f:
        for line in f:
            fr_id, to_ids = line.split(':')
            if to_ids[-2:] == ',\n':
                to_ids = to_ids[:-2]
            for to_id in to_ids.split(','):
                graph.add_edge(int(fr_id), int(to_id))
    return graph


def string_to_job(string: str) -> Job:
    str_id, str_duration, str_distribution = string.split(':', 2)
    dist = DiscreteDistribution.from_dict(ast.literal_eval(str_distribution))
    job = Job(int(str_id), int(str_duration), dist)
    return job


def read_jobs(path_to_file: str) -> list[Job]:
    jobs = []
    with open(path_to_file, 'r') as f:
        for line in f:
            job = string_to_job(line)
            jobs.append(job)
    return jobs


def occidata_graph_file_path_to_ours(path_to_file: str) -> str:
    if path_to_file.split('/')[-2] == 'parsedPSPLib':
        intermediate_folder = 'parsedPSPLib/'
    elif path_to_file.split('/')[-3] == 'FasterGeneratedGraphs':
        intermediate_folder = 'FasterGeneratedGraphs/'
    else:
        intermediate_folder = 'FasterGeneratedGraphs'
    return '../data/occidata/graphs/' + intermediate_folder + path_to_file.split('/')[-1]


def occidata_jobs_file_path_to_ours(path_to_file: str) -> str:
    return '../data/occidata/tasks/uniform/' + path_to_file.split('/')[-1]


def make_schedule_from_str(sch: Schedule, sch_str: str) -> None:
    data = ast.literal_eval(sch_str)
    for j, w, st in data:
        sch.schedule_job(worker_id=w, job_id=j, start_time=st)


def main():
    to_save_solutions_dir = '../data/occidata/solutions/'
    df = make_df_from_all_csv_files('../data/occidata/outputs')
    include = {'distribution': ['uniform'], 'name': ['BBr'], 'jobs_num': [32, 62, 122]}
    df_filtered = filter_dataframe(df, include=include, exclude={})
    print(df_filtered.shape)
    for graph_f, jobs_f, n_workers, sch_str in tqdm(zip(df_filtered['graph_f'],
                                                        df_filtered['jobs_f'],
                                                        df_filtered['workers_num'],
                                                        df_filtered['schedule'])):
        graph_path = occidata_graph_file_path_to_ours(graph_f)
        jobs_path = occidata_jobs_file_path_to_ours(jobs_f)

        jobs = read_jobs(jobs_path)
        graph = read_graph_from_txt(graph_path)
        problem = Problem(n_workers=n_workers, n_jobs=len(jobs), graph=graph, jobs=jobs)
        sch = Schedule(problem=problem)
        make_schedule_from_str(sch, sch_str)
        new_f_name = to_save_solutions_dir + jobs_f.split('_')[-2] + '_' + jobs_f.split('_')[-1].split('.')[-2] + '.json'
        sch.save_to_file(new_f_name)


if __name__ == '__main__':
    main()
