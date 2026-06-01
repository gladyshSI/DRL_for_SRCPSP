import copy
import dataclasses
import json
import typing as tt

from lib.graph import PrecedenceGraph
from lib.job import Job


@dataclasses.dataclass
class Problem:
    n_workers: int
    n_jobs: int
    graph: PrecedenceGraph
    jobs: tt.List[Job]

    def to_dict(self):
        return {'n_workers': self.n_workers,
                'n_jobs': self.n_jobs,
                'graph': self.graph.to_dict(),
                'jobs': [job.to_dict() for job in self.jobs]}

    def reverse(self):
        return Problem(self.n_workers, self.n_jobs, self.graph.get_reversed_copy(), copy.deepcopy(self.jobs))

    @classmethod
    def from_dict(cls, d: dict):
        return cls(n_workers=d.get('n_workers'),
                   n_jobs=d.get('n_jobs'),
                   graph=PrecedenceGraph.from_dict(d.get('graph')),
                   jobs=[Job.from_dict(d_job) for d_job in d.get('jobs')])

    def save_to_file(self, path_to_file: str, indent=None):
        with open(path_to_file, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def read_from_file(cls, path_to_file: str):
        with open(path_to_file, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def __repr__(self):
        return str(self.to_dict())


def get_longest_paths(problem: Problem, reverse: bool = False) -> dict[int, int]:  # job_id -> the longest path's length
    topological_sorted_js = problem.graph.topological_sort(reverse=reverse)
    longest_paths = dict()
    for j in topological_sorted_js:
        longest_paths[j] = 0
    for j in topological_sorted_js:
        predecessors = problem.graph.get_predecessors(j) if not reverse else problem.graph.get_successors(j)
        new_value = max([longest_paths[j]] + [longest_paths[pred_id] + problem.jobs[pred_id].get_duration() for
                       pred_id in predecessors])
        longest_paths[j] = new_value

    return longest_paths
