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

    @classmethod
    def from_dict(cls, d: dict):
        return cls(n_workers=d.get('n_workers'),
                   n_jobs=d.get('n_jobs'),
                   graph=PrecedenceGraph.from_dict(d.get('graph')),
                   jobs=[Job.from_dict(d_job) for d_job in d.get('jobs')])

    def save_to_file(self, path_to_file: str):
        with open(path_to_file, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def read_from_file(cls, path_to_file: str):
        with open(path_to_file, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def __repr__(self):
        return str(self.to_dict())



