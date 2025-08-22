import dataclasses
import typing as tt

from lib.graph import PrecedenceGraph
from lib.job import Job


@dataclasses.dataclass
class Problem:
    n_workers: int
    n_jobs: int
    graph: PrecedenceGraph
    jobs: tt.List[Job]


