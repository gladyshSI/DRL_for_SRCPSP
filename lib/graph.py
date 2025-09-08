import typing as tt
import copy
import numpy as np
from collections import deque, defaultdict
import random


class PrecedenceGraph:
    def __init__(self):
        # node_id -> {successor_ids}
        self._edges: tt.Dict[int, tt.Set[int]] = dict()
        # node_id -> {predecessor_ids}
        self._reverse_edges: tt.Dict[int, tt.Set[int]] = dict()

    def clear(self):
        self._edges.clear()
        self._reverse_edges.clear()

    def check_edge(self, fr_id: int, to_id: int) -> bool:
        return False if fr_id not in self._edges.keys() or to_id not in self._edges[fr_id] else True

    def get_all_ids(self) -> tt.Set[int]:
        left_vertices = set(self._edges.keys())
        right_vertices = set(self._reverse_edges.keys())
        all_vertices = left_vertices | right_vertices
        return all_vertices

    def get_start_ids(self) -> tt.Set[int]:
        left_vertices = set(self._edges.keys())
        right_vertices = set(self._reverse_edges.keys())
        start_vertices = left_vertices - right_vertices
        return start_vertices

    def get_end_ids(self) -> tt.Set[int]:
        left_vertices = set(self._edges.keys())
        right_vertices = set(self._reverse_edges.keys())
        end_vertices = right_vertices - left_vertices
        return end_vertices

    def get_copy_of_all_edges(self) -> tt.Dict[int, tt.Set[int]]:
        return copy.deepcopy(self._edges)

    def add_edge(self, fr_id: int, to_id: int) -> None:
        if fr_id not in self._edges.keys():
            self._edges[fr_id] = set()
        self._edges[fr_id].add(to_id)
        if to_id not in self._reverse_edges.keys():
            self._reverse_edges[to_id] = set()
        self._reverse_edges[to_id].add(fr_id)

    def remove_edge(self, fr_id: int, to_id: int) -> None:
        if self.check_edge(fr_id, to_id):
            self._edges[fr_id].remove(to_id)  # Remove direct edge
            if not self._edges[fr_id]:
                del self._edges[fr_id]
            self._reverse_edges[to_id].remove(fr_id)  # Remove reverse edge
            if not self._reverse_edges[to_id]:
                del self._reverse_edges[to_id]

    # TODO: Think how to make all graphs not isomorphic to each other
    def random_network(self, number_of_nodes: int,
                       start_n_node_range: tt.Tuple[int, int] = (3, 5),
                       end_n_node_range: tt.Tuple[int, int] = (3, 5),
                       seed: int = 1):
        random.seed = seed
        start_num = np.round(np.random.uniform(start_n_node_range[0], start_n_node_range[1])).astype(int)
        end_num = min(number_of_nodes - 2 - start_num,
                      np.round(np.random.uniform(end_n_node_range[0], end_n_node_range[1])).astype(int))

        # Step 1: Connect two dummy tasks with start and end tasks
        dummy_st_id = 0
        self._edges[dummy_st_id] = set()
        for i in range(start_num):
            to_id = i + 1
            self._edges[dummy_st_id].add(to_id)
            if to_id not in self._reverse_edges.keys():
                self._reverse_edges[to_id] = set()
            self._reverse_edges[to_id].add(dummy_st_id)

        dummy_end_id = number_of_nodes - 1
        self._reverse_edges[dummy_end_id] = set()
        for i in range(end_num):
            fr_id = number_of_nodes - 2 - i
            if fr_id not in self._edges.keys():
                self._edges[fr_id] = set()
            self._edges[fr_id].add(dummy_end_id)
            self._reverse_edges[dummy_end_id].add(fr_id)
        # print("Step 1: edges: ", self._edges)
        # print("Step 1: rev:   ", self._reverse_edges)

        # Step 2: Find random predecessor:
        predecessors = list(range(1, start_num + 1))
        for to_id in list(range(start_num + 1, dummy_end_id)):
            fr_id = random.choice(predecessors)
            self.add_edge(fr_id, to_id)
            if to_id < dummy_end_id - end_num:
                predecessors.append(to_id)
        # print("Step 2: edges: ", self._edges)
        # print("Step 2: rev:   ", self._reverse_edges)

        # Step 3: Find random successor & delete redundant edges:
        no_out_ids = [i for i in range(dummy_end_id)
                      if i not in self._edges.keys()]
        for fr_id in no_out_ids:
            to_id_list = list(range(max([start_num + 1, fr_id + 1]), dummy_end_id))
            to_id = random.choice(to_id_list)
            self.add_edge(fr_id, to_id)

            # Find & delete redundant edges:
            all_predecessors = self.get_all_predecessors(fr_id)
            all_predecessors.add(fr_id)
            all_successors = self.get_all_successors(to_id)
            all_successors.add(to_id)
            edges_to_remove = []
            for i in all_predecessors:
                for j in {} if i not in self._edges.keys() else self._edges[i]:
                    if j in all_successors and (i, j) != (fr_id, to_id):
                        edges_to_remove.append((i, j))
            for i, j in edges_to_remove:
                self.remove_edge(i, j)
        # print("Step 3: edges: ", self._edges)
        # print("Step 3: rev:   ", self._reverse_edges)

    def bfs(self, start_id: int, reverse: bool = False) -> tt.List[int]:
        edges = self._reverse_edges if reverse else self._edges
        order = []

        # Check is there such vertex:
        if start_id not in edges.keys():
            # There is no such start id
            return [start_id]

        successors = set()
        q = deque([start_id])
        while q:
            v = q.popleft()
            for next_v in [] if v not in edges.keys() else edges[v]:
                if next_v not in successors:
                    successors.add(next_v)
                    q.append(next_v)
            order.append(v)
        return order

    def topological_sort(self, reverse: bool = False) -> tt.List[int]:
        start_nodes = self.get_start_ids() if not reverse else self.get_end_ids()
        edges = self._edges if not reverse else self._reverse_edges
        reverse_edges = self._reverse_edges if not reverse else self._edges

        order = list(start_nodes)
        scheduled = start_nodes
        candidates = set()
        for i in order:
            candidates = candidates.union(edges[i])
        while candidates:
            for c in candidates:
                if reverse_edges[c] <= scheduled:
                    order.append(c)
                    scheduled.add(c)
                    candidates.remove(c)
                    new_candidates = {} if c not in edges.keys() else edges[c]
                    candidates = candidates.union(new_candidates)
                    break

        return order


    def get_successors(self, fr_id: int) -> tt.Set[int]:
        return self._edges[fr_id].copy() if fr_id in self._edges.keys() else set()

    def get_predecessors(self, fr_id: int) -> tt.Set[int]:
        return self._reverse_edges[fr_id].copy() if fr_id in self._reverse_edges.keys() else set()

    def get_all_successors(self, v_id: int) -> tt.Set[int]:
        return set(self.bfs(v_id, False)[1:]).copy()

    def get_all_predecessors(self, v_id) -> tt.Set[int]:
        return set(self.bfs(v_id, True)[1:]).copy()

    def to_dict(self) -> dict:
        return {'edges': [(fr_id, to_id) for fr_id, to_ids in self._edges.items() for to_id in to_ids]}

    @classmethod
    def from_dict(cls, d: dict):
        edges = d.get('edges')
        g = cls()
        for fr_id, to_ids in edges:
            g.add_edge(fr_id, to_ids)
        return g

    def print_itself(self):
        print(f'num nodes: {len(self.get_all_ids())}')
        print(f'num edges: {sum((len(to_ids) for to_ids in self._edges.values()))}')
        print(f'start ids: {self.get_start_ids()}')
        print(f'end ids: {self.get_end_ids()}')
        print(f'edges: {self._edges}')

    def __repr__(self):
        to_print = (f'num nodes: {len(self.get_all_ids())}\n' +
                    f'num edges: {sum((len(to_ids) for to_ids in self._edges.values()))}\n' +
                    f'start ids: {self.get_start_ids()}\n' +
                    f'edges: {self._edges}')
        return to_print

