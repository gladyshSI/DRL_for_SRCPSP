from collections import defaultdict
from unittest import TestCase

from lib.graph import PrecedenceGraph


def make_g() -> PrecedenceGraph:
    g = PrecedenceGraph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    return g


class TestPrecedenceGraph(TestCase):
    def setUp(self) -> None:
        self.g = make_g()

    def test_clear(self):
        self.g.clear()
        self.assertEqual({}, self.g._edges)
        self.assertEqual({}, self.g._reverse_edges)

    def test_check_edge(self):
        self.assertTrue(self.g.check_edge(0, 1))
        self.assertTrue(self.g.check_edge(0, 2))
        self.assertFalse(self.g.check_edge(0, 3))
        self.assertFalse(self.g.check_edge(-1, 10))

    def test_get_all_ids(self):
        self.assertEqual({0, 1, 2, 3, 4}, self.g.get_all_ids())

    def test_get_start_ids(self):
        self.assertEqual({0}, self.g.get_start_ids())

    def test_get_end_ids(self):
        self.assertEqual({4}, self.g.get_end_ids())

    def test_get_copy_of_all_edges(self):
        self.assertEqual(defaultdict(set, {0: {1, 2},
                                           1: {3},
                                           2: {3},
                                           3: {4}}),
                         self.g.get_copy_of_all_edges())

    def test_add_edge(self):
        self.g.add_edge(0, 4)
        self.assertEqual(defaultdict(set, {0: {1, 2, 4},
                                           1: {3},
                                           2: {3},
                                           3: {4}}),
                         self.g.get_copy_of_all_edges())

    def test_remove_edge(self):
        self.g.remove_edge(0, 1)
        self.g.remove_edge(3, 4)
        self.assertEqual(defaultdict(set, {0: {2},
                                           1: {3},
                                           2: {3}}),
                         self.g.get_copy_of_all_edges())

    def test_bfs(self):
        bfs_from_1 = [1, 3, 4]
        reverse_bfs_from_1 = [1, 0]
        bfs_from_not_existing_10 = [10]
        self.assertEqual(bfs_from_1, self.g.bfs(1))
        self.assertEqual(reverse_bfs_from_1, self.g.bfs(1, reverse=True))
        self.assertEqual(bfs_from_not_existing_10, self.g.bfs(10))

    def test_topological_sort(self):
        g = PrecedenceGraph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 4)
        g.add_edge(2, 4)
        g.add_edge(3, 5)
        g.add_edge(4, 5)
        order = g.topological_sort()
        reversed_order = g.topological_sort(reverse=True)
        for fr_id, to_ids in g.get_copy_of_all_edges().items():
            for to_id in to_ids:
                self.assertTrue(order.index(fr_id) < order.index(to_id))
                self.assertTrue(reversed_order.index(fr_id) > reversed_order.index(to_id))


    def test_get_successors(self):
        self.assertEqual({1, 2}, self.g.get_successors(0))
        self.assertEqual({3}, self.g.get_successors(1))
        self.assertEqual(set(), self.g.get_successors(4))

    def test_get_predecessors(self):
        self.assertEqual(set(), self.g.get_predecessors(0))
        self.assertEqual({1, 2}, self.g.get_predecessors(3))

    def test_get_all_successors(self):
        self.assertEqual({1, 2, 3, 4}, self.g.get_all_successors(0))

    def test_get_all_predecessors(self):
        self.assertEqual({0, 1, 2, 3}, self.g.get_all_predecessors(4))
