from unittest import TestCase

import numpy as np

from lib.distribution import DiscreteDistribution


class TestDiscreteDistribution(TestCase):
    def setUp(self) -> None:
        v1 = np.array([-1, 0, 1])
        p1 = np.array([0.1, 0.2, 0.7])
        v2 = np.array([0, 1, 2])
        p2 = np.array([0.5, 0.3, 0.2])
        self.d1 = DiscreteDistribution(v1, p1)
        self.d2 = DiscreteDistribution(v2, p2)

    def test_init(self):
        v1 = np.array([-1, 0, 0])
        v2 = np.array([0, 1])
        p1 = np.array([0.1, 0.2, 0.7])
        self.assertRaises(ValueError, DiscreteDistribution, v1, p1)  # All values should be different
        self.assertRaises(ValueError, DiscreteDistribution, v2, p1)  # values and probs should have the same size

    def test_add(self):
        d3 = self.d1 + self.d2
        print(d3.values)
        print(d3.probs)
        self.assertTrue(np.array_equal(np.array([-1, 0, 1, 2, 3]), d3.values))
        self.assertTrue(np.array_equal(np.array([0.05, 0.13, 0.43, 0.25, 0.14]), d3.probs.round(4)))

    def test_max_with_distribution(self):
        d3 = self.d1.max_with(self.d2)
        d4 = self.d2.max_with(self.d1)
        self.assertTrue(np.array_equal(np.array([0, 1, 2]), d3.values))
        self.assertTrue(np.array_equal(np.array([0, 1, 2]), d4.values))
        self.assertTrue(np.array_equal(np.array([0.15, 0.65, 0.2]), d3.probs.round(4)))
        self.assertTrue(np.array_equal(np.array([0.15, 0.65, 0.2]), d4.probs.round(4)))

    def test_max_with_constant(self):
        d3 = self.d1.max_with(1)
        d4 = self.d2.max_with(1)
        self.assertTrue(np.array_equal(np.array([1]), d3.values))
        self.assertTrue(np.array_equal(np.array([1, 2]), d4.values))
        self.assertTrue(np.array_equal(np.array([1.]), d3.probs.round(4)))
        self.assertTrue(np.array_equal(np.array([0.8, 0.2]), d4.probs.round(4)))

        d5 = self.d1.max_with(-1)
        self.assertTrue(np.array_equal(np.array([-1, 0, 1]), d5.values))
        self.assertTrue(np.array_equal(np.array([0.1, 0.2, 0.7]), d5.probs.round(4)))

    def test_normalize(self):
        v = np.array([-1, 0, 1])
        p = np.array([2, 49, 49])
        d = DiscreteDistribution(v, p)
        d.normalize(epsilon=0.1)
        self.assertTrue(np.array_equal(np.array([0, 1]), d.values))
        self.assertTrue(np.array_equal(np.array([0.5, 0.5]), d.probs.round(4)))

    def test_min_max_v(self):
        self.assertEqual(-1, self.d1.min_v())
        self.assertEqual(1, self.d1.max_v())
        self.assertEqual(0, self.d2.min_v())
        self.assertEqual(2, self.d2.max_v())

    def test_getitem(self):
        self.assertEqual(0.2, self.d1[0])
        self.assertEqual(0.2, self.d2[2])
        self.assertEqual(0., self.d1[2])

    def test_e(self):
        self.assertEqual(0.6, self.d1.e())
        self.assertEqual(0.7, self.d2.e())
