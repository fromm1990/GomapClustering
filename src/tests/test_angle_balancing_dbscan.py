import unittest
from unittest import TestCase

import numpy as np
from GoMapClustering import AngleBalancingDBSCAN
from numpy.testing._private.utils import assert_equal


class TestAngleBalancingDBSCAN(TestCase):

    def setUp(self) -> None:
        self.test = [
            (10, 10, [[0, 0, 0], [0, 0, 10]], 10),
            (10, 10, [[0, 0, 0], [0, 0, 5]], 5),
            (10, 20, [[0, 0, 0], [0, 0, 20]], 10),
            (10, 20, [[0, 0, 0], [0, 0, 10]], 5),
            (20, 10, [[0, 0, 0], [0, 0, 10]], 20),
            (20, 10, [[0, 0, 0], [0, 0, 5]], 10),
            (20, 20, [[0, 0, 0], [0, 0, 20]], 20),
            (20, 20, [[0, 0, 0], [0, 0, 10]], 10)
        ]


    def test_balance(self):
        for max_distance, max_angle, X, expected in self.test:
            ab_dbscan  = AngleBalancingDBSCAN(max_distance=max_distance, max_angle=max_angle, degrees=True)

            X = np.array(X, dtype=float)
            X[:, -1] = np.radians(X[:, -1])

            X = (ab_dbscan._expand(x[0], x[1], x[2]) for x in X)
            X = (ab_dbscan._balance(x[0], x[1], x[2], x[3]) for x in X)
            X = list(X)

            actual = np.linalg.norm(X[0] - X[1])
            assert(abs(actual - expected) < 0.1), actual


if __name__ == '__main__':
    unittest.main()
