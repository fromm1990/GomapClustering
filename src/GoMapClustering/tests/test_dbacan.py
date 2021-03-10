import unittest
from unittest import TestCase

import numpy as np
from GoMapClustering.AngularClustering import DBACAN
from numpy.testing._private.utils import assert_equal


class TestDBACAN(TestCase):

    def setUp(self) -> None:
        self.testdata = [
            (
                [0, 0, 180, 180], 
                [
                    [0, 0, 180, 180],
                    [0, 0, 180, 180],
                    [180, 180, 0, 0],
                    [180, 180, 0, 0]
                ]
            ),
            (
                [0, 45, 90, 180], 
                [
                    [0, 45, 90, 180],
                    [45, 0, 45, 135],
                    [90, 45, 0, 90],
                    [180, 135, 90, 0]
                ]
            )
        ]
    
    def test_get_distance_matrix(self):
        for test in self.testdata:
            data = np.radians(test[0]).reshape(-1, 1)
            dist_matrix = DBACAN._get_distance_matrix(data)
            diff = dist_matrix - np.radians(test[1])
            assert_equal(diff.sum(), 0)


if __name__ == '__main__':
    unittest.main()
