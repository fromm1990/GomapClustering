import unittest
from unittest import TestCase

import numpy as np
from GoMapClustering.AngularClustering import FCM4DDHard
from numpy.testing._private.utils import assert_equal


class TestFCM4DDHard(TestCase):

    def setUp(self) -> None:
        self.testdata = [
            ([0, 0, 180, 180], 2, 2, 1337),
            ([0, 0, 180, 180], 3, 2, 1337),
            ([1, 2, 180, 180], 2, 2, 1337),
            ([1, 2, 180, 180], 3, 3, 1337),
            ([1, 2, 180, 181], 3, 3, 1337)
        ]
    
    def test_cluster_count(self):
        for test in self.testdata:
            data = np.radians(test[0]).reshape(-1, 1)
            fcm4dd = FCM4DDHard(c=test[1], seed=test[3])
            result = set(fcm4dd.fit_predict(data))

            assert_equal(len(result), test[2])


if __name__ == '__main__':
    unittest.main()
