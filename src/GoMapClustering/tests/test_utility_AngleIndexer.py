import math
import unittest
from unittest import TestCase

from GoMapClustering.utility import AngleIndexer


class TestAngleIndexer(TestCase):

    def setUp(self) -> None:
        self.testdata = [
            (36, 355, 0),
            (36, 0, 0),
            (36, 4, 0),
            (36, 5, 1),
            (36, 10, 1),
            (36, 14, 1),
            (36, 15, 2)
        ]
    
    def test_index(self):
        for size, angle, expected in self.testdata:
            ai = AngleIndexer(size)
            actual = ai.index(math.radians(angle))
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
