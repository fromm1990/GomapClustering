import math

import numpy as np
from numpy import ndarray
from sklearn.cluster import DBSCAN

from GoMapClustering.base import GoMapClusterMixin


class AngleBalancingDBSCAN(GoMapClusterMixin):
    def __init__(self, max_distance=10, max_angle=0.34907, min_samples=5, degrees=False) -> None:
        self.max_distance = max_distance
        self.degrees = degrees
        self.max_angle = math.radians(max_angle) if self.degrees else max_angle
        self.min_samples = min_samples

        self.__dbscan = DBSCAN(
            eps=self.max_distance,
            min_samples=self.min_samples
        )


    def _compute_scale_factor(self) -> float:
        # Using The Law of Cosines c^2=a^2+b^2-2ab*cos(C)
        # However we are on the unit circle therefore 
        # c^2=1^2+1^2-2*1*1*cos(C)
        # => c^2=2-2*cos(C)
        # => c=sqrt(2-2*cos(C))

        return self.max_distance / np.math.sqrt(2 - 2 * np.math.cos(self.max_angle))


    @staticmethod
    def _expand(x: float, y: float, angle: float) -> ndarray:
        return np.array([x, y, math.cos(angle), math.sin(angle)])


    def _balance(self, x, y, cos, sin) -> ndarray:
        scale_factor = self._compute_scale_factor()
        return np.array([x, y, cos * scale_factor, sin * scale_factor])


    def fit_predict(self, X: ndarray, y=None):
        if self.degrees:
            X = np.radians(X)

        X = (self._expand(x[0], x[1], x[2]) for x in X)
        X = (self._balance(x[0], x[1], x[2], x[3]) for x in X)
        self.__dbscan.fit(list(X))
        return self.__dbscan.labels_
