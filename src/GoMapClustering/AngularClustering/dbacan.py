import math

import numpy as np
from GoMapClustering.utility import PI, TWO_PI
from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.base import ClusterMixin
from sklearn.cluster import DBSCAN


class DBACAN(ClusterMixin):
    def __init__(self, eps: float, min_samples: int, degrees=False) -> None:
        '''
        # Parameters
        - eps (float): The maximum angle between two samples for one to be considered as in the neighborhood of the other.
        - min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        - degrees (bool): True if eps is in degrees, false if in radians.
        This includes the point itself.
        '''
        self.degrees = degrees
        if self.degrees:
            eps = math.radians(eps)

        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    
    @staticmethod
    def _get_distance_matrix(X: ndarray) -> ndarray:
        return abs((((X - X.transpose()) + PI) % TWO_PI) - PI)


    def fit_predict(self, X, y=None):
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, (DataFrame, Series)):
            X = X.to_numpy()
        if self.degrees:
            X = np.radians(X)

        X = np.reshape(X, (len(X), 1))
        dist_matrix = self._get_distance_matrix(X)

        return self.dbscan.fit_predict(dist_matrix)


# angles = [0, 0, 0, 90, 90, 90, 180, 180, 180, 270, 270, 270]
# result = DBACAN(20, 3, True).fit_predict(angles)
# print(result)
