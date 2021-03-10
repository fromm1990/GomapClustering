import sklearn.cluster
from pandas import DataFrame
import numpy as np

from GoMapClustering.base import GoMapClusterMixin


class DBSCAN(GoMapClusterMixin):

    def __init__(self, eps: float, min_samples: int) -> None:
        super().__init__()
        self.dbscan = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)


    def cluster(self, df: DataFrame):
        generator = map(self._point_to_array, df['geom'])
        return self.fit_predict(np.array(list(generator)))


    def fit_predict(self, X, y=None):
        return self.dbscan.fit_predict(X)
