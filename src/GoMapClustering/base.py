import numpy as np
from numpy import ndarray
from pandas import DataFrame
from shapely.geometry.point import Point
from sklearn.base import ClusterMixin


class GoMapClusterMixin(ClusterMixin):

    def _point_to_array(self, geom: Point) -> ndarray:
        return np.array([geom.x, geom.y])


    def cluster(self, df: DataFrame):
        generator = map(self._point_to_array, df['geom'])
        generator = map(lambda coords, heading: np.array([coords[0], coords[1], heading]), generator, df['heading'])

        return self.fit_predict(np.array(list(generator)))
