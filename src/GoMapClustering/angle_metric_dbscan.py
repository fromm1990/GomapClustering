import numpy as np
from pandas import DataFrame
from sklearn.cluster import DBSCAN

from GoMapClustering.AngularClustering import DBACAN
from GoMapClustering.base import GoMapClusterMixin


class AngleMetricDBSCAN(GoMapClusterMixin):
    def __init__(
        self,
        max_distance: float,
        max_angle: float,
        min_samples: int
    ) -> None:
        self.__dbscan = DBSCAN(eps=max_distance, min_samples=min_samples)
        self.__dbacan = DBACAN(eps=max_angle, min_samples=min_samples)


    def fit_predict(self, X, y=None):
        '''
        # Parameters
            - X (List[Tuple[float, float, float]]): Tuples of the format (x, y, angle)
            - y (None): Not used, only for compatability reasons
        '''
        X = DataFrame(X, columns=['x', 'y', 'angle'])
        X['spatial_cid'] = self.__dbscan.fit_predict(X[['x', 'y']])
        X['angle_cid'] = -1

        for cid, group in X.groupby('spatial_cid'):
            if cid == -1:
                continue
            
            angles = group['angle'].to_numpy().reshape(-1, 1)
            angle_cids = self.__dbacan.fit_predict(angles)
            X.loc[X['spatial_cid'] == cid, 'angle_cid'] = angle_cids

        rv = np.full((len(X),), -1)
        cid = 0

        for group, data in X.groupby(['spatial_cid', 'angle_cid']):
            if group[0] == -1 or group[1] == -1:
                continue    
            rv[data.index] = cid
            cid += 1

        return rv
