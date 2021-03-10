import numpy as np
from pandas import DataFrame
from sklearn.cluster import DBSCAN

from GoMapClustering.AngularClustering import FCM4DDHard
from GoMapClustering.base import GoMapClusterMixin


class DBSCANFCM4DD(GoMapClusterMixin):

    def __init__(
        self,
        c: int,
        max_spatial_distance=10,
        min_improvement=1e-9,
        min_samples=5,
        m=2,
        max_iterations=100,
        seed=None
    ) -> None:
        '''
        # Parameters
        - c (int): Expected amount of clusters
        - max_spatial_distance (float): The threshold to asses when data points are spatially disconnected
        - min_improvement (float): If the iterative improvement is less than `min_improvement` the result is returned
        - m (float): Fuzziness parameter
        - max_iterations (int): The maximum amount of allowed iterations to improve the result
        - seed (int): The seed used to initialize the algorithm
        '''

        self.__dbscan = DBSCAN(
            eps=max_spatial_distance,
            min_samples=min_samples
        )
        self.__fcm4dd = FCM4DDHard(
            c,
            min_improvement,
            m,
            max_iterations,
            seed
        )


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
            angle_cids = self.__fcm4dd.fit_predict(angles)
            X.loc[X['spatial_cid'] == cid, 'angle_cid'] = angle_cids
        
        rv = np.full((len(X),), -1)
        cid = 0

        for group, data in X.groupby(['spatial_cid', 'angle_cid']):
            if group[0] == -1 or group[1] == -1:
                continue

            rv[data.index] = cid
            cid += 1

        return rv
