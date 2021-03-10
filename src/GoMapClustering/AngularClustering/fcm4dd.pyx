import math
from typing import List

import numpy as np
cimport numpy as np
from numpy cimport ndarray, float32_t
from sklearn.base import ClusterMixin

cdef float PI = math.pi
cdef float NEGATIVE_PI = -1 * math.pi
cdef float TWO_PI = math.pi * 2

# Implemented from:
# https://se.mathworks.com/matlabcentral/fileexchange/59315-fcm4dd-fuzzy-c-means-clustering-algorithm-for-directional-data

# Fix to division by zero not addressed in paper
# https://math.stackexchange.com/questions/3791661/division-by-0-extreme-case-in-fuzzy-c-means-clustering
cdef class FCM4DD():
    cdef readonly int c, max_iterations, seed
    cdef readonly float eps, m

    def __init__(self, int c, float eps=1e-9, float m=2, int max_iterations=100, int seed=-1):
        '''
        # Parameters
            - c (int): Expected amount of clusters
            - eps (float): If the iterative improvement is less than eps the result is returned
            - m (float): Fuzziness parameter
            - max_iterations (int): The maximum amount of allowed iterations to improve the result
            - seed (int): The seed used to initialize the algorithm
        '''

        if c < 2:
            raise Exception('c must be assigend a value >= 2')
        if m <= 0:
            raise Exception('m must be assigned a value > 0')
        if eps <= 0:
            raise Exception('eps must be assigned a value > 0')

        self.c = c
        self.eps = eps
        self.m = m
        self.max_iterations = max_iterations
        if seed == -1:
            self.random = np.random.default_rng()
        else:
            self.random = np.random.default_rng(seed)

        self.seed = self.random.bit_generator._seed_seq.entropy


    cpdef __get_dimensionality(self, ndarray X):
        try:
            return np.shape(X)[1]
        except:
            msg = "\n".join([
                'Expected 2D array, got 1D array instead:',
                X,
                'Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.'
            ])
            raise ValueError(msg)


    cpdef __assign_angle_diffs(
        self,
        ndarray X,
        ndarray[float32_t, ndim=3] F,
        ndarray[double, ndim=2] center
    ):
        cdef int i, j

        for i in range(len(X)):
            for j in range(self.c):
                F[i, j, :] = ((X[i, :] - center[j, :] + PI) % TWO_PI) - PI


    cpdef __assign_1d_cluster_membership(self, ndarray[float32_t, ndim=2] Fi, ndarray[float32_t] ui, float exponent):
        cdef int cluster_count = len(Fi)
        cdef int j, k

        for j in range(cluster_count):
            ui[j] = 0

            for k in range(cluster_count):
                denominator = abs(Fi[k, :])
                # Avoids division by zero
                if denominator == 0:
                    ui[:] = 0
                    ui[k] = 1
                    return
                
                ui[j] += (abs(Fi[j, :]) / denominator) ** exponent
            
            ui[j] = 1 / ui[j]


    cpdef __assign_xd_cluster_membership(self, Fi: List[float], ui: List[float], exponent: float):
        cluster_count = len(Fi)

        for j in range(cluster_count):
            ui[j] = 0
            
            for k in range(cluster_count):
                F2 = np.squeeze(Fi[k, :])
                denominator = np.linalg.norm(F2)
                # Avoids division by zero
                if denominator == 0:
                    ui[:] = 0
                    ui[k] = 1
                    return

                F1 = np.squeeze(Fi[j, :])
                ui[j] += (np.linalg.norm(F1)/denominator) ** exponent
            
            ui[j] = 1 / ui[j]


    cpdef __assign_memberships(
        self,
        ndarray X,
        int ndim,
        ndarray[float32_t, ndim=3] F,
        ndarray[float32_t, ndim=2] u
    ):
        cdef float exponent = 2 / (self.m - 1)
        cdef int i

        for i in range(len(X)):
            if ndim == 1:
                self.__assign_1d_cluster_membership(F[i], u[i], exponent)
            else:
                self.__assign_xd_cluster_membership(F[i], u[i], exponent)


    cpdef __assign_centers(
        self,
        ndarray X,
        int ndim,
        ndarray[float32_t, ndim=3] F,
        ndarray[float32_t, ndim=2] u,
        ndarray[double, ndim=2] center
    ):
        cdef int j, i

        for j in range(self.c):
            p1 = np.zeros((1, ndim), dtype=np.float32)
            p2 = 0
            for i in range(len(X)):
                F1 = np.transpose(np.squeeze(F[i, j, :]))
                p1 += u[i, j] ** self.m * F1
                p2 += u[i, j] ** self.m
            center[j, :] = ((center[j, :] + p1 / p2 + PI) % TWO_PI) - PI


    cpdef __get_obj_fcn(
        self, 
        ndarray X,
        int ndim,
        ndarray[float32_t, ndim=2] u,
        ndarray[double, ndim=2] center
    ):
        cdef int obj_fcn = 0
        cdef int i, j

        for i in range(len(X)):
            for j in range(self.c):
                if ndim == 1:
                    obj_fcn += u[i, j] ** self.m * abs(((X[i, :] - center[j, :] + PI) % TWO_PI) - PI)
                else:
                    obj_fcn += u[i, j] ** self.m * np.linalg.norm(((X[i, :] - center[j, :] + PI) % TWO_PI) - PI)
        
        return obj_fcn


    cpdef fit_predict(self, ndarray X, y=None):
        if self.c >= len(X):
            raise Exception('c must be assigned a value < |X|')

        cdef int ndim = self.__get_dimensionality(X)
        cdef ndarray[double, ndim=2] center = self.random.uniform(NEGATIVE_PI, PI, (self.c, ndim))
        cdef ndarray[float32_t, ndim=2] u = np.zeros((len(X), self.c), dtype=np.float32)
        cdef ndarray[float32_t, ndim=3] F = np.zeros((len(X), self.c, ndim), dtype=np.float32)
        cdef ndarray[float32_t] obj_fcn = np.zeros((self.max_iterations,), dtype=np.float32)
        cdef int it

        for it in range(0, self.max_iterations):
            self.__assign_angle_diffs(X, F, center)
            self.__assign_memberships(X, ndim, F, u)
            self.__assign_centers(X, ndim, F, u, center)
            obj_fcn[it] = self.__get_obj_fcn(X, ndim, u, center)

            if it == 0:
                continue

            if abs(obj_fcn[it] - obj_fcn[it-1]) < self.eps:
                return (center, u)

        return (center, u)
