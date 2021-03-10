import numpy as np
from GoMapClustering.AngularClustering.fcm4dd import FCM4DD


class FCM4DDHard(FCM4DD):

    def __init__(self, c: int, eps=1e-9, m=2, max_iterations=100, seed=None) -> None:
        '''
        # Parameters
            - c (int): Expected amount of clusters
            - eps (float): If the iterative improvement is less than eps the result is returned
            - m (float): Fuzziness parameter
            - max_iterations (int): The maximum amount of allowed iterations to improve the result
            - seed (int): The seed used to initialize the algorithm
        '''
        super().__init__(c, eps, m, max_iterations, seed)


    def fit_predict(self, X, y=None):
        if self.c >= len(X):
            return np.full((len(X),), -1)

        result = super().fit_predict(X, y)[1]
        rv = np.zeros((len(X),), dtype=np.int64)

        for i in range(len(X)):
            best_fit = 0
            for j in range(self.c):
                if result[i][j] > best_fit:
                    best_fit = result[i][j]
                    rv[i] = j

        return rv

# angles = np.radians([
#     8, 9, 13, 13, 14, 18, 22, 27, 30, 34,
#     38, 38, 40, 44, 45, 47, 48, 48, 48, 48,
#     50, 53, 56, 57, 58, 58, 61, 63, 64, 64,
#     64, 65, 65, 68, 70, 73, 78, 78, 78, 83,
#     83, 88, 88, 88, 90, 92, 92, 93, 95, 96,
#     98, 100, 103, 106, 113, 118, 138, 153, 153, 155,
#     204, 215, 223, 226, 237, 238, 243, 244, 250, 251,
#     257, 268, 285, 319, 343, 350
# ])
# angles = np.reshape(angles, (-1, 1))

# result = FCM4DDHard(2, seed=1337).fit_predict(angles)
# print(result)
