import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class PSKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4,
                 random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers = np.zeros(n_clusters)

    def __init_centers__(self, X):
        np.random.seed(self.random_state)
        self.centers = X[sorted(np.random.randint(0, len(X), self.n_clusters))]

    def __update_clusters__(self, X):
        y = np.zeros(len(X))
        for i in range(len(X)):
            diff = self.centers - X[i]
            y[i] = np.argmin(np.linalg.norm(diff, axis=1))
        return y
    
    def __update_centers__(self, X, y):
        for i in range(len(self.centers)):
            bin_mask = y == i
            self.centers[i] = np.mean(X[bin_mask], axis=0)
    
    def fit(self, X, y=None):
        self.__init_centers__(X)

        for _ in range(self.max_iter):
            last_centers = self.centers
            
            y = self.__update_clusters__(X)
                
            self.__update_centers__(X, y)

            max_change = np.max(
                np.linalg.norm(self.centers - last_centers, axis=1))
            if (max_change <= self.tol):
                break
        return self

    def predict(self, X):
        y = np.zeros(len(X))
        for i in range(len(X)):
            diff = self.centers - X[i]
            y[i] = np.argmin(np.linalg.norm(diff, axis=1))
        return y
    
    def fit_predict(self, X, y=None):
        self.fit(X)
        return predict(X)
    

    
class PSKMedians(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4,
                 random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers = np.zeros(n_clusters)

    def __init_centers__(self, X):
        np.random.seed(self.random_state)
        self.centers = X[sorted(np.random.randint(0, len(X), self.n_clusters))]

    def __update_clusters__(self, X):
        y = np.zeros(len(X))
        for i in range(len(X)):
            diff = self.centers - X[i]
            y[i] = np.argmin(np.linalg.norm(diff, axis=1))
        return y
    
    def __update_centers__(self, X, y):
        for i in range(len(self.centers)):
            bin_mask = y == i
            self.centers[i] = np.median(X[bin_mask], axis=0)
    
    def fit(self, X, y=None):
        self.__init_centers__(X)

        for _ in range(self.max_iter):
            last_centers = self.centers
            
            y = self.__update_clusters__(X)
                
            self.__update_centers__(X, y)

            max_change = np.max(
                np.linalg.norm(self.centers - last_centers, axis=1))
            if (max_change <= self.tol):
                break
        return self

    def predict(self, X):
        y = np.zeros(len(X))
        for i in range(len(X)):
            diff = self.centers - X[i]
            y[i] = np.argmin(np.linalg.norm(diff, axis=1))
        return y
    
    def fit_predict(self, X, y=None):
        self.fit(X)
        return predict(X)