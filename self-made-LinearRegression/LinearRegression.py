import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, fit_intercept=True, normalize=False):
        self.__normalize__ = normalize
        self.__fit_intercept__ = fit_intercept

    def __make_normalized__(self, X):
        means, stds = X.mean(axis=0), X.std(axis=0)
        return (X - means) / stds

    def __add_intercept__(self, X):
        one = np.ones(X.shape[0]).reshape(X.shape[0], -1)
        return np.hstack((X, one))

    def __mean_squared_error__(self, y, y_pred):
        return np.mean((y_pred - y) ** 2)

    def __normal_equation__(self, X, y):
        self.w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

    # One stochastic gradient descent step on given example (index)
    def __stochastic_gradient_step__(self, X, y, w, train_ind, eta=0.01):
        norm_scalar = ((np.dot(X[train_ind], w) - y[train_ind]) / X.shape[0])
        return w - 2 * eta * norm_scalar * X[train_ind].reshape(X[train_ind].shape[0], -1)

    # Find the minimum solution for MSE Functional with stochastic gradient descent
    def __stochastic_gradient_descent__(self, X, y, w_init, eta=1e-2, max_iter=1e4,
                                        min_weight_dist=1e-8, seed=42, verbose=False):

        # init weights distance with infinity
        weight_dist = np.inf

        # init weights
        w = w_init.reshape(X.shape[1], -1)

        # list for errors on all iterations
        errors = []

        # Iteration counter
        iter_num = 0

        # set seed
        np.random.seed(seed)

        # Gradient descent
        while weight_dist > min_weight_dist and iter_num < max_iter:

            # Generate random index of sample
            random_ind = np.random.randint(X.shape[0])

            # make step and add error
            w_new = self.__stochastic_gradient_step__(X, y, w, random_ind, eta)
            errors.append(self.__mean_squared_error__(y, np.dot(X, w)))

            # Find the Euclidean norm of old and new weights
            weight_dist = np.linalg.norm(w_new - w)

            iter_num += 1

            # update weights
            w = w_new

            # print info
            if verbose:
                print('Iter = {0}; Error = {1}; w = {2}'.format(iter_num, errors[iter_num - 1], w))

        return w, errors

    # Fits given samples
    # 2 type of fit are available: analylical solution or stochastic gradient descent
    def fit(self, X, y, fit_type='sgd', sample_weight=None, eta=1e-2, max_iter=1e4,
            min_weight_dist=1e-8, seed=42, verbose=False):

        # normalize and add intercept if needed
        if self.__normalize__:
            X = self.__make_normalized__(X)
        if self.__fit_intercept__:
            X = self.__add_intercept__(X)

        # if analytical solution is asked
        if not (fit_type == 'sgd'):
            self.__normal_equation__(X, y)
            return self

        # init weights with numbers close to zero
        if not sample_weight:
            np.random.seed(seed)
            w_init = np.random.uniform(high=5e-4, size=X.shape[1])
        self.w, self.errors = self.__stochastic_gradient_descent__(X, y, w_init, eta, max_iter, min_weight_dist, seed,
                                                                   verbose)
        return self

    # Predicts target variable for given X samples
    # (Model must be fitted)
    def predict(self, X):

        # normalize and add intercept if needed
        if self.__normalize__:
            X = self.__make_normalized__(X)
        if self.__fit_intercept__:
            X = self.__add_intercept__(X)
        return np.dot(X, self.w)

    # MSE(y, predict(X))
    def score(self, X, y):
        return self.__mean_squared_error__(y, self.predict(X))
