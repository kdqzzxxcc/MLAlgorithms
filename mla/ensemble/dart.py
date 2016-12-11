# coding=utf-8
__author__ = 'kdq'
from gbm import LeastSquaresLoss, LogisticLoss
import numpy as np
# logistic function
from scipy.special import expit
from mla.base import BaseEstimator
from mla.ensemble.base import mse_criterion
from mla.ensemble.tree import Tree

class DARTBase(BaseEstimator):
    def __init__(self, n_estimators, learning_rate=0.1, max_features=10, max_depth=2, min_samples_split=10):
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []
        self.weight = []
        self.rank = []
        self.loss = None

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.y_mean = np.mean(y)
        self._train()

    def sample(self):
        if len(self.trees) == 0:
            return []
        tmp_tree = []
        for i, tree in enumerate(self.trees):
            #
            pass

    def _train(self):
        # Initialize model with zeros
        y_pred = np.zeros(self.n_samples, np.float32)
        # Or mean
        # y_pred = np.full(self.n_samples, self.y_mean)

        for n in range(self.n_estimators):
            y_pred, tmp_tree = self.sample()

            residuals = self.loss.grad(self.y, y_pred)
            tree = Tree(regression=True, criterion=mse_criterion)
            # Pass multiple target values to the tree learner
            targets = {
                # Residual values
                'y': residuals,
                # Actual target values
                'actual': self.y,
                # Predictions from previous step
                'y_pred': y_pred
            }
            tree.train(self.X, targets, max_features=self.max_features,
                       min_samples_split=self.min_samples_split, max_depth=self.max_depth, loss=self.loss)
            predictions = tree.predict(self.X)
            error = self.loss.error(self.y, predictions)
            # y_pred += self.learning_rate * predictions

            self.trees.append(tree)
            self.rank.append(error)
            self.weight.append(self.learning_rate * 1)

    def _predict(self, X=None):
        y_pred = np.zeros(X.shape[0], np.float32)

        for i, tree in enumerate(self.trees):
            y_pred += self.weight[i] * tree.predict(X)
        return y_pred

    def predict(self, X=None):
        return self.loss.transform(self._predict(X))


class DARTRegressor(DARTBase):
    def fit(self, X, y=None):
        self.loss = LeastSquaresLoss()
        super(DARTRegressor, self).fit(X, y)


class DARTClassifier(DARTBase):
    def fit(self, X, y=None):
        # Convert labels from {0, 1} to {-1, 1}
        y = (y * 2) - 1
        self.loss = LogisticLoss()
        super(DARTClassifier, self).fit(X, y)