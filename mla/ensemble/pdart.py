# coding=utf-8
__author__ = 'kdq'
from gbm import LeastSquaresLoss, LogisticLoss
import numpy as np
# logistic function
from scipy.special import expit
from mla.base import BaseEstimator
from mla.ensemble.base import mse_criterion
from mla.ensemble.tree import Tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import random
from matplotlib import pyplot as plt
random.seed(1234)

class pDARTBase(BaseEstimator):
    '''
    n_estimators: #trees
    learning_rate: eta
    max_feature: default:None, all feature
                 integer:#features
                 float:fraction of features
    max_depth: default:2
    min_samples_split: min samples for a node to split
    p:drop probability
    min_samples_leaf:default:1, min #nodes in a leaf
     max_leaf_nodes:default:None, max leaf nodes in a tree
    '''
    def __init__(self, n_estimators, learning_rate=1., max_features=None, max_depth=2, min_samples_split=10, p=0.1,
                 min_samples_leaf=1, max_leaf_nodes=None):
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.p = p
        self.trees = []
        self.weight = []
        self.rank = []
        self.raw_y_pred = []
        self.loss = None

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.y_mean = np.mean(y)
        self._train()

    def plot(self, X):
        x = range(len(X))
        plt.figure(1)
        plt.plot(x, X)
        plt.show()

    def sample(self):
        drop_tree = []
        y_pred = np.zeros(self.n_samples, np.float32)
        if len(self.trees) == 0:
            return y_pred, []
        # drop trees from uniform sampling with probability self.p
        rank = np.fabs(self.rank).argsort().argsort() + 1
        for i, tree in enumerate(self.trees):
            rand = np.random.uniform(0, 1)
            # p_{i} = 1 / rank
            if rand < 1. / rank[i]:
                drop_tree.append(i)
            else:
                y_pred += self.weight[i] * self.raw_y_pred[i]
        # if none of trees were droped, random choice 1 tree to drop
        if len(drop_tree) == 0:
            idx = np.random.choice(range(len(self.trees)))
            drop_tree.append(idx)
            y_pred += self.weight[idx] * self.raw_y_pred[idx]
        return y_pred, drop_tree

    def _train(self):
        # Initialize model with zeros
        # y_pred = np.zeros(self.n_samples, np.float32)
        # Or mean
        # y_pred = np.full(self.n_samples, self.y_mean)

        for n in range(self.n_estimators):
            y_pred, drop_tree = self.sample()

            residuals = self.loss.grad(self.y, y_pred)
            # tree = Tree(regression=True, criterion=mse_criterion)
            # Pass multiple target values to the tree learner
            targets = {
                # Residual values
                'y': residuals,
                # Actual target values
                'actual': self.y,
                # Predictions from previous step
                'y_pred': y_pred
            }
            # tree.train(self.X, targets, max_features=self.max_features,
            #            min_samples_split=self.min_samples_split, max_depth=self.max_depth, loss=self.loss)
            tree = DecisionTreeRegressor(criterion='mse', splitter="best", max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split, max_features=self.max_features,
                                         min_samples_leaf=self.min_samples_leaf, max_leaf_nodes=self.max_leaf_nodes)
            tree.fit(self.X, residuals)
            predictions = tree.predict(self.X)
            error = self.loss.error(self.y, predictions)
            # y_pred += self.learning_rate * predictions
            self.raw_y_pred.append(predictions)
            self.trees.append(tree)
            self.rank.append(error)
            # rewrite weight
            l = len(drop_tree)
            self.weight.append(self.learning_rate / (l + 1))
            for idx in drop_tree:
                self.weight[idx] *= 1.0 * l / (l + 1)
        # self.plot(self.rank)

    def _predict(self, X=None):
        y_pred = np.zeros(X.shape[0], np.float32)

        for i, tree in enumerate(self.trees):
            y_pred += self.weight[i] * tree.predict(X)
        return y_pred

    def predict(self, X=None):
        return self.loss.transform(self._predict(X))


class pDARTRegressor(pDARTBase):
    def fit(self, X, y=None):
        self.loss = LeastSquaresLoss()
        super(pDARTRegressor, self).fit(X, y)


class pDARTClassifier(pDARTBase):
    def fit(self, X, y=None):
        # Convert labels from {0, 1} to {-1, 1}
        y = (y * 2) - 1
        self.loss = LogisticLoss()
        super(pDARTClassifier, self).fit(X, y)