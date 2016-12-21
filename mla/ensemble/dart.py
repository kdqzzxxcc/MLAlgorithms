# coding=utf-8
__author__ = 'kdq'
from gbm import LeastSquaresLoss, LogisticLoss, PairwiseLoss
import numpy as np
# logistic function
from scipy.special import expit
from mla.base import BaseEstimator
from mla.ensemble.base import mse_criterion
from mla.ensemble.tree import Tree
import time
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import random
import scipy
import sklearn
from NDCG import dcg_k, dcg, idcg, idcg_k, single_dcg
random.seed(1234)

class DARTBase(BaseEstimator):
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

    def sample(self):
        drop_tree = []
        y_pred = np.zeros(self.n_samples, np.float32)
        if len(self.trees) == 0:
            return y_pred, []
        # drop trees from uniform sampling with probability self.p
        for i, tree in enumerate(self.trees):
            rand = np.random.uniform(0, 1)
            if rand < self.p:
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
        d_n = []
        for n in range(self.n_estimators):
            y_pred, drop_tree = self.sample()
            # print 'tree:{}'.format(n)
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
            # t1 = time.time()
            # print 'start train:{}'.format(time.ctime())
            # tree.train(self.X, targets, max_features=self.max_features,
            #            min_samples_split=self.min_samples_split, max_depth=self.max_depth, loss=self.loss)
            tree = DecisionTreeRegressor(criterion='friedman_mse', splitter="best", max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split, max_features=self.max_features,
                                         min_samples_leaf=self.min_samples_leaf, max_leaf_nodes=self.max_leaf_nodes)
            tree.fit(self.X, residuals)
            # print 'train decision from sklearn {}'.format(time.time() - t1)
            predictions = tree.predict(self.X)
            # predictions = tree.predict(self.X)
            error = self.loss.error(self.y, predictions)
            # y_pred += self.learning_rate * predictions
            self.raw_y_pred.append(predictions)
            self.trees.append(tree)
            self.rank.append(error)
            # rewrite weight
            l = len(drop_tree)
            d_n.extend(drop_tree)
            self.weight.append(self.learning_rate / (l + 1))
            for idx in drop_tree:
                self.weight[idx] *= 1.0 * l / (l + 1)
        # print d_n
        print len(d_n)

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
    
class DARTRanker(DARTBase):
    def _setup_input(self, X, y=None, qids=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError('Number of features must be > 0')

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError('Missed required argument y')

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError('Number of targets must be > 0')

        self.y = y
        
        # qid, a, b: qid, start of qid, end of qid
        self.qids = {}
        pre_qid = qids[0]
        pre_idx = 0
        for idx, qid in enumerate(qids):
            if pre_qid != 0 and pre_qid != qid:
                self.qids[pre_qid] = (pre_qid, pre_idx, idx + 1)
                pre_idx = idx
                pre_qid = qid
        self.qids[pre_qid] = (pre_qid, pre_idx, len(qids) + 1)

    def _update_terminal_regions(self, tree, X, y, lambdas, deltas):
        terminal_regions = tree.apply(X)
        masked_terminal_regions = terminal_regions.copy()
        
        # no subsample, so no mask
        # masked_terminal_regions[~sample_mask] = -1

        for leaf in np.where(tree.children_left ==
                                     sklearn.tree._tree.TREE_LEAF)[0]:
            terminal_region = np.where(masked_terminal_regions == leaf)
            suml = np.sum(lambdas[terminal_region])
            sumd = np.sum(deltas[terminal_region])
            tree.value[leaf, 0, 0] = 0.0 if sumd == 0.0 else (suml / sumd)

        # y_pred += tree.value[terminal_regions, 0, 0] * self.learning_rate
    
    def _calc_lambdas_deltas(self, qid, y, y_pred, idcg):
        ns = y.shape[0]
        lambdas = np.zeros(ns)
        deltas = np.zeros(ns)
        
        sorted_y_pred = np.argsort(y_pred)[::-1]
        rev_sorted_y_pred = np.argsort(sorted_y_pred)
        actual = y[sorted_y_pred]
        pred = y_pred[sorted_y_pred]
        
        dcgs = {}
        
        for i in xrange(ns):
            dcgs[(i, i)] = single_dcg(actual, i, i)
            for j in xrange(i + 1, ns):
                if actual[i] == actual[j]:
                    continue
                dcgs[(i, j)] = single_dcg(actual, i, j)
                dcgs[(j, i)] = single_dcg(actual, j, i)
        
        for i in xrange(ns):
            for j in xrange(i + 1, ns):
                if actual[i] == actual[j]:
                    continue
                    
                deltas_ndcg = np.abs(dcgs[(i, j)] + dcgs[(j, i)] - dcgs[(i, i)] - dcgs[(j, j)])
                
                if actual[i] < actual[j]:
                    logistic = scipy.special.expit(pred[i] - pred[j])
                    l = logistic * deltas_ndcg
                    lambdas[i] -= l
                    lambdas[j] += l
                else:
                    logistic = scipy.special.expit(pred[j] - pred[i])
                    l = logistic * -deltas_ndcg
                    lambdas[i] += l
                    lambdas[j] -= l
                gradient = (1 - logistic) * l
                deltas[i] += gradient
                deltas[j] += gradient
                
        return lambdas[rev_sorted_y_pred], deltas[rev_sorted_y_pred]

    def fit(self, X, y=None, qids=None):
        self.loss = PairwiseLoss()
        self._setup_input(X, y, qids)
        self._train()
    
    
    def _train(self):
        all_lambdas = np.zeros(self.n_samples, np.float32)
        all_deltas = np.zeros(self.n_samples, np.float32)
        idcgs = {}
        for qid, a, b in self.qids.values():
            idcgs[qid] = idcg(self.y[a:b])
        # sample_mask = np.zeros(self.n_samples, dtype=np.bool)
        for n in range(self.n_estimators):
            # calculate lambdas & deltas
            y_pred, drop_tree = self.sample()
            for qid, a, b in self.qids.values():
                lambdas, deltas = self._calc_lambdas_deltas(qid, self.y[a:b], y_pred[a:b], idcgs[qid])
                all_lambdas[a:b] = lambdas
                all_deltas[a:b] = deltas

            tree = DecisionTreeRegressor(criterion='friedman_mse', splitter="best", max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split, max_features=self.max_features,
                                         min_samples_leaf=self.min_samples_leaf, max_leaf_nodes=self.max_leaf_nodes)
            
            tree.fit(self.X, all_lambdas)
            self._update_terminal_regions(tree.tree_, self.X, self.y, all_lambdas, all_deltas)
            
            predictions = tree.predict(self.X)
            
            error = self.loss.error(self.y, predictions, self.qids)
            self.raw_y_pred.append(predictions)
            self.trees.append(tree)
            self.rank.append(error)
            # rewrite weight
            l = len(drop_tree)
            self.weight.append(self.learning_rate / (l + 1))
            for idx in drop_tree:
                self.weight[idx] *= 1.0 * l / (l + 1)