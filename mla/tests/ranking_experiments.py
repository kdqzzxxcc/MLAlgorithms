# coding=utf-8
__author__ = 'kdq'
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification, regression
from sklearn.ensemble import GradientBoostingClassifier as SKGBC
from sklearn.ensemble import RandomForestClassifier as SKRFC
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from tqdm import tqdm
from mla.ensemble import RandomForestClassifier, RandomForestRegressor
from mla.ensemble.gbm import GradientBoostingClassifier, GradientBoostingRegressor, PairwiseLoss
from mla.ensemble.dart import DARTClassifier, DARTRegressor, DARTRanker
from mla.ensemble.pdart import pDARTRegressor, pDARTClassifier, pDARTRanker
from mla.ensemble import NDCG
import xgboost as xgb
import pandas as pd
import numpy as np
import cPickle

import time
import os
import fcntl
from multiprocessing import Process, Pool, cpu_count
import itertools

def get_index(idx):
    pre = idx[0]
    index = []
    cnt = 0
    for x in idx:
        if x != pre:
            index.append(cnt)
            cnt = 0
            pre = x
        cnt += 1
    index.append(cnt)
    return index

def test_xgboost_ranking():
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    train_index = get_index(qids_train)
    dtrain.set_group(train_index)
    dtest = xgb.DMatrix(data=X_test)
    test_index = get_index(qids_test)
    dtest.set_group(test_index)
    qids = pDARTRanker._setup_qids(qids_test)
    param_dart = {
        'booster': 'dart','max_depth': 5, 'learning_rate': 0.4,'objective': 'rank:pairwise', 'silent': False,
        'sample_type': 'uniform', 'normalize_type': 'tree','rate_drop': 0.015, 'skip_drop': 0.0, 'lambda':1.2
    }
    param_mart = {
        'booster': 'gbtree', 'eta':0.4, 'max_depth': 5, 'objective': 'rank:pairwise', 'lambda':1.2
    }
    num_round = 1
    
    clf = xgb.train(params=param_dart, dtrain=dtrain, num_boost_round=num_round)
    predictions = clf.predict(dtest)
    ndcg = PairwiseLoss.error(y_test, predictions, qids, k=3)
    print 'dart ndcg is {}'.format(ndcg)
    
    clf = xgb.train(params=param_mart, dtrain=dtrain, num_boost_round=num_round)
    predictions = clf.predict(dtest)
    ndcg = PairwiseLoss.error(y_test, predictions, qids, k=3)
    print 'mart ndcg is {}'.format(ndcg)


class Lock:
    def __init__(self, filename):
        self.filename = filename		# This will create it if it does not exist already
        self.handle = open(filename, 'a')		# Bitwise OR fcntl.LOCK_NB if you need a non-blocking lock
    def acquire(self):
        fcntl.flock(self.handle, fcntl.LOCK_EX)
    def release(self):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
    def __del__(self):
        self.handle.close()


def parallel_ranking(parameters):
    model_name = parameters['model']
    parameters.pop('model')
    model = model_map[model_name](**parameters)
    model.fit(X_train, y_train, qids_train)
    predictions = model.predict(X_test)
    qids = pDARTRanker._setup_qids(qids_test)
    ndcg = PairwiseLoss.error(y_test, predictions, qids, k = 3)
    print ndcg
    # f = Lock('/home/ilab/pdart_experiments/ranking_{}.csv'.format(model_name))
    # f.acquire()
    # for k, v in parameters.items():
    #     f.handle.write('{}:{},'.format(k, v))
    # f.handle.write('{}\n'.format(ndcg))
    # f.release()

def change_value(x):
    if x == 'None':
        return None
    if x.count('.'):
        return float(x)
    return int(x)

def parameters_recover(parameters, model_name):
    already_parameters = []
    if os.path.exists('/home/ilab/pdart_experiments/ranking_{}.csv'.format(model_name)):
        with open('/home/ilab/pdart_experiments/ranking_{}.csv'.format(model_name)) as f:
            for line in f:
                tmp = line.strip().split(',')
                tmp = tmp[:-1]
                d = {'model':model_name}
                for k in tmp:
                    k = k.split(':')
                    d[k[0]] = change_value(k[1])
                already_parameters.append(d)
    for p in already_parameters:
        if p in parameters:
            parameters.remove(p)
    print('======parameter train exist============')
    print('Model: {}, Parameters: {}'.format(model_name, len(parameters)))

dart_parameters = []
pdart_parameters = []


# build dart parameters
for n_tree in [50, 100, 250, 500, 1000, 1500, 2000]:
    for p in [0, 0.015, 0.03, 0.045]:
        for max_f in [0.5, 0.75, None]:
            d = {'n_estimators': n_tree, 'max_depth': 5, 'max_leaf_nodes': 40, 'p': p, 'max_features': max_f, 'model': 'DART'}
            dart_parameters.append(d)

# build pdart parameters
for n_tree in [50, 100, 250, 500, 1000, 1500, 2000]:
    for max_f in [0.5, 0.75, None]:
        d = {'n_estimators': n_tree, 'max_depth': 5, 'max_leaf_nodes': 40, 'max_features': max_f, 'model': 'PDART'}
        pdart_parameters.append(d)

    
model_map = {'DART': DARTRanker, 'PDART':pDARTRanker}

parameters_recover(dart_parameters, 'DART')
parameters_recover(pdart_parameters, 'PDART')

X_data, y_data, qids_data = [], [], []

X_train, y_train, qids_train, X_test, y_test, qids_test = [], [], [], [], [], []

def load_mslr_10k(fold_path):
    train_path = fold_path + 'train.txt'
    test_path = fold_path + 'test.txt'
    
    def read(path):
        qids = []
        X = []
        y = []
        with open(path, 'rb') as f:
            for line in tqdm(f):
                tmp = line.strip().split(' ')
                y.append(int(tmp[0]))
                qids.append(int(tmp[1].split(':')[-1]))
                x = []
                for t in tmp[2:]:
                    x.append(float(t.split(':')[-1]))
                X.append(x)
        return X, y, qids
    
    x, y, qids = read(train_path)
    X_train.extend(x)
    y_train.extend(y)
    qids_train.extend(qids)

    x, y, qids = read(test_path)
    X_test.extend(x)
    y_test.extend(y)
    qids_test.extend(qids)
    
# for idx in range(1, 6):
#     path = '/home/ilab/datasets/MSLR-WEB10K/Fold{}/'.format(idx)
#     print path
#     load_mslr_10k(path)

# train_split = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 0, 1]]
# test_split = [4, 0, 1, 2, 3]


# for train_idx, test_idx in zip(train_split, test_split):
#     X_train = X_data[train_idx]
#     y_train = y_data[train_idx]
#     qids_train = qids_data[train_idx]
#
#     X_test = X_data[test_idx]
#     y_test = y_data[test_idx]
#     qids_test = qids_data[test_idx]

# def test_parallel()

load_mslr_10k('/home/ilab/datasets/MSLR-WEB10K/Fold1/')

pdart_parameters = {'n_estimators': 1, 'max_depth': 5, 'max_leaf_nodes': 40, 'max_features': 0.5, 'model': 'PDART', 'parallel_gradient':False}

# data reform
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
qids_train = np.array(qids_train, dtype=np.float32)
print X_train.shape, y_train.shape, qids_train.shape
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)
qids_test = np.array(qids_test, dtype=np.float32)

print 'start at:{}'.format(time.ctime())
# parallel_ranking(pdart_parameters)
test_xgboost_ranking()
print 'end at :{}'.format(time.ctime())

