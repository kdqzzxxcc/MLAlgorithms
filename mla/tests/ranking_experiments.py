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
from mla.ensemble.pdart import pDARTRegressor, pDARTClassifier
from mla.ensemble import NDCG
import xgboost as xgb
import pandas as pd
import numpy as np

import time
import os
import fcntl
from multiprocessing import Process, Pool, cpu_count
import itertools

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

def parallel_classification(parameters):
    model_name = parameters['model']
    parameters.pop('model')
    model = model_map[model_name](**parameters)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    ndcg = PairwiseLoss().error(y_test, predictions, qids_test)
    f = Lock('/home/ilab/pdart_experiments/ranking_{}.csv'.format(model_name))
    f.acquire()
    for k, v in parameters.items():
        f.handle.write('{}:{},'.format(k, v))
    f.handle.write('{}\n'.format(ndcg))
    f.release()

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
    
    
model_map = {'RF': SKRFC, 'GBRT': SKGBC, 'DART': DARTClassifier, 'PDART':pDARTClassifier}

X_data, y_data, qids_data = [], [], []

def load_mslr_10k(fold_path):
    # train_path = fold_path + 'train.txt'
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
    
    
    x, y, qids = read(test_path)
    X_data.append(x)
    y_data.append(y)
    qids_data.append(qids)
    
# for idx in range(1, 6):
#     path = '/home/ilab/datasets/MSLR-WEB10K/Fold{}/'.format(idx)
#     print path
#     load_mslr_10k(path)

train_split = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 0, 1]]
test_split = [4, 0, 1, 2, 3]


for train_idx, test_idx in zip(train_split, test_split):
    X_train = X_data[train_idx]
    y_train = y_data[train_idx]
    qids_train = qids_data[train_idx]
    
    X_test = X_data[test_idx]
    y_test = y_data[test_idx]
    qids_test = qids_data[test_idx]


