# coding=utf-8
__author__ = 'kdq'
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification, regression
from sklearn.ensemble import GradientBoostingRegressor as SKGBR
from sklearn.ensemble import RandomForestRegressor as SKRF
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold

from mla.ensemble import RandomForestClassifier, RandomForestRegressor
from mla.ensemble.gbm import GradientBoostingClassifier, GradientBoostingRegressor
from mla.ensemble.dart import DARTClassifier, DARTRegressor
from mla.ensemble.pdart import pDARTRegressor

import xgboost as xgb
from multiprocessing import Pool
import pandas as pd
import numpy as np
import time, os, fcntl

def get_time(f):
    def inner(*args, **kwargs):
        t1 = time.time()
        print
        print('========= now is running on function {} ========'.format(f.__name__))
        ff = f(*args, **kwargs)
        print('run time :{}'.format(time.time() - t1))
        return ff
    return inner


@get_time
def test_skgbr(X_train, y_train, X_test, y_test):
    model = SKGBR(n_estimators=1000, max_depth=5, max_features=0.2, learning_rate=0.05, max_leaf_nodes=50)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = regression.mean_squared_error(y_test, predictions)
    # print np.sum((y_test - predictions) ** 2)
    print 'mse:{}'.format(mse)
    return mse

@get_time
def test_skrf(X_train, y_train, X_test, y_test):
    model = SKRF(n_estimators=1000, max_depth=5, max_features=0.4, max_leaf_nodes=1000, n_jobs=-1, bootstrap=True)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = regression.mean_squared_error(y_test, predictions)
    print 'mse:{}'.format(mse)
    return mse

@get_time
def test_gbr(X_train, y_train, X_test, y_test):
    model = GradientBoostingRegressor(n_estimators=1000, max_depth=5, max_features=None, learning_rate=0.05)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = regression.mean_squared_error(y_test, predictions)
    print 'mse:{}'.format(mse)
    return mse
    # print np.sum((y_test - predictions) ** 2)

@get_time
def test_rf(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=1000, max_features=None, min_samples_split=25, max_depth=5, criterion='mse')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = regression.mean_squared_error(y_test, predictions)
    print 'mse:{}'.format(mse)
    return mse

@get_time
def test_dart(X_train, y_train, X_test, y_test):
    model = DARTRegressor(n_estimators=1000, max_depth=5, max_features=0.4, max_leaf_nodes=50, p=0.01)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = regression.mean_squared_error(y_test, predictions)
    print 'mse:{}'.format(mse)
    return mse

@get_time
def test_pdart(X_train, y_train, X_test, y_test):
    model = pDARTRegressor(n_estimators=50, max_features=0.05, max_depth=5, max_leaf_nodes=50)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = regression.mean_squared_error(y_test, predictions)
    print 'mse:{}'.format(mse)
    return mse

def test_xgbgbr(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.05, n_estimators=1000, objective='reg:linear', nthread=-1)

def test_xgbdart(X_train, y_train, X_test, y_test):
    param = {
        'booster': 'dart',
         'max_depth': 5, 'learning_rate': 1,
         'objective': 'reg:linear', 'silent': False,
         'sample_type': 'uniform',
         'normalize_type': 'tree',
         'rate_drop': 0.1,
         'skip_drop': 0.1
    }
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    clf = xgb.train(params=param, dtrain=dtrain, num_boost_round=50)
    model = xgb.XGBRegressor(max_depth=5, n_estimators=100)


data = pd.read_csv('~/datasets/slice_data/slice_localization_data.csv')
print data.info()
idx = data.patientId.unique().tolist()
kf = KFold(n_splits=10)

skgbr_mse = []
skrf_mse = []
dart_mse = []
pdart_mse = []


def parallel_skrf(X_train, y_train, X_test, y_test):
    n_estimators = [50, 100, 250, 500, 1000]
    max_features = [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.8, None]
    max_leaf_node = [25, 50, 100, 250, 500, 1000]
    max_depth = 5
    criterion = 'mse'


#
# print('start experiment :{}'.format(time.ctime()))
#
# for k, (train_idx, test_idx) in enumerate(kf.split(idx)):
#     print('K-fold:{}'.format(k))
#     t1 = time.time()
#     train = data[data['patientId'].isin(train_idx)]
#     test = data[data['patientId'].isin(test_idx)]
#     X_train = train.drop(['patientId', 'reference'], axis=1)
#     y_train = train['reference']
#     X_test = test.drop(['patientId', 'reference'], axis=1)
#     y_test = test['reference']
#     print('get data ready:{}'.format(time.time() - t1))
#
#     # skrf_mse.append(test_skrf(X_train, y_train, X_test, y_test))
#     # skgbr_mse.append(test_skgbr(X_train, y_train, X_test, y_test))
#     # dart_mse.append(test_dart(X_train, y_train, X_test, y_test))
#     pdart_mse.append(test_pdart(X_train, y_train, X_test, y_test))
#
# print('mse skgbr:{}'.format(np.mean(skgbr_mse)))
# print('mse skrf:{}'.format(np.mean(skrf_mse)))
# print('mse dart:{}'.format(np.mean(dart_mse)))
# print('mse pdart:{}'.format(np.mean(pdart_mse)))
#
# print('end experiment :{}'.format(time.ctime()))

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

def parallel_regression(parameters):
    model_name = parameters['model']
    parameters.pop('model')
    model = model_map[model_name](**parameters)
    acc = []
    for k, (train_idx, test_idx) in enumerate(kf.split(idx)):
        # t1 = time.time()
        train = data[data['patientId'].isin(train_idx)]
        test = data[data['patientId'].isin(test_idx)]
        X_train = train.drop(['patientId', 'reference'], axis=1)
        y_train = train['reference']
        X_test = test.drop(['patientId', 'reference'], axis=1)
        y_test = test['reference']
        # print('get data ready:{}'.format(time.time() - t1))
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc.append(regression.mean_squared_error(y_test, predictions))
    f = Lock('/home/ilab/pdart_experiments/regression_{}.csv'.format(model_name))
    f.acquire()
    for k, v in parameters.items():
        f.handle.write('{}:{},'.format(k, v))
    f.handle.write('{}\n'.format(np.mean(acc)))
    f.release()

def change_value(x):
    if x == 'None':
        return None
    if x.count('.'):
        return float(x)
    return int(x)

def parameters_recover(parameters, model_name):
    already_parameters = []
    if os.path.exists('/home/ilab/pdart_experiments/regression_{}.csv'.format(model_name)):
        with open('/home/ilab/pdart_experiments/regression_{}.csv'.format(model_name)) as f:
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

gbdt_parameters = []
rf_parameters = []
dart_parameters = []
pdart_parameters = []

# build gbdt parameters
for n_tree in [50, 100, 250, 500, 1000]:
    for lr in [0.05, 0.1, 0.2, 0.3, 0.5]:
        for max_f in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for max_leaf in [50, 100, 250, 500, 1000]:
                d = {
                    'n_estimators': n_tree, 'max_depth': 5, 'max_leaf_nodes': max_leaf, 'learning_rate': lr,'max_features': max_f, 'model': 'GBRT'
                }
                gbdt_parameters.append(d)

# build rf parameters
for n_tree in [50, 100, 250, 500, 1000]:
    for max_n in [50, 100, 250, 500, 1000]:
        for max_f in [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.8, 1.0]:
            d = {'n_estimators': n_tree, 'max_depth': 5, 'max_leaf_nodes': max_n, 'max_features': max_f, 'model': 'RF', 'n_jobs':-1}
            rf_parameters.append(d)

# build dart parameters
for n_tree in [50, 100, 250, 500, 1000]:
    for p in [0, 0.01, 0.025, 0.05, 0.1, 0.2]:
        for max_f in [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]:
            for max_leaf in [50, 100, 250, 500, 1000]:
                d = {'n_estimators': n_tree, 'max_depth': 5, 'max_leaf_nodes': max_leaf, 'p': p, 'max_features': max_f, 'model': 'DART'}
                dart_parameters.append(d)

# build pdart parameters
for n_tree in [50, 100, 250, 500, 1000]:
    for max_f in [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]:
        for max_leaf in [50, 100, 250, 500, 1000]:
            d = {'n_estimators': n_tree, 'max_depth': 5, 'max_leaf_nodes': max_leaf, 'max_features': max_f, 'model': 'PDART'}
            pdart_parameters.append(d)

parameters_recover(gbdt_parameters, 'GBRT')
parameters_recover(rf_parameters, 'RF')
parameters_recover(dart_parameters, 'DART')
parameters_recover(pdart_parameters, 'PDART')

model_map = {'RF': SKRF, 'GBRT': SKGBR, 'DART': DARTRegressor, 'PDART':pDARTRegressor}

pool = Pool(8)
print('start experiment :{}'.format(time.ctime()))
pool.map(parallel_regression, gbdt_parameters)
pool.map(parallel_regression, dart_parameters)
pool.map(parallel_regression, pdart_parameters)
pool.map(parallel_regression, rf_parameters)
print('end experiment :{}'.format(time.ctime()))
from sklearn import ensemble