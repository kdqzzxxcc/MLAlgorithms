# coding=utf-8
__author__ = 'kdq'
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification, regression
from sklearn.ensemble import GradientBoostingClassifier as SKGBC
from sklearn.ensemble import RandomForestClassifier as SKRFC
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold

from mla.ensemble import RandomForestClassifier, RandomForestRegressor
from mla.ensemble.gbm import GradientBoostingClassifier, GradientBoostingRegressor
from mla.ensemble.dart import DARTClassifier, DARTRegressor
from mla.ensemble.pdart import pDARTRegressor, pDARTClassifier

import xgboost as xgb

import pandas as pd
import numpy as np
import time

import fcntl
from multiprocessing import Process, Pool, cpu_count
import itertools

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
def test_skgbc(X_train, y_train, X_test, y_test):
    model = SKGBC(n_estimators=250, max_depth=5, max_features=None, learning_rate=0.3, max_leaf_nodes=40)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = classification.accuracy_score(y_test, predictions)
    print 'acc:{}'.format(acc)
    return acc


@get_time
def test_skrfc(X_train, y_train, X_test, y_test):
    model = SKRFC(n_estimators=500, max_depth=5, max_features=None, max_leaf_nodes=1000, n_jobs=-1, bootstrap=True)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = classification.accuracy_score(y_test, predictions)
    print 'acc:{}'.format(acc)
    return acc

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
    model = DARTClassifier(n_estimators=250, max_depth=5, max_features=None, max_leaf_nodes=40, p=0.01)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    acc = classification.accuracy_score(y_test, predictions)
    print 'acc:{}'.format(acc)
    return acc


@get_time
def test_pdart(X_train, y_train, X_test, y_test):
    model = pDARTClassifier(n_estimators=250, max_features=None, max_depth=5, max_leaf_nodes=40)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    acc = classification.accuracy_score(y_test, predictions)
    print 'acc:{}'.format(acc)
    return acc


def test_xgbgbr(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.05, n_estimators=1000, objective='reg:linear', nthread=-1)

def test_xgbdart(X_train, y_train, X_test, y_test):
    param = {
        'booster': 'dart',
         'max_depth': 5, 'learning_rate': 0.1,
         'objective': 'reg:linear', 'silent': False,
         'sample_type': 'uniform',
         'normalize_type': 'tree',
         'rate_drop': 0.1,
         'skip_drop': 0.1
    }
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    clf = xgb.train(params=param, dtrain=dtrain, num_boost_round=50)

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

# data = pd.read_csv('~/datasets/slice_data/slice_localization_data.csv')
# print data.info()
# idx = data.patientId.unique().tolist()
# kf = KFold(n_splits=10)

def parallel_classification(parameters):
    model_name = parameters['model']
    parameters.pop('model')
    model = model_map[model_name](**parameters)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    acc = classification.accuracy_score(y_test, predictions)
    f = Lock('/home/ilab/pdart_experiments/classification_{}.csv'.format(model_name))
    f.acquire()
    for k, v in parameters.items():
        f.handle.write('{}:{},'.format(k, v))
    f.handle.write('{}\n'.format(acc))
    f.release()

cores = cpu_count()
train = np.fromfile('/home/ilab/datasets/fd/fd_train.dat', dtype='uint8')
label = pd.read_csv('/home/ilab/datasets/fd/fd_train.lab', header=None).values
label[label < 0] = 0
label = label.reshape((label.shape[0], ))
train = train.reshape((train.shape[0] / 900, 900))

X_train = train[:300000]
y_train = label[:300000]
X_test = train[500000:700000]
y_test = label[500000:700000]

gbdt_parameters = []
rf_parameters = []
dart_parameters = []
pdart_parameters = []

# build gbdt parameters
for n_tree in [50, 100, 250, 500, 1000]:
    for lr in [0.2, 0.3, 0.4, 0.5]:
        for max_f in [0.5, 0.75, None]:
            d = {'n_estimators': n_tree, 'max_depth': 5, 'max_leaf_nodes': 40, 'learning_rate': lr, 'max_features': max_f, 'model': 'GBRT'}
            gbdt_parameters.append(d)

# build rf parameters
for n_tree in [50, 100, 250, 500, 1000]:
    for max_n in [50, 100, 250, 1000]:
        for max_f in [0.5, 0.75, None]:
            d = {'n_estimators': n_tree, 'max_depth': 5, 'max_leaf_nodes': max_n, 'max_features': max_f, 'model': 'RF'}
            rf_parameters.append(d)

# build dart parameters
for n_tree in [50, 100, 250, 500, 1000]:
    for p in [0, 0.015, 0.03, 0.045]:
        for max_f in [0.5, 0.75, None]:
            d = {'n_estimators': n_tree, 'max_depth': 5, 'max_leaf_nodes': 40, 'p': p, 'max_features': max_f, 'model': 'DART'}
            dart_parameters.append(d)

# build pdart parameters
for n_tree in [50, 100, 250, 500, 1000]:
    for max_f in [0.5, 0.75, None]:
        d = {'n_estimators': n_tree, 'max_depth': 5, 'max_leaf_nodes': 40, 'max_features': max_f, 'model': 'PDART'}
        pdart_parameters.append(d)

model_map = {'RF': SKRFC, 'GBRT': SKGBC, 'DART': DARTClassifier, 'PDART':pDARTClassifier}

pool = Pool(cores / 2)
print('start experiment :{}'.format(time.ctime()))
pool.map(parallel_classification, gbdt_parameters)

print('end experiment :{}'.format(time.ctime()))


# def time_out(x):
#     print x
#     return x
# pool = Pool(4)
# res = pool.map(time_out, [{'a':1}, {'b':2}])
# print res

# Usagetry:
# lock = Lock("./test")
# lock.acquire()	# Do important stuff that needs to be synchronizedfinally:
# lock.release()