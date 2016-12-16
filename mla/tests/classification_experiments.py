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
    acc = classification.accuracy_score(y_test, predictions)
    print 'acc:{}'.format(acc)
    return acc


@get_time
def test_pdart(X_train, y_train, X_test, y_test):
    model = pDARTClassifier(n_estimators=250, max_features=None, max_depth=5, max_leaf_nodes=40)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
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


# data = pd.read_csv('~/datasets/slice_data/slice_localization_data.csv')
# print data.info()
# idx = data.patientId.unique().tolist()
# kf = KFold(n_splits=10)

train = np.fromfile('/home/ilab/datasets/fd/fd_train.dat', dtype='uint8')
label = pd.read_csv('/home/ilab/datasets/fd/fd_train.lab', header=None).values
label[label < 0] = 0
label = label.reshape((label.shape[0], ))
train = train.reshape((train.shape[0] / 900, 900))

X_train = train[:300000]
y_train = label[:300000]
X_test = train[300000:500000]
y_test = label[300000:500000]

skgbr_mse = []
skrf_mse = []
dart_mse = []
pdart_mse = []


print('start experiment :{}'.format(time.ctime()))

test_skrfc(X_train, y_train, X_test, y_test)
test_skgbc(X_train, y_train, X_test, y_test)
test_dart(X_train, y_train, X_test, y_test)
test_pdart(X_train, y_train, X_test, y_test)

print('end experiment :{}'.format(time.ctime()))