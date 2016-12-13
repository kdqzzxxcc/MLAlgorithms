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
    model = SKRF(n_estimators=1000, max_depth=5, max_features=0.4, max_leaf_nodes=1000, n_jobs=-1)
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
    model = pDARTRegressor(n_estimators=1000, max_features=0.4, max_depth=5, max_leaf_nodes=50)
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
         'max_depth': 5, 'learning_rate': 0.1,
         'objective': 'reg:linear', 'silent': False,
         'sample_type': 'uniform',
         'normalize_type': 'tree',
         'rate_drop': 0.1,
         'skip_drop': 0.1
    }
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    clf = xgb.train(params=param, dtrain=dtrain, num_boost_round=50)


data = pd.read_csv('~/datasets/slice_data/slice_localization_data.csv')
print data.info()
idx = data.patientId.unique().tolist()
kf = KFold(n_splits=10)

skgbr_mse = []
skrf_mse = []
dart_mse = []
pdart_mse = []

print('start experiment :{}'.format(time.ctime()))

for k, (train_idx, test_idx) in enumerate(kf.split(idx)):
    print('K-fold:{}'.format(k))
    t1 = time.time()
    train = data[data['patientId'].isin(train_idx)]
    test = data[data['patientId'].isin(test_idx)]
    X_train = train.drop(['patientId', 'reference'], axis=1)
    y_train = train['reference']
    X_test = test.drop(['patientId', 'reference'], axis=1)
    y_test = test['reference']
    print('get data ready:{}'.format(time.time() - t1))

    skrf_mse.append(test_skrf(X_train, y_train, X_test, y_test))
    skgbr_mse.append(test_skgbr(X_train, y_train, X_test, y_test))
    dart_mse.append(test_dart(X_train, y_train, X_test, y_test))
    pdart_mse.append(test_pdart(X_train, y_train, X_test, y_test))

print('mse skgbr:{}'.format(np.mean(skgbr_mse)))
print('mse skrf:{}'.format(np.mean(skrf_mse)))
print('mse dart:{}'.format(np.mean(dart_mse)))
print('mse pdart:{}'.format(np.mean(pdart_mse)))

print('end experiment :{}'.format(time.ctime()))