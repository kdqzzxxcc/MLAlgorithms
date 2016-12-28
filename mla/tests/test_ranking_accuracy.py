# coding=utf-8
__author__ = 'kdq'
from tqdm import tqdm
import numpy as np
from mla.ensemble import lambdamart, gbm
from mla.ensemble.pdart import pDARTRanker

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

def check_data(x):
    l = len(x)
    x = np.array(x)
    x = x.reshape([l, 1])
    return x
X_train, y_train, qids_train, X_test, y_test, qids_test = [], [], [], [], [], []
load_mslr_10k('/home/ilab/datasets/MSLR-WEB10K/Fold1/')
qids = pDARTRanker._setup_qids(qids_test)
# reform data
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
qids_train = check_data(qids_train)
qids_test = check_data(qids_test)
y_train = check_data(y_train)
y_test = np.array(y_test, dtype=np.float32)
#

print X_test.shape, X_train.shape
train_data = np.hstack([y_train, qids_train, X_train])
model = lambdamart.LambdaMART(training_data=train_data, n_estimators=1, learning_rate=0.4, max_depth=5)
model.fit()
predictions = model.predict_all(X_test)
ndcg = gbm.PairwiseLoss.error(y_test, predictions, qids, k=3)
print ndcg