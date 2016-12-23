# coding=utf-8
__author__ = 'kdq'
import numpy as np

def dcg(scores):
    return np.sum([(np.power(2, scores[i]) - 1) / np.log2(i + 2) for i in xrange(len(scores))])

def idcg(scores):
    return dcg(sorted(scores, reverse=True))

def dcg_k(scores, k=3):
    return np.sum([(np.power(2, scores[i]) - 1) / np.log2(i + 2) for i in xrange(len(scores[:k]))])

def idcg_k(scores, k=3):
    return dcg_k(sorted(scores, reverse=True), k)

def single_dcg(scores, i, j):
    '''
    calculate label in position i change to position j
    '''
    return ( np.power(2, scores[i]) - 1 ) / np.log2(j + 2)
    
def ndcg(y, y_pred, k=3):
    idcg_y = idcg_k(y, k)
    sorted_pred = np.argsort(y_pred)[::-1]
    sorted_pred = sorted_pred[:k]
    dcg_ = dcg_k(y[sorted_pred], k)
    return dcg_ / max(idcg_y, 1)

