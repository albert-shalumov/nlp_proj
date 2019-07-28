import numpy as np

EPS = 1e-9
def _tp(conf_mat, i):
    return conf_mat[i,i]

def _fp(conf_mat, i):
    return np.sum(conf_mat[i,:])-conf_mat[i,i]

def _tn(conf_mat, i):
    return np.sum(conf_mat)-_tp(conf_mat,i)-_fp(conf_mat,i)-_fn(conf_mat,i)

def _fn(conf_mat, i):
    return np.sum(conf_mat[:,i])-conf_mat[i,i]


def MicroAvg(conf_mat):
    l = conf_mat.shape[0]
    num,prec_denom, rec_denom = 0,0,0
    for i in range(l):
        num += _tp(conf_mat, i)
        prec_denom += _tp(conf_mat,i)+_fp(conf_mat,i)
        rec_denom += _tp(conf_mat, i)+_fn(conf_mat, i)

    return num/(prec_denom+EPS), num/(rec_denom+EPS)

def MacroAvg(conf_mat):
    l = conf_mat.shape[0]
    mat_sum = np.sum(conf_mat)
    prec, rec = 0,0
    for i in range(l):
        prec += _tp(conf_mat, i)/(_tp(conf_mat, i)+_fp(conf_mat,i)+EPS)
        rec += _tp(conf_mat, i)/(_tp(conf_mat, i)+_fn(conf_mat,i)+EPS)

    return prec/l, rec/l

def AvgAcc(conf_mat):
    l = conf_mat.shape[0]
    s=0
    for i in range(l):
        s += (_tp(conf_mat,i)+_tn(conf_mat,i))/(_tp(conf_mat,i)+_tn(conf_mat,i)+_fn(conf_mat,i)+_fp(conf_mat,i))
    return s/l

def Fscore(precision, recall, beta=1):
    return (1+beta**2)*precision*recall/(precision*beta**2+recall)

def NormalizeConfusion(conf_mat):
    gt_sum = np.sum(conf_mat,axis=0)+EPS
    return conf_mat/gt_sum
