import numpy as np

EPS = 1e-6

def MicroAvg(conf_mat):
    l = conf_mat.shape[0]
    mat_sum = np.sum(conf_mat)
    num,prec_denom, rec_denom = 0,0,0
    for i in range(l):
        tp = conf_mat[i,i]
        num += tp
        fp = np.sum(conf_mat[i,:])-tp
        fn = np.sum(conf_mat[:,i])-tp
        prec_denom += tp+fp
        rec_denom += tp+fn

    return num/(prec_denom+EPS), num/(rec_denom+EPS)

def MacroAvg(conf_mat):
    l = conf_mat.shape[0]
    mat_sum = np.sum(conf_mat)
    prec, rec = 0,0
    for i in range(l):
        tp = conf_mat[i,i]
        fp = np.sum(conf_mat[i,:])-tp
        fn = np.sum(conf_mat[:,i])-tp

        prec += tp/(tp+fp+EPS)
        rec += tp/(tp+fn+EPS)

    return prec/l, rec/l

def AvgAcc(conf_mat):
    l = conf_mat.shape[0]
    mat_sum = np.sum(conf_mat)
    s=0
    for i in range(l):
        tp = conf_mat[i,i]
        tn = mat_sum-(np.sum(conf_mat[i,:])+np.sum(conf_mat[:,i])-tp)
        s += (tp+tn)/(mat_sum+EPS)
    return s/l


def Fscore(recall,precision,beta=1):
    return (1+beta**2)*precision*recall/(precision*beta**2+recall)

def NormalizeConfusion(conf_mat):
    gt_sum = np.sum(conf_mat,axis=0)+EPS
    return conf_mat/gt_sum
