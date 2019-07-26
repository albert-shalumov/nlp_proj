from nltk.tag.crf import CRFTagger
import codecs
import numpy as np
import metrics

states = [u'a',u'e',u'u',u'i',u'o',u'*']
state_idx = {x:i for i,x in enumerate(states)}

def ExtractNgrams(word, n=1):
    start_symb=u'-'
    start_w = [start_symb]*(n-1)
    w = start_w+list(word)
    l = []
    for i in range(len(w)-n+1):
        l.append(''.join(w[i:i+n]))
    return l

def ExtractFeatures(sentence, i):
    return []

def TestCRF(ftr_set, iters=5):
    conf_mat = np.zeros((len(states), len(states)))

    return conf_mat


print("CRF: ")
conf_mat = TestCRF(0, 50)
precision, recall = metrics.MicroAvg(conf_mat)
f1 = metrics.Fscore(precision, recall, 1)
print('MicroAvg:',precision,recall,f1)
precision, recall = metrics.MacroAvg(conf_mat)
f1 = metrics.Fscore(recall, precision, 1)
print('MacroAvg:', precision, recall, f1)
print('AvgAcc:',metrics.AvgAcc(conf_mat))
conf_mat = metrics.NormalizeConfusion(conf_mat)
print('ConfMat:\n', np.array_str(conf_mat, max_line_width=300, precision=4))
print('----------------------------------------------')


