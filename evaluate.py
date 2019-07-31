from nltk.tag.hmm import *
import codecs
import numpy as np
from sklearn.metrics import confusion_matrix
import metrics
import hmm
#import memm
#import crf



def evaluate():
    words = list()
    hmm_predict_words = list()
    memm_predict_words = list()
    crf_predict_words = list()
    voweled_words = list()
    syllable_words = list()
    romanized_words = list()


    with codecs.open('data/HaaretzOrnan_annotated_test.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if line.startswith(u'#') or len(line)==0:
                continue
            split_line = line.split(u' ')
            words.append(split_line[2])
            w = line.split(u' ')[3]
            voweled_words.append(w.replace(u'-', u''))
            syllable_words.append(line.split(u' ')[3])
            romanized_words.append(line.split(u' ')[4])

    hmm_predicted_words = hmm.Predict(words,2)
    print(hmm_predicted_words)

    '''print('----------------------------------------------')
    print("HMM Predict")
    print('----------------------------------------------')
    precision, recall = metrics.MicroAvg(conf_mat)
    f1 = metrics.Fscore(precision, recall, 1)
    print('MicroAvg:', precision, recall, f1)
    precision, recall = metrics.MacroAvg(conf_mat)
    f1 = metrics.Fscore(recall, precision, 1)
    print('MacroAvg:', precision, recall, f1)
    print('AvgAcc:', metrics.AvgAcc(conf_mat))
    conf_mat = metrics.NormalizeConfusion(conf_mat)
    print('ConfMat:\n', np.array_str(conf_mat, max_line_width=300, precision=4))


    memm_predict_words = memm.Predict(words)

    crf_predict_words = crf.Predict(words)
'''


