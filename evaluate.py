from nltk.tag.hmm import *
import codecs
import numpy as np
from sklearn.metrics import confusion_matrix
import metrics
from metrics import EditDistance
from hmm import HMM
from memm import MEMM
from crf_word import CRF as CRF_WORD
from post_proc.syllabification import syllabification
from post_proc.post_processing import romanize


def PrintConfMat(conf_mat, model_name):
    print('----------------------------------------------')
    print(model_name)
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


def evaluate():
    words = list()
    voweled_words = list()
    syllable_words = list()
    p_syllable_words = list()
    romanized_words = list()
    p_romanized_words = list()

    with codecs.open('data/HaaretzOrnan_annotated_test.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if line.startswith(u'#') or len(line) == 0:
                continue
            split_line = line.split(u' ')
            words.append(split_line[2])
            w = split_line[3]
            voweled_words.append(w.replace(u'-', u''))
            syllable_words.append(split_line[3])
            romanized_words.append(split_line[4])

    H = HMM(2)
    H.prep_data().shuffle(None).split(0).train()
    hmm_predict_words = H.predict(words)

    for w in hmm_predict_words:
        p_syllable_words.append(syllabification(w))

    for w in p_syllable_words:
        p_romanized_words.append(romanize(w))

    states = [u'a', u'e', u'u', u'i', u'o', u'*']
    state_idx = {x: i for i, x in enumerate(states)}

    conf_mat_voweled = np.zeros((len(states), len(states)))
    conf_mat_romanize = np.zeros((len(states), len(states)))

    for i, w in enumerate(hmm_predict_words):
        for j in range(1, len(w), 2):
            conf_mat_voweled[state_idx[w[j]], state_idx[voweled_words[i][j]]] += 1

    PrintConfMat(conf_mat_voweled, 'HMM - voweled')


    total = 0
    for i, w in enumerate(p_syllable_words):
        total += EditDistance(w, syllable_words[i])
    print('HMM SYLLABLE:', total / len(p_syllable_words))

    total = 0
    for i, w in enumerate(p_romanized_words):
        total += EditDistance(w, romanized_words[i])
    print('HMM ROMANIZE:', total / len(p_romanized_words))

    '''syl_states = [u'a', u'e', u'u', u'i', u'o', u'*', u'-', u'a-', u'e-', u'u-', u'i-', u'o-', u'*-']
    syl_state_idx = {x: i for i, x in enumerate(syl_states)}
    conf_mat_syllable = np.zeros((len(syl_states), len(syl_states)))
    for i,w in enumerate(p_syllable_words):
        for j in range(1,len(w),2):
            if (j+1) < len(w) and w[j+1] == '-':
                if (j+1) < len(syllable_words[i]) and syllable_words[i][j+1] == '-':
                    conf_mat_syllable[syl_state_idx[w[j:j+2]],syl_state_idx[syllable_words[i][j:j+2]]] += 1
                else:
                    conf_mat_syllable[syl_state_idx[w[j:j+2]],syl_state_idx[syllable_words[i][j]]] += 1
            else:
                if (j+1) < len(syllable_words[i]) and syllable_words[i][j+1] == '-':
                    conf_mat_syllable[syl_state_idx[w[j]],syl_state_idx[syllable_words[i][j:j+2]]] += 1
                else:
                    conf_mat_syllable[syl_state_idx[w[j]],syl_state_idx[syllable_words[i][j]]] += 1

    PrintConfMat(conf_mat_syllable, 'HMM - syllable')

    for i,w in enumerate(p_romanized_words):
        for j in range(len(w)):
            if j%2 == 1:
                conf_mat_romanize[state_idx[w[j]],state_idx[romanized_words[i][j]]] += 1
    PrintConfMat(conf_mat_romanize, 'HMM - romanize')
    '''

    p_syllable_words = list()
    p_romanized_words = list()

    M = MEMM()
    M.prep_data().shuffle(None).split(0).train()
    memm_predict_words = M.predict(words)

    for w in memm_predict_words:
        p_syllable_words.append(syllabification(w))

    for w in p_syllable_words:
        p_romanized_words.append(romanize(w))

    states = [u'a', u'e', u'u', u'i', u'o', u'*']
    state_idx = {x: i for i, x in enumerate(states)}

    conf_mat_voweled = np.zeros((len(states), len(states)))
    conf_mat_syllable = np.zeros((len(states), len(states)))
    conf_mat_romanize = np.zeros((len(states), len(states)))

    for i, w in enumerate(memm_predict_words):
        for j in range(1, len(w), 2):
            conf_mat_voweled[state_idx[w[j]], state_idx[voweled_words[i][j]]] += 1

    PrintConfMat(conf_mat_voweled, 'MEMM - voweled')

    total = 0
    for i, w in enumerate(p_syllable_words):
        total += EditDistance(w, syllable_words[i])
    print('MEMM SYLABLE:',total / len(p_syllable_words))

    total = 0
    for i, w in enumerate(p_romanized_words):
        total += EditDistance(w, romanized_words[i])
    print('MEMM ROMANIZE:', total / len(p_romanized_words))

    '''for i,w in enumerate(p_syllable_words):
        for j in range(len(w)):
            if j%2 == 1:
                conf_mat_syllable[state_idx[w[j]],state_idx[syllable_words[i][j]]] += 1
                
    PrintConfMat(conf_mat_syllable, 'MEMM - syllable')
    
    for i,w in enumerate(p_romanized_words):
        for j in range(len(w)):
            if j%2 == 1:
                conf_mat_romanize[state_idx[w[j]],state_idx[romanized_words[i][j]]] += 1
    PrintConfMat(conf_mat_romanize, 'MEMM - romanize')
    '''

    p_syllable_words = list()
    p_romanized_words = list()
    C = CRF_WORD()
    C.prep_data().shuffle(None).split(0).train()
    crf_predict_words = C.predict(words)

    for w in crf_predict_words:
        p_syllable_words.append(syllabification(w))

    for w in p_syllable_words:
        p_romanized_words.append(romanize(w))

    states = [u'a', u'e', u'u', u'i', u'o', u'*']
    state_idx = {x: i for i, x in enumerate(states)}

    conf_mat_voweled = np.zeros((len(states), len(states)))


    for i, w in enumerate(crf_predict_words):
        for j in range(1, len(w), 2):
            conf_mat_voweled[state_idx[w[j]], state_idx[voweled_words[i][j]]] += 1

    PrintConfMat(conf_mat_voweled, 'CRF - voweled')

    total = 0
    for i, w in enumerate(p_syllable_words):
        total += EditDistance(w, syllable_words[i])
    print('CRF SYLLABLE:', total / len(p_syllable_words))

    total = 0
    for i, w in enumerate(p_romanized_words):
        total += EditDistance(w, romanized_words[i])
    print('CRF ROMANIZE:', total / len(p_romanized_words))

if __name__ == "__main__":
    evaluate()
