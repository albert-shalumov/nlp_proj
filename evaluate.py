from nltk.tag.hmm import *
import codecs
import statistics
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
    romanized_words = list()
    states = [u'a', u'e', u'u', u'i', u'o', u'*']
    state_idx = {x: i for i, x in enumerate(states)}

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

    p_syllable_words = list()
    p_romanized_words = list()
    H = HMM(2)
    H.prep_data().shuffle(None).split(0).train()
    hmm_predict_words = H.predict(words)

    for w in hmm_predict_words:
        p_syllable_words.append(syllabification(w))

    for w in p_syllable_words:
        p_romanized_words.append(romanize(w))

    conf_mat_voweled = np.zeros((len(states), len(states)))

    for i, w in enumerate(hmm_predict_words):
        for j in range(1, len(w), 2):
            conf_mat_voweled[state_idx[w[j]], state_idx[voweled_words[i][j]]] += 1

    #PrintConfMat(conf_mat_voweled, 'HMM - voweled')
    dError = list()
    for i, w in enumerate(hmm_predict_words):
        dError.append(EditDistance(w, words[i]))
    print('HMM Edit Distance:')
    print('Avg: ', sum(dError) / len(dError))
    print('Med: ', statistics.median(dError))
    print('Min: ', min(dError))
    print('Max: ', max(dError))

    dError = list()
    for i, w in enumerate(p_syllable_words):
        dError.append(EditDistance(w, syllable_words[i]))
    print('HMM Syllable Edit Distance:')
    print('Avg: ', sum(dError)/len(dError))
    print('Med: ', statistics.median(dError))
    print('Min: ',min(dError))
    print('Max: ',max(dError))

    dError = list()
    for i, w in enumerate(p_romanized_words):
        dError.append(EditDistance(w, romanized_words[i]))
    print('HMM Romanize Edit Distance:')
    print('Avg: ', sum(dError) / len(dError))
    print('Med: ', statistics.median(dError))
    print('Min: ', min(dError))
    print('Max: ', max(dError))


    p_syllable_words = list()
    p_romanized_words = list()

    M = MEMM()
    M.prep_data().shuffle(None).split(0).train()
    memm_predict_words = M.predict(words)

    for w in memm_predict_words:
        p_syllable_words.append(syllabification(w))

    for w in p_syllable_words:
        p_romanized_words.append(romanize(w))

    conf_mat_voweled = np.zeros((len(states), len(states)))

    for i, w in enumerate(memm_predict_words):
        for j in range(1, len(w), 2):
            conf_mat_voweled[state_idx[w[j]], state_idx[voweled_words[i][j]]] += 1

    #PrintConfMat(conf_mat_voweled, 'MEMM - voweled')
    dError = list()
    for i, w in enumerate(memm_predict_words):
        dError.append(EditDistance(w, words[i]))
    print('MEMM Edit Distance:')
    print('Avg: ', sum(dError) / len(dError))
    print('Med: ', statistics.median(dError))
    print('Min: ', min(dError))
    print('Max: ', max(dError))

    dError = list()
    for i, w in enumerate(p_syllable_words):
        dError.append(EditDistance(w, syllable_words[i]))
    print('MEMM Syllable Edit Distance:')
    print('Avg: ', sum(dError) / len(dError))
    print('Med: ', statistics.median(dError))
    print('Min: ', min(dError))
    print('Max: ', max(dError))

    dError = list()
    for i, w in enumerate(p_romanized_words):
        dError.append(EditDistance(w, romanized_words[i]))
    print('MEMM Romanize Edit Distance:')
    print('Avg: ', sum(dError) / len(dError))
    print('Med: ', statistics.median(dError))
    print('Min: ', min(dError))
    print('Max: ', max(dError))


    p_syllable_words = list()
    p_romanized_words = list()
    C = CRF_WORD()
    C.prep_data().shuffle(None).split(0).train()
    crf_predict_words = C.predict(words)

    for w in crf_predict_words:
        p_syllable_words.append(syllabification(w))

    for w in p_syllable_words:
        p_romanized_words.append(romanize(w))

    conf_mat_voweled = np.zeros((len(states), len(states)))


    for i, w in enumerate(crf_predict_words):
        for j in range(1, len(w), 2):
            conf_mat_voweled[state_idx[w[j]], state_idx[voweled_words[i][j]]] += 1

    #PrintConfMat(conf_mat_voweled, 'CRF - voweled')
    dError = list()
    for i, w in enumerate(crf_predict_words):
        dError.append(EditDistance(w, words[i]))
    print('CRF Edit Distance:')
    print('Avg: ', sum(dError) / len(dError))
    print('Med: ', statistics.median(dError))
    print('Min: ', min(dError))
    print('Max: ', max(dError))

    dError = list()
    for i, w in enumerate(p_syllable_words):
        dError.append(EditDistance(w, syllable_words[i]))
    print('CRF Syllable Edit Distance:')
    print('Avg: ', sum(dError) / len(dError))
    print('Med: ', statistics.median(dError))
    print('Min: ', min(dError))
    print('Max: ', max(dError))

    dError = list()
    for i, w in enumerate(p_romanized_words):
        dError.append(EditDistance(w, romanized_words[i]))
    print('CRF Romanize Edit Distance:')
    print('Avg: ', sum(dError) / len(dError))
    print('Med: ', statistics.median(dError))
    print('Min: ', min(dError))
    print('Max: ', max(dError))

if __name__ == "__main__":
    evaluate()
