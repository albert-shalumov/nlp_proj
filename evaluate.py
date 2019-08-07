from nltk.tag.hmm import *
import codecs
import numpy as np
from sklearn.metrics import confusion_matrix
import metrics
from memm import MEMM
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

    M = MEMM()
    M.prep_data().shuffle(None).split().train()
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

    print(romanized_words)
    print(p_romanized_words)

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


if __name__ == "__main__":
    evaluate()
