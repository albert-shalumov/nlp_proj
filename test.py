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

stage_names = ['', 'Vowels', 'Syllabification', 'Romanization']

def PrintConfMat(conf_mat):
    precision, recall = metrics.MicroAvg(conf_mat)
    f1 = metrics.Fscore(precision, recall, 1)
    print('MicroAvg:', precision, recall, f1)
    precision, recall = metrics.MacroAvg(conf_mat)
    f1 = metrics.Fscore(recall, precision, 1)
    print('MacroAvg:', precision, recall, f1)
    print('AvgAcc:', metrics.AvgAcc(conf_mat))
    conf_mat = metrics.NormalizeConfusion(conf_mat)
    print('ConfMat:\n', np.array_str(conf_mat, max_line_width=300, precision=3))

def LoadTestData(file='data/HaaretzOrnan_annotated_test.txt'):
    words, vow_words, syll_words, rom_words = [], [], [], []
    with codecs.open(file, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if line.startswith(u'#') or len(line) == 0:
                continue
            split_line = line.split(u' ')
            words.append(split_line[2])
            vow_words.append(split_line[3].replace(u'-', u''))
            syll_words.append(split_line[3])
            rom_words.append(split_line[4])
    return words, vow_words, syll_words, rom_words

def CalcConfMatrix(pred, gold):
    vow = list(u'euioa*')
    vow_idx = {x: i for i, x in enumerate(vow)}
    conf_mat = np.zeros((len(vow), len(vow)))
    for j in range(1, len(pred), 2):
        conf_mat[vow_idx[pred[j]], vow_idx[gold[j]]] += 1
    return conf_mat

def TestModel(model, data):
    conf_mat = None
    dist = [None]
    pred_stage = [None]
    pred_stage.append(model.predict(data[0]))  # predict test data
    pred_stage.append([syllabification(w) for w in pred_stage[1]])  # calculate syllabification
    pred_stage.append([romanize(w) for w in pred_stage[2]])  # calculate romanization

    # Calculate confusuion matrix
    conf_mat = np.zeros((6,6))
    for i, w in enumerate(pred_stage[1]):
        conf_mat += CalcConfMatrix(w, data[1][i])

    for stage in range(1,4):
        tmp_dist = [EditDistance(w, data[stage][i]) for i, w in enumerate(pred_stage[stage])]
        dist.append((sum(tmp_dist)/len(tmp_dist), statistics.median(tmp_dist), min(tmp_dist), max(tmp_dist))) # avg,med.min,max

    return conf_mat, dist

def test():
    states = [u'a', u'e', u'u', u'i', u'o', u'*']
    state_idx = {x: i for i, x in enumerate(states)}

    (words, voweled_words, syllable_words, romanized_words) = LoadTestData()

    untrained_models = [(HMM(2), 'HMM, ngram=2'), (MEMM(), 'MEMM'), (CRF_WORD(), 'CRF (word level)')]
    for model,name in untrained_models:
        trained_model = model.prep_data().shuffle(None).split(0).train()
        conf_mat, dist = TestModel(trained_model, (words, voweled_words, syllable_words, romanized_words))
        print('\n\n')
        print(name)
        print('='*80)
        print('Vowel metrics:')
        print('-'*50)
        PrintConfMat(conf_mat)
        print('-'*50)
        print('Edit distance:')
        print('-'*50)
        for stage in range(1,4):
            print('Stage = {}:'.format(stage_names[stage]))
            print('   Average = {}\n   Median = {}\n   Min = {}\n   Max = {}'.format(dist[stage][0],dist[stage][1],dist[stage][2],dist[stage][3]))


if __name__ == "__main__":
    test()
