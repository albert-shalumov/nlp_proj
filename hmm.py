from nltk.tag.hmm import *
import codecs
import numpy as np
import metrics
from copy import deepcopy

class HMM:
    def __init__(self, ngram):
        self.ngram = ngram

    def prep_data(self, file='data/HaaretzOrnan_annotated.txt'):
        ngrams = set()
        self.data = []
        with codecs.open(file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                if line.startswith(u'#') or len(line) == 0:
                    continue
                w = line.split(u' ')[3]
                w = w.replace(u'-', u'')
                word_ngrams = HMM._extract_ngrams(w[::2], self.ngram)
                self.data.append(list(zip(word_ngrams, list(w[1::2]))))
                ngrams.update(word_ngrams)
        self.ngrams = list(ngrams)
        return self

    def shuffle(self, seed=None):
        inds = np.arange(len(self.data))
        np.random.seed(seed)
        np.random.shuffle(inds)
        self.data=[self.data[i] for i in inds]
        return self

    def split(self, valid_ratio=0.1):
        # Split to train and validation based on ratio.
        # If ratio is 0 use all data for training
        num_train = int(len(self.data)*(1-valid_ratio))
        self.train_set = self.data[:num_train]
        self.valid_set = None if valid_ratio==0 else self.data[num_train:]
        return self

    def train(self):
        hmm_trainer = HiddenMarkovModelTrainer(states = HMM.VOWELS, symbols = self.ngrams)
        self.model = hmm_trainer.train(labeled_sequences=self.train_set)
        return self

    def eval(self):
        conf_mat = np.zeros((len(HMM.VOWELS), len(HMM.VOWELS)))
        valid_word_cons = [[x[0] for x in w] for w in self.valid_set]
        valid_word_vowel = [[x[1] for x in w] for w in self.valid_set]
        predicted = self.model.tag_sents(valid_word_cons)
        predicted = [[x[1] for x in w] for w in predicted]
        for w_ind in range(len(predicted)):
            for vow_ind, pred_vow in enumerate(predicted[w_ind]):
                conf_mat[self.VOWELS_IDX[pred_vow], self.VOWELS_IDX[valid_word_vowel[w_ind][vow_ind]]] += 1
        return conf_mat

    def predict(self, pred_set):
        result = []
        for i, w_cons in enumerate(pred_set):
            predicted = self.model.best_path(HMM._extract_ngrams(w_cons, self.ngram))
            result.append(''.join(x+y for x, y in zip(w_cons, predicted)))
        return result

    VOWELS = [u'a',u'e',u'u',u'i',u'o',u'*']
    VOWELS_IDX = {x:i for i,x in enumerate(VOWELS)}
    @staticmethod
    def _extract_ngrams(word, ngram):
        start_symb=u'-'
        start_w = [start_symb]*(ngram-1)
        w = start_w+list(word)
        l = []
        for i in range(len(w)-ngram+1):
            l.append(''.join(w[i:i+ngram]))
        return l


if __name__ == '__main__':
    for ngram in range(1,5,1):
        if 'conf_mat' in locals():
            del conf_mat
        for iters in range(5):
            hmm = HMM(ngram)
            if 'conf_mat' in locals():
                conf_mat += hmm.prep_data().shuffle(None).split(0.1).train().eval()
            else:
                conf_mat  = hmm.prep_data().shuffle(None).split(0.1).train().eval()
        print("|Ngram| = {}: ".format(ngram))
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
