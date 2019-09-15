from nltk.tag.hmm import *
from nltk.probability import *
from itertools import product
import codecs
import numpy as np
import metrics
from copy import deepcopy
import sys
class HMM:
    def __init__(self, config):
        self.ngram = config['ngram']
        if config['est'] not in HMM.ESTIMATORS:
            raise Exception('Unknown estimator {}. See HMM.ESTIMATORS for supported ones.'.format(config['est']))
        self.est = config['est']
        if self.est == 'add-delta':
            self.delta = config['delta']


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

    def train(self, load_model=None):
        est = self._gen_estimator()
        hmm_trainer = HiddenMarkovModelTrainer(states = HMM.VOWELS, symbols = self.ngrams)
        self.model = hmm_trainer.train_supervised(self.train_set, estimator=est)
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
        for sep_sent in pred_set:
            pred_sent = []
            for i, w_cons in enumerate(sep_sent):
                predicted = self.model.best_path(HMM._extract_ngrams(w_cons, self.ngram))
                pred_sent.append(''.join(x+y for x, y in zip(w_cons, predicted)))
            result.append(pred_sent)
        return result

    VOWELS = [u'a',u'e',u'u',u'i',u'o',u'*']
    VOWELS_IDX = {x:i for i,x in enumerate(VOWELS)}
    ESTIMATORS = ['mle', 'laplace', 'add-delta', 'good-turing']
    @staticmethod
    def _extract_ngrams(word, ngram):
        start_symb=u'-'
        start_w = [start_symb]*(ngram-1)
        w = start_w+list(word)
        l = []
        for i in range(len(w)-ngram+1):
            l.append(''.join(w[i:i+ngram]))
        return l

    def _gen_estimator(self):
        if self.est == 'mle':
            est = lambda fd, bins: MLEProbDist(fd)
        elif self.est == 'laplace':
            est = lambda fd, bins: LaplaceProbDist(fd)
        elif self.est == 'good-turing':
            est = lambda fd, bins: SimpleGoodTuringProbDist(fd)
        elif self.est == 'add-delta':
            est = lambda fd, bins: LidstoneProbDist(fd, gamma=self.delta)
        else:
            print('Unknown smoothing "{}". Reverting to MLE.'.format(self.est))
            est = lambda fd, bins: MLEProbDist(fd)
        return est

def search_hparams():
    verbose=False
    ngrams = [{'ngram':x} for x in range(1,8,1)]
    smooth = [{'est':'mle'}, {'est':'laplace'}, {'est':'good-turing'}]
    smooth.extend([{'est':'add-delta', 'delta':x/10} for x in range(1,10,1)])

    with open('hmm_res.csv', 'w') as f:
        for config in itertools.product(ngrams, smooth):
            config = {**config[0], **config[1]}
            if 'conf_mat' in locals():
                del conf_mat
            for iters in range(7):
                hmm = HMM(config)
                if 'conf_mat' in locals():
                    conf_mat += hmm.prep_data().shuffle(None).split(0.1).train().eval()
                else:
                    conf_mat  = hmm.prep_data().shuffle(None).split(0.1).train().eval()
            res_str = '{};'.format(config)
            print("Configuration = {}: ".format(config))
            precision, recall = metrics.MicroAvg(conf_mat)
            f1 = metrics.Fscore(precision, recall, 1)
            res_str += '{};'.format(f1)
            print('MicroAvg:',precision,recall,f1)
            precision, recall = metrics.MacroAvg(conf_mat)
            f1 = metrics.Fscore(recall, precision, 1)
            res_str += '{};'.format(f1)
            print('MacroAvg:', precision, recall, f1)
            acc = metrics.AvgAcc(conf_mat)
            res_str += '{};'.format(acc)
            print('AvgAcc:', acc)
            f.write(res_str+'\n')
            conf_mat = metrics.NormalizeConfusion(conf_mat)
            if verbose:
                print('ConfMat:\n', np.array_str(conf_mat, max_line_width=300, precision=4))
                print('----------------------------------------------')

def check_seeds():
    config = {'ngram': 3, 'est': 'add-delta', 'delta': 0.3}
    print("seed, accuracy")
    for seed in range(11):
        if 'conf_mat' in locals():
            del conf_mat
        for iters in range(7):
            hmm = HMM(config)
            if 'conf_mat' in locals():
                conf_mat += hmm.prep_data().shuffle(seed).split(0.1).train().eval()
            else:
                conf_mat = hmm.prep_data().shuffle(seed).split(0.1).train().eval()
        acc = metrics.AvgAcc(conf_mat)
        print(seed, acc)

def print_usage():
    print("Usage:")
    print("hmm.py [search/seeds]")
    print("search - searches for best configuration")
    print("seeds - checks various seeds")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print_usage()
    elif sys.argv[1] == 'search':
        search_hparams()
    elif sys.argv[1] == 'seeds':
        check_seeds()
    else:
        print_usage()
