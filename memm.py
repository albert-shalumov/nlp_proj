import codecs
import numpy as np
import metrics
from sklearn.linear_model import LogisticRegression
from functools import reduce as reduce
from itertools import combinations

'''
Class for MEMM method.
Most functions return self, therefore calls can be chained: memm.shuffle().split().train() etc.
'''
class MEMM:
    def __init__(self, config):
        self.ftrs = config['ftrs']
        for ftr in self.ftrs:
            if ftr not in MEMM.CONFIG:
                raise Exception('Unknown feature {}. See MEMM.CONFIG for supported ones.'.format(MEMM.CONFIG))

    def prep_data(self, file='data/HaaretzOrnan_annotated.txt'):
        # Load words, discarding separation to syllables
        words = []
        with codecs.open(file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                if line.startswith(u'#') or len(line) == 0:
                    continue
                w = line.split(u' ')[3]
                w = w.replace(u'-', u'')
                words.append(w)
        # Extract features, allocate arrays
        self.num_cons_chars = reduce((lambda x, y: MEMM._len(x)+MEMM._len(y)), words)//2
        self.X = np.zeros((self.num_cons_chars, MEMM.MAX_FTR_LEN))
        self.Y = np.zeros(self.num_cons_chars)

        sample=0
        for w in words:
            for i in range(len(w)//2):
                MEMM._word_ftr(w[::2], i, self.X[sample, :], self.ftrs)  # extract features from words without vowels
                self.Y[sample] = MEMM.VOWELS_IDX[w[1::2][i]]  # set label to vowel
                sample += 1
        return self

    def shuffle(self, seed=None):
        # Shuffle based on seed
        inds = np.arange(self.X.shape[0])
        np.random.seed(seed)
        np.random.shuffle(inds)
        self.X = self.X[inds,:]
        self.Y = self.Y[inds]
        return self

    def split(self, valid_ratio=0.1):
        # Split to train and validation based on ratio.
        # If ratio is 0 use all data for training
        self.train_X = self.X[:int(self.num_cons_chars*(1-valid_ratio)), :]
        self.train_Y = self.Y[:int(self.num_cons_chars*(1-valid_ratio))]
        if valid_ratio==0:
            self.valid_X = None
            self.valid_Y = None
        else:
            self.valid_X = self.X[self.train_X.shape[0]:,:]
            self.valid_Y = self.Y[self.train_X.shape[0]:]
        return self

    def train(self, load_model=None):
        # Train model using logistic regression.
        # If number of features becomes too big - use PCA or similar to reduce
        self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000)
        self.model.fit(self.train_X, self.train_Y)
        return self

    def eval(self):
        conf_mat = np.zeros((len(MEMM.VOWELS), len(MEMM.VOWELS)))
        if self.valid_X is None:
            print("Empty validation set")
            return conf_mat

        # Fill confusion matrix
        predicted = self.model.predict(self.valid_X)
        for i in range(predicted.shape[0]):
            conf_mat[int(predicted[i]), int(self.valid_Y[i])] += 1

        return conf_mat

    def predict(self, pred_set):
        result = []
        for sep_sent in pred_set:
            pred_sent = []
            for i, w_cons in enumerate(sep_sent):
                X = np.zeros((len(w_cons), MEMM.MAX_FTR_LEN))
                for j in range(len(w_cons)):
                    MEMM._word_ftr(w_cons, j, X[j, :], self.ftrs)  # extract features from words
                vowels = [MEMM.VOWELS[int(x)] for x in self.model.predict(X)]
                pred_sent.append(''.join(x+y for x, y in zip(w_cons, vowels)))
            result.append(pred_sent)
        return result


    @staticmethod
    def _len(x):
        return len(x) if isinstance(x, str) else int(x)

    MAX_FTR_LEN = 210
    @staticmethod
    def _word_ftr(word, i, ftr_vec, ftr):
        idx = 0

        # is it the first letter
        if 'IS_FIRST' in ftr:
            ftr_vec[idx] = 1 if i == 0 else 0
        idx += 1

        # is it the last letter
        if 'IS_LAST' in ftr:
            ftr_vec[idx] = 1 if i == (len(word)-1) else 0
        idx += 1

        # current letter
        if 'VAL' in ftr:
            ftr_vec[idx+MEMM.ARNON_CHARS_IDX[word[i]]] = 1
        idx += len(MEMM.ARNON_CHARS)

        # prev letter
        if 'PRV_VAL' in ftr:
            if i > 0:
                ftr_vec[idx+MEMM.ARNON_CHARS_IDX[word[i-1]]] = 1
        idx += len(MEMM.ARNON_CHARS)

        # next letter
        if 'NXT_VAL' in ftr:
            if i < (len(word)-1):
                ftr_vec[idx+MEMM.ARNON_CHARS_IDX[word[i+1]]] = 1
        idx += len(MEMM.ARNON_CHARS)

        # first letter
        if 'FRST_VAL' in ftr:
            ftr_vec[idx+MEMM.ARNON_CHARS_IDX[word[0]]] = 1
        idx += len(MEMM.ARNON_CHARS)

        # last letter
        if 'LST_VAL' in ftr:
            ftr_vec[idx+MEMM.ARNON_CHARS_IDX[word[-1]]] = 1
        idx += len(MEMM.ARNON_CHARS)

        # second letter
        if 'SCND_VAL' in ftr:
            ftr_vec[idx + MEMM.ARNON_CHARS_IDX[word[1]]] = 1
        idx += len(MEMM.ARNON_CHARS)

        # second last letter
        if 'SCND_LST_VAL' in ftr:
            ftr_vec[idx + MEMM.ARNON_CHARS_IDX[word[-2]]] = 1
        idx += len(MEMM.ARNON_CHARS)

        # word length
        if 'LEN' in ftr:
            l = min(len(word), 8)
            ftr_vec[idx+l] = 1
        idx += 9

        if 'IDX' in ftr:
            l = min(i, 8)
            ftr_vec[idx+l] = 1
        idx += 9

        assert idx <= MEMM.MAX_FTR_LEN

    VOWELS = [u'a',u'e',u'u',u'i',u'o',u'*']
    VOWELS_IDX = {x:i for i,x in enumerate(VOWELS)}
    ARNON_CHARS = [u'\u02c0', u'b',u'g', u'd', u'h', u'w', u'z', u'\u1e25', u'\u1e6d', u'y',
            u'k', u'k', u'l', u'm', u'm', u'n', u'n', u's', u'\u02c1', u'p', u'p', u'\u00e7',
            u'\u00e7', u'q', u'r', u'\u0161', u't']
    ARNON_CHARS_IDX = {x:i for i,x in enumerate(ARNON_CHARS)}
    CONFIG = ['IS_FIRST', 'IS_LAST', 'IDX', 'VAL', 'PRV_VAL', 'NXT_VAL', 'FRST_VAL', 'LST_VAL', 'SCND_VAL', 'SCND_LST_VAL', 'LEN']

if __name__ == '__main__':
    verbose = False
    with open('memm_res.csv','w') as f:
        for num_ftrs in range(len(MEMM.CONFIG)):
            num_ftrs += 1
            for ftrs in combinations(MEMM.CONFIG, num_ftrs):
                config = {'ftrs':ftrs}
                if 'conf_mat' in locals():
                    del conf_mat
                for i in range(5):
                    memm = MEMM(config)
                    if 'conf_mat' in locals():
                        conf_mat += memm.prep_data().shuffle(0).split(0.1).train().eval()
                    else:
                        conf_mat = memm.prep_data().shuffle(0).split(0.1).train().eval()
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
                print('AvgAcc:',acc)
                f.write(res_str+'\n')
                conf_mat = metrics.NormalizeConfusion(conf_mat)
                if verbose:
                    print('ConfMat:\n', np.array_str(conf_mat, max_line_width=300, precision=4))
                    print('----------------------------------------------')
