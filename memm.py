from nltk.tag.crf import CRFTagger
import codecs
import numpy as np
import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition.pca import PCA
from functools import reduce as reduce
from collections import defaultdict
'''
vowels = [u'a',u'e',u'u',u'i',u'o',u'*']
vowels_idx = {x:i for i,x in enumerate(vowels)}
chars = [u'\u02c0', u'b',u'g', u'd', u'h', u'w', u'z', u'\u1e25', u'\u1e6d', u'y',
        u'k', u'k', u'l', u'm', u'm', u'n', u'n', u's', u'\u02c1', u'p', u'p', u'\u00e7',
        u'\u00e7', u'q', u'r', u'\u0161', u't']
chars_idx = {x:i for i,x in enumerate(chars)}

def Predict(words_in):
    def len_(a):
        return len(a) if isinstance(a, str) else int(a)

    def extract_ftrs_lbls(words, chars=1):
        X = np.zeros((chars, MAX_FTR_LEN))
        Y = np.zeros(chars)
        sample = 0
        for w in words:
            for i in range(len(w) // 2):
                WordMemmFtr(w[::2], i, X[sample, :])
                Y[sample] = vowels_idx[w[1::2][i]]
                sample += 1
        return X, Y

    def extract_ftrs(words, chars=1):
        X = np.zeros((chars, MAX_FTR_LEN))
        sample = 0
        for w in words:
            for i in range(len(w)):
                WordMemmFtr(w, i, X[sample, :])
                sample += 1
        return X

    words = []
    # print('Preparing data')
    with codecs.open('data/HaaretzOrnan_annotated.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if line.startswith(u'#') or len(line) == 0:
                continue
            w = line.split(u' ')[3]
            w = w.replace(u'-', u'')
            words.append(w)

    np.random.shuffle(words)
    train = words

    num_train_chars = reduce((lambda x, y: len_(x) + len_(y)), train)
    X, Y = extract_ftrs_lbls(train, num_train_chars // 2)

    # Train
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
    model.fit(X, Y)

    # Predict
    num_chars = reduce((lambda x,y:len_(x)+len_(y)), words_in)
    X = extract_ftrs(words_in, num_chars)
    pred = model.predict(X)
    res = []
    word_idx = 0
    tot_ch_idx = 0
    word_ch_idx = 0
    word_vow = ''
    while tot_ch_idx<num_chars:
        word_vow += vowels[pred[tot_ch_idx]]
        tot_ch_idx += 1
        word_ch_idx += 1
        if word_ch_idx == len(words_in[word_idx]):
            res.append(''.join(x + y for x, y in zip(words_in[word_idx], word_vow)))
            word_idx += 1
            word_ch_idx = 0
            word_vow = ''

    return res

MAX_FTR_LEN = 150
def WordMemmFtr(word, i, ftr):
    idx=0

    # is it the first letter
    ftr[idx] = 1 if i==0 else 0
    idx+=1

    # is it the last letter
    ftr[idx] = 1 if i == (len(word)-1) else 0
    idx+=1

    # current letter
    ftr[idx+chars_idx[word[i]]] = 1
    idx+=len(chars)

    # prev letter
    if i>0:
        ftr[idx+chars_idx[word[i-1]]] = 1
    idx += len(chars)

    # next letter
    if i<(len(word)-1):
        ftr[idx+chars_idx[word[i+1]]] = 1
    idx += len(chars)

    # first letter
    ftr[idx+chars_idx[word[0]]] = 1
    idx += len(chars)

    # last letter
    ftr[idx+chars_idx[word[-1]]] = 1
    idx += len(chars)

    # word length
    l = min(len(word), 5)
    ftr[idx+l] = 1
    idx += 6

    assert idx <= MAX_FTR_LEN

def WordMEMM(ftr_set, iters=5):
    def len_(a):
        return len(a) if isinstance(a, str) else int(a)

    def extract_ftrs(words, chars=1):
        X = np.zeros((chars, MAX_FTR_LEN))
        Y = np.zeros(chars)

        sample = 0
        for w in words:
            for i in range(len(w)//2):
                WordMemmFtr(w[::2], i, X[sample, :])
                Y[sample] = vowels_idx[w[1::2][i]]
                sample += 1

        return X,Y

    words = []
    #print('Preparing data')
    with codecs.open('data/HaaretzOrnan_annotated.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if line.startswith(u'#') or len(line)==0:
                continue
            w = line.split(u' ')[3]
            w = w.replace(u'-', u'')
            words.append(w)

    np.random.shuffle(words)
    valid_size = max(1, len(words)//10)
    train = words[:-valid_size]
    valid = words[-valid_size:]

    num_train_chars = reduce((lambda x,y:len_(x)+len_(y)), train)
    num_valid_chars = reduce((lambda x, y: len_(x)+len_(y)), valid)

    X,Y = extract_ftrs(train, num_train_chars//2)

    # Train
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
    model.fit(X,Y)

    # Predict
    X, Y = extract_ftrs(valid, num_valid_chars//2)

    pred = model.predict(X)
    conf_mat = np.zeros((len(vowels), len(vowels)))
    for i in range(pred.shape[0]):
        conf_mat[int(pred[i]),int(Y[i])] += 1

    return conf_mat
'''

'''
Class for MEMM method.
Most functions return self, therefore calls can be chained: memm.shuffle().split().train() etc.
'''
class MEMM:
    def __init__(self):
        pass

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
                MEMM._word_ftr(w[::2], i, self.X[sample, :])  # extract features from words without vowels
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

    def train(self):
        # Train model using logistic regression.
        # If number of features becomes too big - use PCA reduce
        self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
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
        pass

    @staticmethod
    def _len(x):
        return len(x) if isinstance(x, str) else int(x)

    MAX_FTR_LEN = 150
    @staticmethod
    def _word_ftr(word, i, ftr):
        idx = 0

        # is it the first letter
        ftr[idx] = 1 if i == 0 else 0
        idx += 1

        # is it the last letter
        ftr[idx] = 1 if i == (len(word)-1) else 0
        idx += 1

        # current letter
        ftr[idx+MEMM.ARNON_CHARS_IDX[word[i]]] = 1
        idx += len(MEMM.ARNON_CHARS)

        # prev letter
        if i > 0:
            ftr[idx+MEMM.ARNON_CHARS_IDX[word[i-1]]] = 1
        idx += len(MEMM.ARNON_CHARS)

        # next letter
        if i < (len(word)-1):
            ftr[idx+MEMM.ARNON_CHARS_IDX[word[i+1]]] = 1
        idx += len(MEMM.ARNON_CHARS)

        # first letter
        ftr[idx+MEMM.ARNON_CHARS_IDX[word[0]]] = 1
        idx += len(MEMM.ARNON_CHARS)

        # last letter
        ftr[idx+MEMM.ARNON_CHARS_IDX[word[-1]]] = 1
        idx += len(MEMM.ARNON_CHARS)

        # word length
        l = min(len(word), 5)
        ftr[idx+l] = 1
        idx += 6
        assert idx <= MEMM.MAX_FTR_LEN

    VOWELS = [u'a',u'e',u'u',u'i',u'o',u'*']
    VOWELS_IDX = {x:i for i,x in enumerate(VOWELS)}
    ARNON_CHARS = [u'\u02c0', u'b',u'g', u'd', u'h', u'w', u'z', u'\u1e25', u'\u1e6d', u'y',
            u'k', u'k', u'l', u'm', u'm', u'n', u'n', u's', u'\u02c1', u'p', u'p', u'\u00e7',
            u'\u00e7', u'q', u'r', u'\u0161', u't']
    ARNON_CHARS_IDX = {x:i for i,x in enumerate(ARNON_CHARS)}

if __name__ == '__main__':
    print("Word MEMM: ")
    for i in range(5):
        memm = MEMM()
        if 'conf_mat' in locals():
            conf_mat += memm.prep_data().shuffle(None).split(0.1).train().eval()
        else:
            conf_mat = memm.prep_data().shuffle(None).split(0.1).train().eval()
    precision, recall = metrics.MicroAvg(conf_mat)
    f1 = metrics.Fscore(precision, recall, 1)
    print('MicroAvg:',precision,recall,f1)
    precision, recall = metrics.MacroAvg(conf_mat)
    f1 = metrics.Fscore(recall, precision, 1)
    print('MacroAvg:', precision, recall, f1)
    print('AvgAcc:',metrics.AvgAcc(conf_mat))
    conf_mat = metrics.NormalizeConfusion(conf_mat)
    print('ConfMat:\n', np.array_str(conf_mat, max_line_width=300, precision=3))
    print('----------------------------------------------')
