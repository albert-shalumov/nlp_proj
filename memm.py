from nltk.tag.crf import CRFTagger
import codecs
import numpy as np
import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition.pca import PCA
from functools import reduce as reduce
from collections import defaultdict

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

# ========================================================================
# ========================================================================
# ========================================================================
if __name__ == '__main__':
    print("Word MEMM: ")
    conf_mat = WordMEMM(0, 50)
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
