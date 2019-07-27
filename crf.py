from nltk.tag.crf import CRFTagger
import codecs
import numpy as np
import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition.pca import PCA
from functools import reduce as reduce

vowels = [u'a',u'e',u'u',u'i',u'o',u'*']
vowels_idx = {x:i for i,x in enumerate(vowels)}
chars = [u'\u02c0', u'b',u'g', u'd', u'h', u'w', u'z', u'\u1e25', u'\u1e6d', u'y',
        u'k', u'k', u'l', u'm', u'm', u'n', u'n', u's', u'\u02c1', u'p', u'p', u'\u00e7',
        u'\u00e7', u'q', u'r', u'\u0161', u't']
chars_idx = {x:i for i,x in enumerate(chars)}



def ExtractNgrams(word, n=1):
    start_symb=u'-'
    start_w = [start_symb]*(n-1)
    w = start_w+list(word)
    l = []
    for i in range(len(w)-n+1):
        l.append(''.join(w[i:i+n]))
    return l

MAX_FTR_LEN = 150
# i is the letter position
def ExtractWordFtr(word, i, ftr):
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


# j is the word in the sentence
# i is the letter position
def ExtractSentenceFtr(sentence, j, i):
    return []

def SentenceCRF(ftr_set, iters=5):
    data = list()
    symbols = set()

    #print('Preparing data')
    with codecs.open('data/HaaretzOrnan_annotated.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if line.startswith(u'#') or len(line)==0:
                continue
            w = line.split(u' ')[3]
            w = w.replace(u'-', u'')
            ngrams = ExtractNgrams(w[::2], 1)
            vowel = list(w[1::2])
            data.append(list(zip(ngrams, vowel)))
            symbols.update(ngrams)
    conf_mat = np.zeros((len(vowels), len(vowels)))

    return conf_mat

def len_(a):
    return len(a) if isinstance(a,str) else int(a)


def WordCRF(ftr_set, iters=5):
    data = list()
    symbols = set()

    words = []
    num_chars = 0

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

    X = np.zeros((num_train_chars//2, MAX_FTR_LEN))
    Y = np.zeros(num_train_chars//2)

    sample=0
    for w in train:
        for i in range(len(w)//2):
            ExtractWordFtr(w[::2], i, X[sample,:])
            Y[sample] = vowels_idx[w[1::2][i]]
            sample+=1

    conf_mat = np.zeros((len(vowels), len(vowels)))

    # Train
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
    model.fit(X,Y)

    # Predict
    X = np.zeros((num_valid_chars//2, MAX_FTR_LEN))
    Y = np.zeros(num_valid_chars//2)

    sample=0
    for w in valid:
        for i in range(len(w)//2):
            ExtractWordFtr(w[::2], i, X[sample,:])
            Y[sample] = vowels_idx[w[1::2][i]]
            sample += 1
    pred = model.predict(X)
    for i in range(pred.shape[0]):
        conf_mat[int(pred[i]),int(Y[i])] += 1

    return conf_mat

print("Word-wise CRF: ")
conf_mat = WordCRF(0, 50)
precision, recall = metrics.MicroAvg(conf_mat)
f1 = metrics.Fscore(precision, recall, 1)
print('MicroAvg:',precision,recall,f1)
precision, recall = metrics.MacroAvg(conf_mat)
f1 = metrics.Fscore(recall, precision, 1)
print('MacroAvg:', precision, recall, f1)
print('AvgAcc:',metrics.AvgAcc(conf_mat))
conf_mat = metrics.NormalizeConfusion(conf_mat)
print('ConfMat:\n', np.array_str(conf_mat, max_line_width=300, precision=4))
print('----------------------------------------------')#

# print("Sentence-wise CRF: ")
# conf_mat = SentenceCRF(0, 50)
# precision, recall = metrics.MicroAvg(conf_mat)
# f1 = metrics.Fscore(precision, recall, 1)
# print('MicroAvg:',precision,recall,f1)
# precision, recall = metrics.MacroAvg(conf_mat)
# f1 = metrics.Fscore(recall, precision, 1)
# print('MacroAvg:', precision, recall, f1)
# print('AvgAcc:',metrics.AvgAcc(conf_mat))
# conf_mat = metrics.NormalizeConfusion(conf_mat)
# print('ConfMat:\n', np.array_str(conf_mat, max_line_width=300, precision=4))
# print('----------------------------------------------')

