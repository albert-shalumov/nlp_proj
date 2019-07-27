from nltk.tag.crf import CRFTagger
import codecs
import numpy as np
import metrics

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

MAX_FTR_LEN = 50
# i is the letter position
def ExtractWordFtr(word, i, ftr):
    idx=0

    ftr[idx] = 1 if i==0 else 0  # first letter
    idx+=1

    ftr[idx] = 1 if i == (len(word)-1) else 0  # last letter
    idx+=1

    ftr[idx+chars_idx[word[i]]] = 1  # unigram
    idx+=len(chars)

    assert idx<=MAX_FTR_LEN


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
            print(line)
            w = line.split(u' ')[3]
            w = w.replace(u'-', u'')
            #ngrams = ExtractNgrams(w[::2], 1)
            #vowel = list(w[1::2])
            words.append(w)
            num_chars+=len(w)/2
            print(num_chars)

    num_chars = int(num_chars)
    X = np.zeros((num_chars, MAX_FTR_LEN))
    Y = np.zeros((num_chars, len(vowels)))

    sample=0
    for w in words:
        for i in range(len(w)//2):
            ExtractWordFtr(w[::2], i, X[sample,:])
            Y[sample, vowels_idx[w[1::2][i]]] = 1
            sample+=1

    conf_mat = np.zeros((len(vowels), len(vowels)))

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
print('----------------------------------------------')

print("Sentence-wise CRF: ")
conf_mat = SentenceCRF(0, 50)
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

