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

# ========================================================================
# =====================  CRF with word features only  ====================
# ========================================================================

def WordCrfFtr(tokens, i):
    #print(tokens, i, tokens[i])
    feature_list = []

    feature_list.append("pos="+str(i))
    feature_list.append("cur="+tokens[i])
    feature_list.append("first="+tokens[0])
    feature_list.append("last="+tokens[-1])
    feature_list.append("len="+str(len(tokens)))
    if i>0:
        feature_list.append("prev="+tokens[i-1])

    if i<(len(tokens)-1):
        feature_list.append("next="+tokens[i+1])

    return feature_list

def WordCRF(ftr_set, iters=5):
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
    #train = words[:-valid_size]
    #valid = words[-valid_size:]
    train = []
    for w in words[:-valid_size]:
        train.append(list(zip(list(w[::2]), list(w[1::2]))))

    ct = CRFTagger(WordCrfFtr, verbose=False, training_opt={"c1":0,"c2":0, "num_memories":50, "epsilon":1e-7, "delta":1e-8})
    ct.train(train, 'word_crf_model')

    conf_mat = np.zeros((len(vowels), len(vowels)))
    valid_in = []
    valid_gold = []
    for w in words[:-valid_size]:
        valid_in.append(list(w[::2]))
        valid_gold.append(list(w[1::2]))

    valid_pred = ct.tag_sents(valid_in)
    for i,w in enumerate(valid_pred):
        pred_vow = [x[1] for x in w]
        for j in range(len(pred_vow)):
            conf_mat[vowels_idx[pred_vow[j]], vowels_idx[valid_gold[i][j]]] += 1

    return conf_mat

# ========================================================================
# =====================  CRF with sentence features   ====================
# ========================================================================

def SentCrfFtr(tokens, i):
    feature_list = []
    word_pos = tokens[0][1]
    sent = tokens[0][2].split(' ')
    word = sent[word_pos]

    # Word features
    feature_list.append("pos="+str(i))
    feature_list.append("cur="+word[i])
    feature_list.append("first="+word[0])
    feature_list.append("last="+word[-1])
    feature_list.append("len="+str(len(word)))
    if i>0:
        feature_list.append("prev="+word[i-1])

    if i<(len(word)-1):
        feature_list.append("next"+word[i+1])

    if word_pos==0:
        feature_list.append('FIRST_WORD')

    if word_pos == (len(sent)-1):
        feature_list.append('LAST_WORD')

    if word_pos>0:
        feature_list.append("prev_w_last_ch="+sent[word_pos-1][-1])

    # Sentence features
    #feature_list.append(str(word_pos))
    return feature_list


def SentenceCRF(ftr_set, iters=5):
    sents = []
    #print('Preparing data')
    with codecs.open('data/HaaretzOrnan_annotated.txt', encoding='utf-8') as f:
        lines = f.readlines()
        sents.append([])
        for line in lines:
            line = line.rstrip()
            if line.startswith(u'#'):
                continue
            if len(line)==0:
                if len(sents[-1])>0:
                    sents.append([])
                continue

            w = line.split(u' ')[3]
            w = w.replace(u'-', u'')
            sents[-1].append(w)

    if len(sents[-1])==0: # remove last "sentence" if empty
        sents.remove(sents[-1])

    np.random.shuffle(sents)
    valid_size = max(1, len(sents)//10)
    #train = words[:-valid_size]
    #valid = words[-valid_size:]
    train = []
    for sent in sents[:-valid_size]:
        sent_cons = u' '.join([x[::2] for x in sent])
        for i,w in enumerate(sent):
            w_cons = list(w[::2])
            w_pos = [i]*len(w[::2])
            sent = [sent_cons]*len(w[::2])
            d = list(zip(w_cons, w_pos, sent))
            train.append(list(zip(d, list(w[1::2]))))

    ct = CRFTagger(SentCrfFtr, verbose=False, training_opt={"c1":0,"c2":0, "num_memories":50, "epsilon":1e-7, "delta":1e-8})
    ct.train(train, 'sent_crf_model')

    conf_mat = np.zeros((len(vowels), len(vowels)))
    valid_in = []
    valid_gold = []
    for sent in sents[-valid_size:]:
        sent_cons = u' '.join([x[::2] for x in sent])
        for i,w in enumerate(sent):
            w_cons = list(w[::2])
            w_pos = [i]*len(w[::2])
            sent = [sent_cons]*len(w[::2])
            d = list(zip(w_cons, w_pos, sent))
            valid_in.append(list(d))
            valid_gold.append(list(w[1::2]))

    valid_pred = ct.tag_sents(valid_in)
    for i,w in enumerate(valid_pred):
        pred_vow = [x[1] for x in w]
        for j in range(len(pred_vow)):
            conf_mat[vowels_idx[pred_vow[j]], vowels_idx[valid_gold[i][j]]] += 1

    return conf_mat

# ========================================================================
# ========================================================================
# ========================================================================

print("Word CRF: ")
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

