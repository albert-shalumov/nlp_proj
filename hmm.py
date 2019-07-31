from nltk.tag.hmm import *
import codecs
import numpy as np
from sklearn.metrics import confusion_matrix
import metrics

states = [u'a',u'e',u'u',u'i',u'o',u'*']
state_idx = {x:i for i,x in enumerate(states)}

def ExtractNgrams(word, n=1):
    start_symb=u'-'
    start_w = [start_symb]*(n-1)
    w = start_w+list(word)
    l = []
    for i in range(len(w)-n+1):
        l.append(''.join(w[i:i+n]))
    return l

def Predict(words, ngram=1):
    data = list()
    symbols = set()
    with codecs.open('data/HaaretzOrnan_annotated.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if line.startswith(u'#') or len(line)==0:
                continue
            w = line.split(u' ')[3]
            w = w.replace(u'-', u'')
            ngrams = ExtractNgrams(w[::2], ngram)
            vowel = list(w[1::2])
            data.append(list(zip(ngrams, vowel)))
            symbols.update(ngrams)

    np.random.shuffle(data)
    train = data
    hmm_trainer = HiddenMarkovModelTrainer(states=states, symbols=list(symbols))
    model = hmm_trainer.train(labeled_sequences=train)

    result = []
    for i, w_cons in enumerate(words):
        pred_set = model.best_path(w_cons)
        result.append(''.join(x+y for x,y in zip(w_cons, pred_set)))

    return result

def TestNgram(n=1,iters=5):
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
            ngrams = ExtractNgrams(w[::2], n)
            vowel = list(w[1::2])
            data.append(list(zip(ngrams, vowel)))
            symbols.update(ngrams)


    conf_mat = np.zeros((len(states),len(states)))

    for i in range(iters):
        np.random.shuffle(data)
        valid_size = max(1, len(data)//10)
        train = data[:-valid_size]
        valid = data[-valid_size:]

        #print('Training {}/{}'.format(i,iters))
        hmm_trainer = HiddenMarkovModelTrainer(states = states, symbols = list(symbols))
        model = hmm_trainer.train(labeled_sequences=train)
        #print('Prediction')
        valid_word_cons = [[x[0] for x in w] for w in valid]
        valid_word_vowel = [[x[1] for x in w] for w in valid]

        #print('Results:')
        for i,w_cons in enumerate(valid_word_cons):
            #print('     ',w_cons)
            #print('Gold:',valid_word_vowel[i])
            pred_set = model.best_path(w_cons)
            for j in range(len(pred_set)):
                conf_mat[state_idx[pred_set[j]],state_idx[valid_word_vowel[i][j]]] += 1
            #print('Pred:',pred_set)
            #print('-------------------------------')
    return conf_mat


for i in range(1,5,1):
    print("|Ngram| = {}: ".format(i))
    conf_mat = TestNgram(i, 50)
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
