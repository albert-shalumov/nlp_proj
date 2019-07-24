from nltk.tag.hmm import *
import codecs
import numpy as np
from sklearn.metrics import confusion_matrix
import metrics

print('Preparing data')
data = list()
symbols = set()
states = ['a','e','u','i','o','*']
state_idx = {x:i for i,x in enumerate(states)}

with codecs.open('data/HaaretzOrnan_annotated.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        if line.startswith(u'#') or len(line)==0:
            continue
        w = line.split(u' ')[3]
        w = w.replace(u'-', u'')
        cons = list(w[::2])
        vowel = list(w[1::2])
        data.append(list(zip(cons, vowel)))
        symbols.update(cons)


iterations = 50
conf_mat = np.zeros((len(states),len(states)))
for i in range(iterations):
    np.random.shuffle(data)
    valid_size = max(1, len(data)//7)
    train = data[:-valid_size]
    valid = data[-valid_size:]

    print('Training {}/{}'.format(i,iterations))
    hmm_trainer = HiddenMarkovModelTrainer(states = ['a','e','u','i','o','*'], symbols=list(symbols))
    model = hmm_trainer.train(labeled_sequences=train)
    #print('Prediction')
    valid_word_cons = [[x[0] for x in w] for w in valid]
    valid_word_vowel = [[x[1] for x in w] for w in valid]

    #print('Results:')
    for i,w_cons in enumerate(valid_word_cons):
    #    print('     ',w_cons)
    #    print('Gold:',valid_word_vowel[i])
        pred_set = model.best_path(w_cons)
        for j in range(len(pred_set)):
            conf_mat[state_idx[pred_set[j]],state_idx[valid_word_vowel[i][j]]] += 1
    #    print('Pred:',pred_set)
    #    print('-------------------------------')
print('MicroAvg:',metrics.MicroAvg(conf_mat))
print('MacroAvg:',metrics.MacroAvg(conf_mat))
print('AvgAcc:',metrics.AvgAcc(conf_mat))
print('ConfMat:\n', np.array_str(conf_mat, max_line_width=300, precision=4))