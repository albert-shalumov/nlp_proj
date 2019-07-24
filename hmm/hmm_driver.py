from nltk.tag.hmm import *
import codecs
import numpy as np

print('Preparing data')
data = list()
symbols = set()

with codecs.open('../data/HaaretzOrnan_annotated.txt', encoding='utf-8') as f:
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

np.random.shuffle(data)
valid_size = max(1, len(data)//7)
train = data[:-valid_size]
valid = data[-valid_size:]

print('Training')
hmm_trainer = HiddenMarkovModelTrainer(states = ['a','e','u','i','o','*'], symbols=list(symbols))
model = hmm_trainer.train(labeled_sequences=train)

print('Prediction')
valid_word_cons = [[x[0] for x in w] for w in valid]
valid_word_vowel = [[x[1] for x in w] for w in valid]
print('Accuracy:',model.evaluate(valid))
print('Results:')

for i,w_cons in enumerate(valid_word_cons):
    print('     ',w_cons)
    print('Gold:',valid_word_vowel[i])
    pred_set = model.best_path(w_cons)
    print('Pred:',pred_set)
    print('-------------------------------')
