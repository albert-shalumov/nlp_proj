from nltk.tag.hmm import *

print('Preparing data')
train_set = [[],[]]
test_set = [[],[]]
symbols = []

print('Training')
hmm_trainer = HiddenMarkovModelTrainer(states = ['a','e','u','i','o','*'], symbols=symbols)
model = hmm_trainer.train(labeled_sequences=train_set)

print('Prediction')
pred_set = model.tag(test_set[0])
