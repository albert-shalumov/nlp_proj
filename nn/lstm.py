import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from embedding import Embedding,FastTextEmb
import codecs
import re
import numpy as np

dict_in = set(' ')
dict_out1 = set(' ')
dict_out1.update(list('euioa*'))
dict_out1.update(['e-','u-','i-','o-','a-','*-'])
dict_out2 = set()
dict_out2.update(list('euioa- '))

with codecs.open('../data/HaaretzOrnan_annotated.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        if line.startswith(u'#') or len(line) == 0:
            continue

        w_in = line.split(u' ')[2]
        dict_in.update(list(w_in))

        #w_out1 = line.split(u' ')[3]
        #dict_out1.update([x for x in re.sub('[^euioa*\-]',' ',w_out1).split(' ') if len(x)>0])

        w_out2 = line.split(u' ')[4]
        dict_out2.update([x for x in re.sub('[euioa\-]',' ',w_out2).split(' ') if len(x)>0])


in_int2char = dict(enumerate(dict_in))
in_char2int = {char: ind for ind, char in in_int2char.items()}

out2_int2char = dict(enumerate(dict_out2))
out2_char2int = {char: ind for ind, char in out2_int2char.items()}

out1_int2char = dict(enumerate(dict_out1))
out1_char2int = {char: ind for ind, char in out1_int2char.items()}


# data is small enough to hold entirely in memory
max_len = 5
data_in = []
data_out = []
# exctract sequences for sentence
with codecs.open('../data/HaaretzOrnan_annotated.txt', encoding='utf-8') as f:
    lines = f.readlines()
    stc_in = [' ']*(max_len-1)
    stc_out = []
    for line in lines:
        line = line.rstrip()
        if line.startswith(u'#'):
            continue
        if len(line) == 0:
            data_in.append(stc_in)
            data_out.append(stc_out)
            stc_in = [' ']*(max_len-1)
            stc_out = []
            #if len(data_in[-1]) > 0:
            #    data_in.append([])
            continue

        # append words to in,out
        w_in = line.split(u' ')[2]
        w_out = line.split(u' ')[3]
        ch_in = list(w_in)
        ch_out = [x for x in re.split('['+w_in+']',w_out) if len(x)>0]
        assert len(ch_in)==len(ch_out)

        for i in range(len(ch_in)):
            stc_in.append(ch_in[i])
            stc_out.append(ch_out[i])
        stc_in.append(' ')
        stc_out.append(' ')

        #print('wait')

seq_in = []
seq_out = []
# pool sequences
for i in range(len(data_out)):
    for j in range(max_len, len(data_in)):
        seq_in.append(data_in[i][j-max_len:j])
        seq_out.append(data_out[i][j-max_len])

print('wait')

# embed samples
in_emb_size = len(dict_in)+1
X = np.zeros((len(seq_out), max_len*in_emb_size))
out_emb_size = len(dict_out1)+1
Y = np.zeros((len(seq_out), 1))

for i in range(len(seq_out)):
    for j,ch in enumerate(seq_in[i]):
        X[i,j*in_emb_size+in_char2int[ch]] = 1
    Y[i] = out1_char2int[seq_out[i]]

print('wait')

# shuffle+train/valid split
inds =np.arange(len(seq_out))
np.random.shuffle(inds)
X = X[inds]
Y = Y[inds]

X_valid = X[np.newaxis,:X.shape[0]//20,:]
Y_valid = Y[np.newaxis,:X.shape[0]//20]

X_train = X[np.newaxis,X_valid.shape[0]:,:]
Y_train = Y[np.newaxis,Y_valid.shape[0]:]
print('wait')

# pytorch model
train__in_seq = torch.from_numpy(X_train.astype(np.float32))
train_tgt_seq = torch.Tensor(Y_train.astype(np.float32))

valid_in_seq = torch.from_numpy(X_valid.astype(np.float32))
valid_tgt_seq = torch.Tensor(Y_valid.astype(np.float32))

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden


# Instantiate the model with hyperparameters
model = Model(input_size=in_emb_size*max_len, output_size=out_emb_size, hidden_dim=12, n_layers=1)
# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)

# Define hyperparameters
n_epochs = 1000
lr=0.01

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training Run
for epoch in range(1, n_epochs+1):
    optimizer.zero_grad()  # Clears existing gradients from previous epoch
    input_seq = train__in_seq.to(device)
    output, hidden = model(train__in_seq)
    loss = criterion(output, train_tgt_seq.view(-1).long())
    loss.backward()  # Does backpropagation and calculates gradients
    optimizer.step()  # Updates the weights accordingly

    if epoch%5 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

    if epoch%10==0:
        with torch.no_grad():
            valid_in_seq = valid_in_seq.to(device)
            valid_tgt_seq = valid_tgt_seq.to(device)
            model.eval()
            y_pred,_ = model(valid_in_seq)
            # TODO: convert to pytorch ops
            y_pred = np.exp(y_pred.numpy())
            norm = np.sum(y_pred,axis=-1)
            y_pred = y_pred/norm[:,np.newaxis]
            y_pred = np.argmax(y_pred, -1)
            y_gt = Y_valid[0,:,0]
            #print(y_pred.shape)
            #print(y_gt.shape)

            print("Valid acc: {}".format(np.sum(y_pred==y_gt)/y_pred.shape[0]))

    if epoch%200==0:
        lr = lr/10
