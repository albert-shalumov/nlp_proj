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
    for j in range(max_len, len(data_in[i])):
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
    Y[i] = out1_char2int[seq_out[i]]  # value
    #Y[i,out1_char2int[seq_out[i]]] = 1  # one hot

print('wait')

# shuffle+train/valid split
inds =np.arange(len(seq_out))
np.random.shuffle(inds)
X = X[inds]
Y = Y[inds]
batch_size = 16
train_smpls = batch_size*int(X.shape[0]*.95/batch_size)
X_train = X[np.newaxis,: train_smpls,:]
X_train = np.reshape(X_train, (batch_size,-1,X_train.shape[-1]))
valid_smpls = batch_size*int((X.shape[0]-train_smpls)/batch_size)
X_valid = X[np.newaxis,train_smpls:train_smpls+valid_smpls,:]
X_valid = np.reshape(X_valid, (batch_size,-1,X_valid.shape[-1]))

Y_train = Y[:train_smpls]
Y_train = np.reshape(Y_train, (batch_size,-1,Y_train.shape[-1]))
Y_valid = Y[np.newaxis,train_smpls:train_smpls+valid_smpls]
Y_valid = np.reshape(Y_valid, (batch_size,-1,Y_valid.shape[-1]))


print('wait')

# pytorch model
train_in_seq = torch.from_numpy(X_train.astype(np.float32))
train_tgt_seq = torch.Tensor(Y_train.astype(np.float32))

valid_in_seq = torch.from_numpy(X_valid.astype(np.float32))
valid_tgt_seq = torch.Tensor(Y_valid.astype(np.float32)).long()

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

class Model2(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model2, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        drop_prob = 0.001

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        #self.sigmoid = nn.Sigmoid() # softmax loss includes sigmoid


    def forward(self, x, hidden):
        batch_size = x.size(0)
        #x = x.long()
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        #out = self.sigmoid(out)

        out = out.view(-1, self.output_size)
        #out = out[:, -1]
        return out, hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


# Instantiate the model with hyperparameters
#model = Model(input_size=in_emb_size*max_len, output_size=out_emb_size, hidden_dim=32, n_layers=3)
model = Model2(input_size=in_emb_size*max_len, output_size=out_emb_size, hidden_dim=max_len, n_layers=25)
# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)

# Define hyperparameters
n_epochs = 100000
lr=1e-1

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

# Training Run
train_loss = []
valid_loss = []
cnt=0
abort = False
for epoch in range(1, n_epochs+1):
    hidden = model.init_hidden(batch_size)
    optimizer.zero_grad()  # Clears existing gradients from previous epoch
    input_seq = train_in_seq.to(device)
    output, _ = model(train_in_seq, hidden)
    loss = criterion(output, train_tgt_seq.view(-1).long())
    loss.backward()
    optimizer.step()

    if True or epoch%20==0:
        with torch.no_grad():
            valid_in_seq = valid_in_seq.to(device)
            valid_tgt_seq = valid_tgt_seq.to(device)
            pred,_ = model(valid_in_seq, model.init_hidden(batch_size))
            v_loss = criterion(pred, valid_tgt_seq.view(-1).long()).item()
            y = pred.detach().numpy()

    train_loss.append(loss.item())
    valid_loss.append(v_loss)

    if epoch>10: # early stop check
        if v_loss>=np.average(np.array(valid_loss[-6:])):
            cnt+=1
        else:
            cnt=0

    print(train_loss[-1],valid_loss[-1])

    if cnt>=5:
        cnt = 0
        lr = lr*0.95
        for g in optimizer.param_groups:
            g['lr'] = lr

    if abort:
        break

with open('loss1.csv','w') as f:
    for i, _ in enumerate(train_loss):
        line = str(i) + ',' + str(train_loss[i])+','+str(valid_loss[i]) + '\n'
        f.write(line)
