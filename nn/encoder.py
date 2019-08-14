from embedding import Embedding_ as WordEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import codecs
import time
import numpy as np
import re


class EncoderRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(EncoderRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        drop_prob = 0.25

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        lstm_out, hidden = self.lstm(input, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)

        out = out.view(-1, self.output_size)
        #out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

class Encoder:
    def __init__(self, hidden_dim=250, n_layers=1, device=None):
        self.hid_dim = hidden_dim
        self.n_layers = n_layers
        self.proc_units = None
        self.n_layers = n_layers
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        self.batch_size = 16
        self.word_embedding = WordEmbedding()
        self.word_embedding.LoadModel(True)
        self.word_emb_size = 150

    def prep_model(self, file='../data/HaaretzOrnan_annotated.txt'):
        self.proc_units = [[]]
        with codecs.open(file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                if line.startswith(u'#'):
                    continue
                if len(line) == 0:
                    continue
                split_line = line.split(u' ')
                self.proc_units[-1] += [(split_line[2], split_line[3])]
                self.proc_units.append([])
        if len(self.proc_units[-1]) == 0:
            self.proc_units.pop(-1)

        # Calculate max unit length, dictionaries
        self.max_len = Encoder.CalcMaxLen(self.proc_units)+5
        unit_dict_sets = Encoder.BuildCharSets(self.proc_units)
        unit_dict_sets[0].add(' ')
        unit_dict_sets[1].update(list('euioa*'))
        unit_dict_sets[1].update(['e-', 'u-', 'i-', 'o-', 'a-', '*-', ' '])
        self.emb_in_size = len(unit_dict_sets[0])+1
        self.emb_out_size = len(unit_dict_sets[1])+1

        # Init lookup tables
        self.input_int2char, self.input_char2int = Encoder.CreateDictionaries(unit_dict_sets[0])
        self.output_int2char, self.output_char2int = Encoder.CreateDictionaries(unit_dict_sets[1])

        # Create encoder
        self.encoder = EncoderRNN(self.emb_in_size*self.max_len+self.word_emb_size, self.emb_out_size,self.hid_dim, self.n_layers).to(self.device)

        # Build sliding window sequences
        X = []
        Y = []
        for i in range(len(self.proc_units)):
            in_word = self.proc_units[i][0][0]
            out_word = self.proc_units[i][0][1]
            ch_out = [x for x in re.split('[' + in_word + ']', out_word) if len(x) > 0]
            for j, ch in enumerate(in_word):
                w = in_word[:j+1]
                w = w.rjust(self.max_len)
                X.append((w, in_word))
                Y.append(ch_out[j])

        # Convert to numpy arrays
        self.X = np.zeros((len(X), self.word_emb_size + self.max_len*self.emb_in_size))
        self.Y = np.zeros(len(Y))
        for i in range(len(X)):
            self.X[i, :self.word_emb_size] = self.word_embedding.GetEmbedding(X[i][1])
            for j,ch in enumerate(X[i][0]):
                self.X[i, self.word_emb_size + j*self.emb_in_size+self.input_char2int[ch]] = 1
            self.Y[i] = self.output_char2int[Y[i]]

        return self

    def shuffle(self, seed=None):
        self.inds = np.arange(len(self.proc_units))
        np.random.seed(seed)
        np.random.shuffle(self.inds)
        return self

    def split(self, valid_ratio=0.1):
        train_smpls = self.batch_size*int(self.X.shape[0]*(1-valid_ratio)/self.batch_size)
        valid_smpls = self.batch_size*int((self.X.shape[0]-train_smpls)/self.batch_size)

        self.train_X = self.X[np.newaxis, :train_smpls, :].reshape(self.batch_size, -1, self.X.shape[-1])
        self.train_Y = self.Y[np.newaxis, :train_smpls].reshape(self.batch_size, -1)
        if valid_smpls == 0:
            self.valid_X = None
            self.valid_Y = None
        else:
            self.valid_X = self.X[np.newaxis, train_smpls:train_smpls+valid_smpls, :].reshape(self.batch_size, -1, self.X.shape[-1])
            self.valid_Y = self.Y[np.newaxis, train_smpls:train_smpls+valid_smpls].reshape(self.batch_size, -1)

        return self

    def train(self, epochs=10):
        lr = 1e-2
        optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True, cooldown=5)
        criterion = nn.CrossEntropyLoss()

        train_inp_seq = torch.from_numpy(self.train_X).float().to(self.device)
        train_tgt_seq = torch.Tensor(self.train_Y).long().to(self.device)
        if self.valid_Y is not None:
            valid_inp_seq = torch.from_numpy(self.valid_X).float().to(self.device)
            valid_tgt_seq = torch.Tensor(self.valid_Y).long().to(self.device)

        for epoch in range(1, epochs+1):
            hidden = self.encoder.init_hidden(self.batch_size)
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            output, _ = self.encoder(train_inp_seq, hidden)
            loss = criterion(output, train_tgt_seq.view(-1))
            loss.backward()
            optimizer.step()

            if epoch%5 == 0:
                if self.valid_X is None:
                    scheduler.step(loss.item())
                    print(epoch, loss.item())
                else:
                    with torch.no_grad():
                        pred, _ = self.encoder(valid_inp_seq, self.encoder.init_hidden(self.batch_size))
                        val_loss = criterion(pred, valid_tgt_seq.view(-1))
                        scheduler.step(val_loss.item(),epoch)
                    print(epoch, loss.item(), val_loss.item())

    def predict(self, pred_set):
        res_set = []
        for sentence in pred_set:
            res_set.append([])
            for word in sentence:
                res_set[-1].append([])
                input = np.zeros((len(word), self.word_emb_size+self.emb_in_size*self.max_len))
                input[:,:self.word_emb_size] = self.word_embedding.GetEmbedding(word)
                for j, ch in enumerate(word):
                    input[j,self.word_emb_size+j*self.emb_in_size+self.input_char2int[ch]] = 1
                input = input.reshape((1,input.shape[0],input.shape[-1]))
                input = torch.from_numpy(input).float().to('cpu')
                with torch.no_grad():
                    pred, _ = self.encoder(input, self.encoder.init_hidden(input.size(0)))
                    y = pred.detach().numpy()
                    inds = list(np.argmax(y, axis=-1))
                    for j, ch in enumerate(word):
                        res_set[-1][-1].append(ch)
                        res_set[-1][-1].append(self.output_int2char[max(inds[j],1)])
                    res_set[-1][-1] = ''.join(res_set[-1][-1])
        return res_set

    @staticmethod
    def BuildCharSets(proc_units):
        sets = [set(), set()]
        for unit_list in proc_units:
            for unit in unit_list:
                units = list(unit)
                sets[0].update(list(units[0]))
                ch_out = [x for x in re.split('[' + units[0] + ']', units[1]) if len(x) > 0]
                sets[1].update(ch_out)
        return sets

    @staticmethod
    def CalcMaxLen(proc_units):
        max_len = 0
        for unit_list in proc_units:
            for unit in unit_list:
                units = list(unit)
                max_len = max(max_len, len(units[0]))
                max_len = max(max_len, len(units[1]))
        return max_len

    @staticmethod
    def CreateDictionaries(unit_set, reserve_zero=True):
        int2char = {i+1:ch for i,ch in enumerate(list(unit_set))}
        char2int = {char: ind for ind, char in int2char.items()}
        return int2char, char2int

if __name__=='__main__':
    encoder = Encoder(256, 1, device='cpu')
    encoder.prep_model().shuffle(0).split(0.99).train(500)
    print(encoder.predict([[u'ˀnšym',u'nršmym',u'twpˁh']]))
