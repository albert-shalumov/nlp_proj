from embedding import Embedding_ as WordEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import codecs
import time
import numpy as np
import re
import sys

class EncoderRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout, device):
        super(EncoderRNN, self).__init__()
        self.bidir = False
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.drop_prob = dropout
        self.device=device
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=(self.drop_prob if self.n_layers>1 else 0), batch_first=True, bidirectional=self.bidir)
        self.fc_cat = nn.Linear((2 if self.bidir else 1)*hidden_dim+150, hidden_dim)
        self.cat_act = nn.LeakyReLU(0.01)
        self.fc1 = nn.Linear(hidden_dim, output_size)
        self.fc2 = nn.Linear(hidden_dim+150, hidden_dim)
        self.act1 = nn.LeakyReLU(0.01)
        self.dropout2 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(output_size, output_size)

    def forward(self, input, mask, hidden):
        batch_size = input.size(0)
        lstm_out, hidden = self.lstm(input[:,:,150:]*mask, hidden)
        out = self.fc_cat(torch.cat((lstm_out, input[:,:,:150]),-1))
        out = self.cat_act(out)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc1(out)
        out = self.act1(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = out.view(-1, self.output_size)
        #out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers*(2 if self.bidir else 1), batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.n_layers*(2 if self.bidir else 1), batch_size, self.hidden_dim).to(self.device)) # (h_0, c_0) - for lstm
        return hidden

class Encoder:
    def __init__(self, n_layers=1, droput=0.1, device=None):
        self.hid_dim = None
        self.n_layers = n_layers
        self.proc_units = None
        self.n_layers = n_layers
        self.dropout = droput
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        self.batch_size = 128
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
        self.hid_dim = 128
        unit_dict_sets = Encoder.BuildCharSets(self.proc_units)
        unit_dict_sets[0].add(' ')
        unit_dict_sets[1].update(list('euioa*'))
        #unit_dict_sets[1].update(['e-', 'u-', 'i-', 'o-', 'a-', '*-'])
        self.emb_in_size = len(unit_dict_sets[0])+1
        self.emb_out_size = len(unit_dict_sets[1])+1

        # Init lookup tables
        self.input_int2char, self.input_char2int = Encoder.CreateDictionaries(unit_dict_sets[0])
        self.output_int2char, self.output_char2int = Encoder.CreateDictionaries(unit_dict_sets[1])

        # Create encoder
        self.encoder = EncoderRNN(self.emb_in_size*self.max_len, self.emb_out_size,self.hid_dim, self.n_layers,self.dropout, self.device).to(self.device)

        # Build sliding window sequences
        X = []
        Y = []
        for i in range(len(self.proc_units)):
            in_word = self.proc_units[i][0][0]
            out_word = self.proc_units[i][0][1]
            #ch_out = [x for x in re.split('[' + in_word + ']', out_word) if len(x) > 0]
            ch_out = list(out_word.replace('-','')[1::2])
            for j, ch in enumerate(in_word):
                w = in_word[:j+1]
                w = w.rjust(self.max_len)
                X.append((w, in_word))
                Y.append(ch_out[j])

        # Convert to numpy arrays
        self.X = np.zeros((len(X), self.word_emb_size + self.max_len*self.emb_in_size))
        self.Y = np.zeros(len(Y))
        self.Mask = np.zeros((len(X), self.max_len*self.emb_in_size))
        for i in range(len(X)):
            self.X[i, :self.word_emb_size] = self.word_embedding.GetEmbedding(X[i][1])
            for j,ch in enumerate(X[i][0]):
                self.X[i, self.word_emb_size + j*self.emb_in_size+self.input_char2int[ch]] = 1
                if ch!=' ':
                    self.Mask[i, j*self.emb_in_size:(j+1)*self.emb_in_size] = 1
            self.Y[i] = self.output_char2int[Y[i]]

        return self

    def shuffle(self, seed=None):
        self.inds = np.arange(self.X.shape[0])
        np.random.seed(seed)
        np.random.shuffle(self.inds)
        self.X = self.X[self.inds,:]
        self.Y = self.Y[self.inds]
        self.Mask = self.Mask[self.inds, :]
        return self

    def split(self, valid_ratio=0.1):
        assert valid_ratio>0

        train_smpls = self.batch_size*int(self.X.shape[0]*(1-valid_ratio)/self.batch_size)
        valid_smpls = self.batch_size*int((self.X.shape[0]-train_smpls)/self.batch_size)

        self.train_X = self.X[np.newaxis, :train_smpls, :].reshape(self.batch_size, -1, self.X.shape[-1])
        self.train_Y = self.Y[np.newaxis, :train_smpls].reshape(self.batch_size, -1)
        self.train_Mask = self.Mask[np.newaxis, :train_smpls, :].reshape(self.batch_size, -1, self.Mask.shape[-1])
        self.valid_X = self.X[np.newaxis, train_smpls:train_smpls+valid_smpls, :].reshape(self.batch_size, -1, self.X.shape[-1])
        self.valid_Y = self.Y[np.newaxis, train_smpls:train_smpls+valid_smpls].reshape(self.batch_size, -1)
        self.valid_Mask = self.Mask[np.newaxis, train_smpls:train_smpls+valid_smpls, :].reshape(self.batch_size, -1, self.Mask.shape[-1])

        return self

    def train(self, epochs=10, lr=1e-2, alg = 'adagrad'):
        if alg=='adagrad':
            optim_alg = optim.Adagrad
        elif alg=='rprop':
            optim_alg = optim.Rprop
        elif alg == 'adamax':
            optim_alg = optim.Adamax
        elif alg == 'adamw':
            optim_alg = optim.AdamW
        elif alg == 'asgd':
            optim_alg = optim.ASGD
        elif alg == 'rmsprop':
            optim_alg = optim.RMSprop
        elif alg=='lr_scan':
            optim_alg = optim.SGD
        else:
            raise Exception('Incorrect optimizer')

        optimizer = optim_alg(self.encoder.parameters(), lr=lr)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-4, verbose=False, cooldown=5)
        criterion = nn.CrossEntropyLoss()

        train_inp_seq = torch.from_numpy(self.train_X).float().to(self.device)
        train_tgt_seq = torch.Tensor(self.train_Y).long().to(self.device)
        train_mask = torch.Tensor(self.train_Mask).to(self.device)
        valid_inp_seq = torch.from_numpy(self.valid_X).float().to(self.device)
        valid_tgt_seq = torch.Tensor(self.valid_Y).long().to(self.device)
        valid_mask = torch.Tensor(self.valid_Mask).to(self.device)

        min_val_loss = (9999, 0)

        for epoch in range(1, epochs+1):
            hidden = self.encoder.init_hidden(self.batch_size)
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            output, _ = self.encoder(train_inp_seq, train_mask, hidden)
            loss = criterion(output, train_tgt_seq.view(-1))
            # l2 regularization
            for name, param in self.encoder.named_parameters():
                if 'bias' not in name:
                    pass#loss += (0.5 * 1e-7 * torch.sum(torch.pow(param, 2)))
            loss.backward()
            optimizer.step()
            abort = False

            #print(loss.item())
            if alg=='lr_scan' and epoch%10==0:
                lr = 1e-7*2**(epoch//10)
                for g in optimizer.param_groups:
                    g['lr'] = lr

            if epoch%30 == 0:
                #print(loss.item())
                #print(encoder.predict([[u'ˀnšym', u'nršmym', u'twpˁh']]), '\n')

                with torch.no_grad():
                    pred, _ = self.encoder(valid_inp_seq, valid_mask, self.encoder.init_hidden(self.batch_size))
                    val_loss = criterion(pred, valid_tgt_seq.view(-1))
                    #scheduler.step(loss.item(),epoch)
                #print(epoch, loss.item(), val_loss.item())
                if val_loss.item()<min_val_loss[0]:
                    min_val_loss = (val_loss.item(), epoch)
                    print(loss.item(), min_val_loss)
                    print(encoder.predict([[u'ˀnšym', u'nršmym', u'twpˁh']]), '\n')

                if epoch>(min_val_loss[1]+500):  # no decrease for n epochs
                    break

            if abort:
                return

        #print(loss.item(), min_val_loss)

    def predict(self, pred_set):
        res_set = []
        for sentence in pred_set:
            res_set.append([])
            for word in sentence:
                res_set[-1].append([])
                input = np.zeros((len(word), self.word_emb_size+self.emb_in_size*self.max_len))
                mask = np.zeros((len(word), self.emb_in_size*self.max_len))
                input[:,:self.word_emb_size] = self.word_embedding.GetEmbedding(word)
                for j, ch in enumerate(word):
                    input[j,self.word_emb_size+j*self.emb_in_size+self.input_char2int[ch]] = 1
                    if ch!=' ':
                        mask[j,j*self.emb_in_size:(j+1)*self.emb_in_size] = 1
                input = input.reshape((1,input.shape[0],input.shape[-1]))
                mask = mask.reshape((1,mask.shape[0],mask.shape[-1]))
                input = torch.from_numpy(input).float().to(self.device)
                mask = torch.from_numpy(mask).float().to(self.device)
                with torch.no_grad():
                    pred, _ = self.encoder(input, mask, self.encoder.init_hidden(input.size(0)))
                    pred = nn.Softmax(dim=-1)(pred)
                    y = pred.detach().to('cpu').numpy()#
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
                #ch_out = [x for x in re.split('[' + units[0] + ']', units[1]) if len(x) > 0]
                #sets[1].update(ch_out)
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
    algs = ['adagrad', 'rprop', 'adamax', 'adamw', 'asgd', 'rmsprop']
    lrs = np.logspace(-3, -1, 50)
    #sys.stdout = open('log.txt', 'w', encoding='utf-8')
    #while True:
    #    rand_state = np.random.RandomState()
    #    n_layers = rand_state.randint(1,4)
    #    dropout = 0#rand_state.random_sample()*0.25#
    #    lr = 1e-2#lrs[rand_state.randint(0,50)]#
    #    alg = 'adamax'#algs[rand_state.randint(0,len(algs))]
    #    print('-----------------------------------------')
    #    print(n_layers, dropout, lr, alg)
    #    print('-----------------------------------------')
    #    encoder = Encoder(n_layers, dropout)
    #    encoder.prep_model().shuffle().split(0.01).train(50000, lr, alg)
    #    print(encoder.predict([[u'ˀnšym',u'nršmym',u'twpˁh']]), '\n')



    encoder = Encoder(1, 0)
    encoder.prep_model().shuffle().split(0.05).train(50000, alg='adamax')
    #print(encoder.predict([[u'ˀnšym',u'nršmym',u'twpˁh']]))
