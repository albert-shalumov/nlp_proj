import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import codecs
import time
import numpy as np
import re

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = [nn.GRU(self.hidden_size, self.hidden_size) for i in range(n_layers)]

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for gru in self.gru:
            output, hidden = gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

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
        self.emb_size = len(unit_dict_sets[0])+1

        # Init lookup tables
        self.input_int2char, self.input_char2int = Encoder.CreateDictionaries(unit_dict_sets[0])
        self.output_int2char, self.output_char2int = Encoder.CreateDictionaries(unit_dict_sets[1])


        # Create encoder
        self.encoder = EncoderRNN(len(unit_dict_sets[0])+1, self.hid_dim, self.n_layers).to(self.device)

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
                X.append(w)
                Y.append(ch_out[j])

        # Convert to numpy arrays
        self.X = np.zeros((len(X), self.max_len*self.emb_size))
        self.Y = np.zeros(len(Y))
        for i in range(len(X)):
            for j,ch in enumerate(X[i]):
                self.X[i,j*self.emb_size+self.input_char2int[ch]] = 1
            self.Y[i] = self.output_char2int[Y[i]]

        return self

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
    encoder = Encoder(256, 2)
    encoder.prep_model()#.shuffle().split(0).train()
    #print(encoder.predict([[u'ˀnšym',u'nršmym']]))
