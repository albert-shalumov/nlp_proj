import torch
import torch.nn as nn
import torch.optim as optim
import codecs
import numpy as np
from itertools import product
from embedding_mds import Embedding as Embedding_MDS
from embedding_nn import Embedding as Embedding_NN
import metrics
import os
import sys

class EncoderRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout, device):
        super(EncoderRNN, self).__init__()
        self.bidir = 1
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.drop_prob = dropout
        self.device=device
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=(self.drop_prob if self.n_layers>1 else 0), batch_first=True, bidirectional=bool(self.bidir))
        self.fc = nn.Linear((2 if self.bidir else 1) * hidden_dim, self.output_size)

    def forward(self, input, mask, hidden):
        batch_size,seq_len, _ = input.size()
        out, _ = self.lstm(input*mask, hidden)
        out = out.contiguous().view(-1,out.shape[2])
        out = self.fc(out)
        out = out.view(batch_size, seq_len, self.output_size)
        return out, None

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers*(2 if self.bidir else 1), batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.n_layers*(2 if self.bidir else 1), batch_size, self.hidden_dim).to(self.device)) # (h_0, c_0) - for lstm
        return hidden

class Encoder:
    def __init__(self, config):
        self.hid_dim = config['hidden_dim']
        self.n_layers = config['n_layers']
        self.bidir = True
        self.dropout = 0
        self.max_len = config['win_len']
        if 'device' not in config.keys():
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = config['device']

        if config['embedding'] == 'nn':
            self.emb = Embedding_NN()
            self.emb.load('emb_model_nn.npy')
        elif config['embedding'] == 'mds':
            self.emb = Embedding_MDS()
            self.emb.load('emb_model_mds.npy')
        else: # 'one-hot'
            self.emb = Embedding_MDS()

    def prep_model(self, file='data/HaaretzOrnan_annotated.txt'):
        self.proc_units = [[]]
        with codecs.open(file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                if line.startswith(u'#'):
                    continue
                if len(line) == 0:
                    if len(self.proc_units[-1])>0:
                        self.proc_units.append([])
                    continue
                split_line = line.split(u' ')
                self.proc_units[-1] += [(split_line[2], split_line[3])]
        if len(self.proc_units[-1]) == 0:
            self.proc_units.pop(-1)

        # Calculate max unit length, dictionaries
        unit_dict_sets = Encoder.BuildCharSets(self.proc_units)
        unit_dict_sets[0].add(' ')
        unit_dict_sets[1].update(list('euioa*'))
        self.emb_in_size = self.emb.len()
        self.emb_out_size = len(unit_dict_sets[1])+1

        # Init lookup tables
        self.input_int2char, self.input_char2int = Encoder.CreateDictionaries(unit_dict_sets[0])
        self.output_int2char, self.output_char2int = Encoder.CreateDictionaries(unit_dict_sets[1])

        # Create encoder
        self.encoder = EncoderRNN(self.emb_in_size, self.emb_out_size,self.hid_dim, self.n_layers,self.dropout, self.device).to(self.device)

        # Build sliding window sequences
        X = []
        Y = []
        for i in range(len(self.proc_units)):
            stc_in = ' '.join([x[0] for x in self.proc_units[i]])
            stc_out = ' '.join([x[1].replace('-','')[1::2] for x in self.proc_units[i]])

            for j, ch in enumerate(stc_in):
                w = stc_in[:j+1]
                if len(w)<self.max_len:
                    w = w.rjust(self.max_len)
                else:
                    w = w[-self.max_len:]
                if w[-1] != ' ':
                    X.append(w)
                    Y.append(stc_out[j])

        # Convert to numpy arrays
        self.X = np.zeros((len(X), self.max_len, self.emb_in_size))
        self.Y = np.zeros(len(Y))
        self.Mask = np.zeros((len(X), self.max_len, self.emb_in_size))
        for i in range(len(X)):
            for j,ch in enumerate(X[i]):
                self.X[i, j, :] = self.emb[ch]
                if ch!=' ':
                    self.Mask[i, j,:] = 1

            self.Y[i] = self.output_char2int[Y[i]]

        return self

    def shuffle(self, seed=None):
        self.inds = np.arange(self.X.shape[0])
        np.random.seed(seed)
        np.random.shuffle(self.inds)
        self.X = self.X[self.inds,...]
        self.Y = self.Y[self.inds]
        self.Mask = self.Mask[self.inds, ...]
        return self

    def split(self, valid_ratio=0.1, train_words=9999999):
        assert valid_ratio>0

        train_smpls = int(self.X.shape[0]*(1-valid_ratio))
        valid_smpls = self.X.shape[0]-train_smpls
        act_train_smpls = min(train_smpls, train_words)

        self.train_X = self.X[:act_train_smpls, np.newaxis, ...]
        self.train_Y = self.Y[:act_train_smpls,np.newaxis]
        self.train_Mask = self.Mask[:act_train_smpls, np.newaxis, ...]

        self.valid_X = self.X[train_smpls:train_smpls+valid_smpls, np.newaxis, ...]
        self.valid_Y = self.Y[train_smpls:train_smpls+valid_smpls, np.newaxis]
        self.valid_Mask = self.Mask[train_smpls:train_smpls+valid_smpls, np.newaxis, ...]

        return self


    def train(self, epochs=10, lr=1e-2, alg = 'adamw'):
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 1.0/2.0,10,cooldown=1)
        criterion = nn.CrossEntropyLoss()

        train_inp_seq = torch.from_numpy(self.train_X).float().to(self.device)
        train_tgt_seq = torch.Tensor(self.train_Y).long().to(self.device)
        train_mask = torch.Tensor(self.train_Mask).to(self.device)
        valid_inp_seq = torch.from_numpy(self.valid_X).float().to(self.device)
        valid_tgt_seq = torch.Tensor(self.valid_Y).long().to(self.device)
        valid_mask = torch.Tensor(self.valid_Mask).to(self.device)

        min_val_loss = (9999, 0)
        best_model = None
        hidden = self.encoder.init_hidden(self.train_X.shape[0])
        for epoch in range(1, epochs+1):
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            output, _ = self.encoder(train_inp_seq[:,0,...], train_mask[:,0,...], hidden)
            loss = criterion(output[:,-1,:], train_tgt_seq[:,0,...].view(-1))
            # l2 regularization
            #for name, param in self.encoder.named_parameters():
            #    if 'bias' not in name:
            #        pass#loss += (0.5 * 1e-7 * torch.sum(torch.pow(param, 2)))
            loss.backward()
            optimizer.step()

            abort = False

            if epoch%10 == 0:
                #print(loss.item())
                #print(encoder.predict([[u'ˀnšym', u'nršmym', u'twpˁh']]), '\n')
                with torch.no_grad():
                    val_loss = 0
                    pred, _ = self.encoder(valid_inp_seq[:,0,...], valid_mask[:,0,...], self.encoder.init_hidden(self.valid_X.shape[0]))
                    val_loss = criterion(pred[:,-1,:], valid_tgt_seq[:,0,...].view(-1))
                    scheduler.step(loss.item())
                    #print(epoch, loss.item(), val_loss.item())

                if val_loss.item()<min_val_loss[0]:
                    min_val_loss = (val_loss.item(), epoch)
                    #print(loss.item(), min_val_loss)
                    #print(self.predict([[u'ˀnšym', u'nršmym', u'twpˁh']]), '\n')
                    best_model = self.encoder.state_dict()

                if epoch>(min_val_loss[1]+500):  # no decrease for n epochs
                    break

            if abort:
                return self

        # Restore best model and evaluate it
        self.encoder.load_state_dict(best_model)
        self.encoder.eval()

        #print(min_val_loss)
        return self

    def eval(self):
        self.encoder.eval()
        conf_mat = np.zeros((self.emb_out_size, self.emb_out_size))
        valid_inp_seq = torch.from_numpy(self.valid_X).float().to(self.device)
        valid_tgt_seq = torch.Tensor(self.valid_Y).long().to(self.device)
        valid_mask = torch.Tensor(self.valid_Mask).to(self.device)

        # Fill confusion matrix
        pred, _ = self.encoder(valid_inp_seq[:, 0, ...], valid_mask[:, 0, ...], self.encoder.init_hidden(self.valid_X.shape[0]))
        pred = nn.Softmax(dim=-1)(pred)
        y = pred.detach().to('cpu').numpy()  #
        predicted = np.argmax(y, axis=-1)

        for i in range(predicted.shape[0]):
            conf_mat[int(predicted[i][-1]), int(self.valid_Y[i])] += 1

        return conf_mat[:-1,:-1]

    def predict(self, pred_set):
        self.encoder.eval()
        res_set = []
        for sentence in pred_set:
            stc_in = ' '.join(sentence)
            stc_out = u''

            # Init sliding window
            for j in range(len(stc_in)):
                stc_out += stc_in[j]

                w = stc_in[:j+1]
                if len(w)<self.max_len:
                    w = w.rjust(self.max_len)
                else:
                    w = w[-self.max_len:]

                # Init input tensor
                input = np.zeros((1, 1, self.max_len, self.emb_in_size))
                mask = np.zeros((1, 1, self.max_len, self.emb_in_size))
                for k, ch in enumerate(w):
                    input[0, 0, k, :] = self.emb[ch]
                    if ch != ' ':
                        mask[0,0, k, :] = 1
                hidden = self.encoder.init_hidden(1)
                input = torch.from_numpy(input).float().to(self.device)
                mask = torch.from_numpy(mask).float().to(self.device)
                pred, hidden = self.encoder(input[:, 0, ...], mask[:, 0, ...], hidden)
                if stc_out[-1] != ' ':
                    pred = nn.Softmax(dim=-1)(pred)
                    y = pred.detach().to('cpu').numpy()  #
                    inds = list(np.argmax(y, axis=-1))
                    stc_out += self.output_int2char[inds[0][-1]]
            res_set .append(stc_out.split())

        return res_set

    def save(self, file):
        torch.save(self.encoder.state_dict(), file)

    def load(self, file):
        self.encoder.load_state_dict(torch.load(file))
        return self

    @staticmethod
    def BuildCharSets(proc_units):
        sets = [set(), set()]
        for unit_list in proc_units:
            for unit in unit_list:
                sets[0].update(list(unit[0]))
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


def search_hparams():
    verbose = False
    with open('rnn_res.csv','a') as f:
        num_layers = [1,2,3,4,5,6]
        n_hidden = [32,64,128]
        emb = ['one-hot', 'mds', 'nn']
        win_len = [4]
        for config_tuple in product(num_layers, n_hidden, emb, win_len):
            config = {'n_layers':config_tuple[0], 'hidden_dim':config_tuple[1], 'embedding':config_tuple[2], 'win_len':config_tuple[3]}
            if 'conf_mat' in locals():
                del conf_mat
            for i in range(5):
                rnn = Encoder(config)
                if 'conf_mat' in locals():
                    conf_mat += rnn.prep_model().shuffle(None).split(0.05).train(epochs=10000, lr=1e-2, alg='adamw').eval()
                else:
                    conf_mat = rnn.prep_model().shuffle(None).split(0.05).train(epochs=10000,  lr=1e-2, alg='adamw').eval()
            res_str = '{};'.format(config)
            print("Configuration = {}: ".format(config))
            precision, recall = metrics.MicroAvg(conf_mat)
            f1 = metrics.Fscore(precision, recall, 1)
            res_str += '{};'.format(f1)
            print('MicroAvg:', precision, recall, f1)
            precision, recall = metrics.MacroAvg(conf_mat)
            f1 = metrics.Fscore(recall, precision, 1)
            res_str += '{};'.format(f1)
            print('MacroAvg:', precision, recall, f1)
            acc = metrics.AvgAcc(conf_mat)
            res_str += '{};'.format(acc)
            print('AvgAcc:', acc)
            f.write(res_str + '\n')
            conf_mat = metrics.NormalizeConfusion(conf_mat)
            if verbose:
                print('ConfMat:\n', np.array_str(conf_mat, max_line_width=300, precision=4))
                print('----------------------------------------------')
            f.flush()

def check_seeds():
    config = {'n_layers': 3, 'hidden_dim': 32, 'embedding': 'mds', 'win_len': 4}
    print("seed, accuracy")
    for seed in range(11):
        if 'conf_mat' in locals():
            del conf_mat
        for i in range(5):
            rnn = Encoder(config)
            if 'conf_mat' in locals():
                conf_mat += rnn.prep_model().shuffle(seed).split(0.05).train(epochs=10000, lr=1e-2, alg='adamw').eval()
            else:
                conf_mat = rnn.prep_model().shuffle(seed).split(0.05).train(epochs=10000, lr=1e-2, alg='adamw').eval()
        acc = metrics.AvgAcc(conf_mat)
        print(seed, acc)

def print_usage():
    print("Usage:")
    print("rnn.py [search/seeds]")
    print("search - searches for best configuration")
    print("seeds - checks various seeds")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print_usage()
    elif sys.argv[1] == 'search':
        search_hparams()
    elif sys.argv[1] == 'seeds':
        check_seeds()
    else:
        print_usage()
