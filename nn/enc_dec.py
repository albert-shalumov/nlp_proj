import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import codecs
import numpy as np

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


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p, max_length, n_layers=1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = [nn.GRU(hidden_size, hidden_size) for i in range(n_layers)]
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        for gru in self.gru:
            output, hidden = gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


'''
Class for Encoder-Decoder RNN method.
Most functions return self, therefore calls can be chained: nn.shuffle().split().train() etc.
References:
1) https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
2) https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
3) https://github.com/jadore801120/attention-is-all-you-need-pytorch

Note:
-----
Current implementation is quite basic, without optimization.

Params:
-------
hidden_dim - size of hidden dimension
n_layers - tuple (encoder layers, decoder layers)
stage - 1 for vowel&syllab.,
        2 for romanization
device - force device ("cpu" to use cpu)
'''
class EncoderDecoder:
    def __init__(self, hidden_dim=250, n_layers=(1,1), stage=1, device=None):
        self.hid_dim = hidden_dim
        self.n_layers = n_layers
        self.stage = stage
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
                self.proc_units[-1] += [(split_line[2], split_line[3] if self.stage == 1 else split_line[4])]
                self.proc_units.append([])
        if len(self.proc_units[-1]) == 0:
            self.proc_units.pop(-1)

        # Calculate max unit length, dictionaries
        self.max_len = EncoderDecoder.CalcMaxLen(self.proc_units)+5
        unit_dict_sets = EncoderDecoder.BuildCharSets(self.proc_units)

        # Add EOW symbol
        unit_dict_sets[0].add('<e>')
        unit_dict_sets[1].add('<e>')

        # Init lookup tables
        self.input_int2char, self.input_char2int = EncoderDecoder.CreateDictionaries(unit_dict_sets[0])
        self.output_int2char, self.output_char2int = EncoderDecoder.CreateDictionaries(unit_dict_sets[1])

        # Create encoder, decoder
        self.encoder = EncoderRNN(len(unit_dict_sets[0])+1, self.hid_dim, self.n_layers[0]).to(self.device)
        self.decoder = AttnDecoderRNN(self.hid_dim, len(unit_dict_sets[1])+1, 0.1, self.max_len, self.n_layers[1]).to(self.device)

        return self


    def shuffle(self, seed=None):
        self.inds = np.arange(len(self.proc_units))
        np.random.seed(seed)
        np.random.shuffle(self.inds)
        return self

    def split(self, valid_ratio=0.1):
        num_train = int(len(self.inds)*(1-valid_ratio))
        self.train_inds = self.inds[:num_train]
        self.valid_inds = None if valid_ratio == 0 else self.inds[num_train:]
        return self

    def train(self):
        # Local constants
        teacher_forcing_ratio = 0.7
        lr = 1e-2
        enc_optim = optim.SGD(self.encoder.parameters(), lr=lr)
        dec_optim = optim.SGD(self.decoder.parameters(), lr=lr)
        enc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(enc_optim, factor=0.5)
        dec_scheduler = optim.lr_scheduler.ReduceLROnPlateau(dec_optim, factor=0.5)
        criterion = nn.NLLLoss()

        training_pairs = [self._tensor(self.proc_units[i][0]) for i in self.train_inds]
        #validation_pairs = [self._tensor(self.proc_units[i][0]) for i in self.valid_inds]

        for epoch in range(500):
            loss_sum = 0
            for iter in range(len(training_pairs)):  # single sample
                training_pair = training_pairs[iter]
                input_tensor = training_pair[0]
                target_tensor = training_pair[1]

                encoder_hidden = self.encoder.initHidden()
                enc_optim.zero_grad()
                dec_optim.zero_grad()
                input_length = input_tensor.size(0)
                target_length = target_tensor.size(0)
                encoder_outputs = torch.zeros(self.max_len, self.encoder.hidden_size, device=self.device)

                loss = 0

                for ei in range(input_length):
                    encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                    encoder_outputs[ei] = encoder_output[0, 0]

                decoder_input = torch.tensor([[EncoderDecoder.START_SYMB]], device=self.device)
                decoder_hidden = encoder_hidden

                if np.random.random() < teacher_forcing_ratio: # use teacher forcing - use target instead of predicted. faster training
                    # Teacher forcing: Feed the target as the next input
                    for di in range(target_length):
                        decoder_output, decoder_hidden, decoder_attention = self.decoder(
                            decoder_input, decoder_hidden, encoder_outputs)
                        loss += criterion(decoder_output, target_tensor[di])
                        decoder_input = target_tensor[di]  # Teacher forcing
                else:
                    # Without teacher forcing: use its own predictions as the next input
                    for di in range(target_length):
                        decoder_output, decoder_hidden, decoder_attention = self.decoder(
                            decoder_input, decoder_hidden, encoder_outputs)
                        topv, topi = decoder_output.topk(1)
                        decoder_input = topi.squeeze().detach()  # detach from history as input

                        loss += criterion(decoder_output, target_tensor[di])
                        if decoder_input.item() == self.output_char2int['<e>']:
                            break

                loss.backward()
                enc_optim.step()
                dec_optim.step()

                loss_value = loss.item()/target_length
                loss_sum += loss_value
            print(loss_sum/len(training_pairs))
            enc_scheduler.step(loss_sum/len(training_pairs))
            dec_scheduler.step(loss_sum/len(training_pairs))

        return self

    def predict(self, pred_set):
        res_set = []
        with torch.no_grad():
            for sentence in pred_set:
                res_set.append([])
                for word in sentence:
                    res_set[-1].append([])
                    input_tensor = self._tensor((word,))
                    input_length = input_tensor.size()[0]
                    encoder_hidden = self.encoder.initHidden()
                    encoder_outputs = torch.zeros(self.max_len, self.encoder.hidden_size, device=self.device)

                    for ei in range(input_length):
                        encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                        encoder_outputs[ei] += encoder_output[0, 0]

                    decoder_input = torch.tensor([[EncoderDecoder.START_SYMB]], device=self.device)
                    decoder_hidden = encoder_hidden

                    for di in range(self.max_len):
                        decoder_output, decoder_hidden, decoder_attention = self.decoder(
                            decoder_input, decoder_hidden, encoder_outputs)
                        topv, topi = decoder_output.data.topk(1)
                        if topi.item() == self.output_char2int['<e>']:
                            break
                        else:
                            res_set[-1][-1].append(self.output_int2char[topi.item()])
                        decoder_input = topi.squeeze().detach()
                    res_set[-1][-1] = ''.join(res_set[-1][-1])

        return res_set

    def _tensor(self, unit_pair):
        w = [self.input_char2int[ch] for ch in unit_pair[0]]+[self.input_char2int['<e>']]
        input_tensor = torch.tensor(w, dtype=torch.long, device=self.device).view(-1, 1)
        if len(unit_pair)==2:
            w = [self.output_char2int[ch] for ch in unit_pair[1]]+[self.output_char2int['<e>']]
            target_tensor = torch.tensor(w, dtype=torch.long, device=self.device).view(-1, 1)
            return (input_tensor, target_tensor)
        else:
            return input_tensor

    START_SYMB = 0

    @staticmethod
    def BuildCharSets(proc_units):
        sets = [set(), set()]
        for unit_list in proc_units:
            for unit in unit_list:
                units = list(unit)
                sets[0].update(list(units[0]))
                sets[1].update(list(units[1]))
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
        int2char = dict(enumerate(list(unit_set)))
        char2int = {char: ind for ind, char in int2char.items()}
        if reserve_zero:  # zero reserved - swap out
            zero_ch = int2char[0]
            int2char[len(unit_set)] = zero_ch
            int2char.pop(0)
            char2int[zero_ch] = len(unit_set)
        return int2char, char2int

if __name__=='__main__':
    enc_dec = EncoderDecoder(256, (2,2), 1)
    enc_dec.prep_model().shuffle().split(0).train()
    #print(enc_dec.predict([[u'ˀnšym',u'nršmym']]))
