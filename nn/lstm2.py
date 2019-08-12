import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from embedding import Embedding,FastTextEmb
import codecs
import re
import numpy as np
import time
import random

PRED_STAGE = 1

def BuildCharSets(sents):
    sets = [set() for i in range(len(sents[0][0]))]
    for sent in sents:
        for word in sent:
            word_list = list(word)
            for set_id in range(len(word_list)):
                sets[set_id].update(list(word_list[set_id]))
    return sets

def CalcMaxLen(sents):
    max_len = 0
    for sent in sents:
        for word in sent:
            word_list = list(word)
            for set_id in range(len(word_list)):
                max_len = max(max_len, len(word_list[set_id]))
    return max_len

def TensorsFromWordPair(word):
    w = [dicts[0][1][ch] for ch in word[0][0]] + [dicts[0][1]['<e>']]
    input_tensor = torch.tensor(w, dtype=torch.long, device=device).view(-1, 1)

    w = [dicts[1][1][ch] for ch in word[0][1]] + [dicts[1][1]['<e>']]
    target_tensor = torch.tensor(w, dtype=torch.long, device=device).view(-1, 1)
    return (input_tensor, target_tensor)

sentences = [[]]
'''
# For now ignore sentence context
with codecs.open('../data/HaaretzOrnan_annotated.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        if line.startswith(u'#'):
            continue
        if len(line) == 0:
            if len(sentences[-1])>0:
                sentences.append([])
            continue
        split_line = line.split(u' ')
        sentences[-1].append((split_line[2],split_line[3],split_line[4]))
if len(sentences[-1])==0:
    sentences.pop(-1)

# Add single word sentences
for i in range(len(sentences)):
    sentences += [[x] for x in sentences[i]]
'''
with codecs.open('../data/HaaretzOrnan_annotated.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        if line.startswith(u'#'):
            continue
        if len(line) == 0:
            continue
        split_line = line.split(u' ')
        sentences[-1] += [(split_line[2],split_line[3] if PRED_STAGE==1 else split_line[4])]
        sentences.append([])
if len(sentences[-1])==0:
    sentences.pop(-1)
max_len = CalcMaxLen(sentences)+5
START_SYMB=0


sets = BuildCharSets(sentences)
for set in sets: # custom chars
    set.update(['<e>'])

dicts = []
for set in sets:
    int2char = dict(enumerate(list(set)))
    char2int = {char: ind for ind, char in int2char.items()}
    zero_ch = int2char[0] # zero reserved - swap out
    int2char[len(set)]=zero_ch
    char2int[zero_ch] = len(set)
    dicts.append((int2char, char2int))





'''
from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=max_len):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

teacher_forcing_ratio = 0.5
device = 'cpu'

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=max_len):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[START_SYMB]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == dicts[1][1]['<e>']:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [TensorsFromWordPair(random.choice(sentences)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (iter, iter / n_iters * 100, print_loss_avg))

        #if iter % plot_every == 0:
        #    plot_loss_avg = plot_loss_total / plot_every
        #    plot_losses.append(plot_loss_avg)
        #    plot_loss_total = 0#

hidden_size = 256
encoder1 = EncoderRNN(len(sets[0])+1, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, len(sets[1])+1).to(device)

trainIters(encoder1, attn_decoder1, 7500, print_every=5)