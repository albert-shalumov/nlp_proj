import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from embedding import Embedding,FastTextEmb

class Net:
    def __init__(self, embeddding):
        self.output_ch = list('qwertyuiopasdfghjklzxcvbnm-*') # * = End Of Word
        self.output_dict = {x:i for i,x in enumerate(self.output_ch)}

        self.embd = embeddding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = nn.LSTM(150, 15)
        self.encoder = nn.LSTM(150, 15)

    def forward(self, input, hid_enc, hid_dec):
        out_enc, hid_enc = self.encoder(self.embd[input], hid_enc)



        return output, hidden
    def Train(self, train_stc, valid_stc):

        pass


if __name__ == '__main__':
    #emb = FastTextEmb()
    #emb.Load()
    #LSTM(emb)