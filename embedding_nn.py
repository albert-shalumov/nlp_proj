import numpy as np
import codecs
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Embedding:
    def __init__(self):
        self.mapping = np.identity(len(Embedding.ARNON_CHARS))

    def learn(self, file='../data/HaaretzOrnan_annotated.txt'):
        prob_distr = self._calc_corpus_dist(file)
        dist = self._calc_dist(prob_distr)

        class embedder(nn.Module):
            def __init__(self, size_in, size_out):
                super(embedder, self).__init__()
                emb_dim_size = size_out
                self.fc1 = nn.Linear(size_in, size_in)
                self.act1 = nn.PReLU()
                self.fc2 = nn.Linear(size_in, emb_dim_size) # embedding

                self.act2 = nn.PReLU()
                self.fc3 = nn.Linear(emb_dim_size, emb_dim_size)
                self.act3 = nn.PReLU()
                self.fc4 = nn.Linear(emb_dim_size, size_in)
                self.softmax = nn.Softmax(dim=-1)

            def forward(self, input):
                out1 = self.fc2((self.act1(self.fc1(input))))
                out2 = torch.log2(self.softmax(self.fc4(self.act3(self.fc3(self.act2(out1)))))+1e-7)
                return out1, out2

        criterion2 = nn.NLLLoss()
        emb_net = embedder(self.mapping.shape[0], 8)
        inp = torch.from_numpy(self.mapping).float().to('cpu')
        tgt_out1 = torch.from_numpy(dist).float().to('cpu')
        tgt_out2 = torch.from_numpy(np.argmax(self.mapping,-1)).long().to('cpu')

        d = Variable()
        l = Variable()
        reg = Variable()
        optimizer = optim.Adam(emb_net.parameters(), lr=1e-2)
        best_loss = np.inf
        best_mapping = None
        best_loss = np.inf
        for i in range(5000):
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            out1, out2 = emb_net(inp)
            d = 0
            l = 0
            reg = 0
            for j in range(self.mapping.shape[0]):
                for k in range(j+1, self.mapping.shape[0]):
                    tmp = out1[j,:]-out1[k,:]
                    est_dist_sqr = torch.t(tmp).matmul(tmp)
                    d += (est_dist_sqr-tgt_out1[j,k]**2)**2

            distr_dist = 100*d
            dec_err = criterion2(out2, tgt_out2)

            #for name, param in emb_net.named_parameters():
            #    if 'bias' not in name:
            #        reg += (0.5 * 1e-3 * torch.sum(torch.pow(param, 2)))
#
            loss = distr_dist + dec_err+reg
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if i%10==0:
                    #for name, param in emb.named_parameters():
                    #    if 'fc' in name:
                    #        #print(name, torch.min(param), torch.max(param))
                    print(i, d.item(), dec_err.item(), loss.item())
                    if loss.item()<best_loss:
                        best_loss=loss.item()
                        best_mapping = out1.detach().numpy()
        self.mapping = best_mapping
        print("Best loss:",best_loss)

    def save(self, file):
        np.save(file, self.mapping)

    def load(self, file):
        self.mapping = np.load(file)

    def len(self):
        return self.mapping.shape[1]

    def __getitem__(self, ch):
        return self.mapping[Embedding.ARNON_CHARS_IDX[ch],:]

    VOWELS = [u'a', u'e', u'u', u'i', u'o', u'*']
    VOWELS_IDX = {x: i for i, x in enumerate(VOWELS)}
    ARNON_CHARS = [u'\u02c0', u'b', u'g', u'd', u'h', u'w', u'z', u'\u1e25', u'\u1e6d', u'y',
                   u'k', u'k', u'l', u'm', u'm', u'n', u'n', u's', u'\u02c1', u'p', u'p', u'\u00e7',
                   u'\u00e7', u'q', u'r', u'\u0161', u't', u' ']
    ARNON_CHARS_IDX = {x: i for i, x in enumerate(ARNON_CHARS)}

    @staticmethod
    def _calc_corpus_dist(file):
        prob_distr = np.zeros((len(Embedding.ARNON_CHARS), len(Embedding.VOWELS)))
        with codecs.open(file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                if line.startswith(u'#') or len(line) == 0:
                    continue
                w = line.split(u' ')[3]
                w = w.replace(u'-', u'')
                for i in range(len(w) // 2):
                    prob_distr[Embedding.ARNON_CHARS_IDX[w[i * 2]], Embedding.VOWELS_IDX[w[i * 2 + 1]]] += 1
        prob_distr += 0.3  # smoothing
        prob_distr = prob_distr / np.sum(prob_distr, 1)[:, np.newaxis]  # prob distrib
        return prob_distr

    @staticmethod
    def _calc_dist(prob_distr):
        dist = np.zeros((prob_distr.shape[0], prob_distr.shape[0]))
        for i in range(dist.shape[0]):
            for j in range(dist.shape[0]):
                if i==j:
                    continue
                dist[i,j] = wasserstein_distance(prob_distr[i,:],prob_distr[j,:])
        return dist

    @staticmethod
    def _calc_err(dist, eigvec, eigval, n):
        new_rep = np.empty((n,dist.shape[0]))

        if n>=dist.shape[0] or eigval[n,n]<=0:
            return np.nan

        # Fill new representation
        for i in range(dist.shape[0]):
            for j in range(n):
                new_rep[j,i] = eigvec[i,j]*np.sqrt(eigval[j,j])

        # calc new distances
        new_dist = np.zeros((dist.shape[0], dist.shape[0]))
        for i in range(new_dist.shape[0]):
            for j in range(new_dist.shape[0]):
                if i==j:
                    continue
                new_dist[i,j] = np.linalg.norm(new_rep[:,i]-new_rep[:,j])

        avg_err = np.sum(np.sum(np.abs(new_dist-dist)))/2

        return avg_err


if __name__ == '__main__':
    emb = Embedding()
    emb.learn('data/HaaretzOrnan_annotated.txt')
    emb.save('emb_model_nn')
    for i in range(emb.mapping.shape[0]):
        print(emb.mapping[i], "'"+Embedding.ARNON_CHARS[i]+"'")
    print(emb.len())
    #print(emb['t'])
