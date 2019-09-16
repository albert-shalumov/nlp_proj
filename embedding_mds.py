import numpy as np
import codecs
from scipy.stats import wasserstein_distance

'''
Use multi-dimensional scaling to map consonant to fewer dimension.
Distance between consonants calculated as Wasserstein distance(=Earth Mover Distance) between vowel co-appearance distribution 
'''
class Embedding:
    def __init__(self):
        self.mapping = np.identity(len(Embedding.ARNON_CHARS)) # default to 1-hot encoding

    '''
    Ref.:
    https://www.stat.pitt.edu/sungkyu/course/2221Fall13/lec8_mds_combined.pdf
    '''
    def learn(self, file='../data/HaaretzOrnan_annotated.txt'):
        prob_distr = Embedding._calc_corpus_dist(file) # calculate probability distribution with smoothing
        dist = Embedding._calc_dist(prob_distr)
        n = prob_distr.shape[0]
        A = -0.5 * dist * dist
        H = np.identity(n) - np.ones((n, n)) / n
        B = np.linalg.multi_dot([H, A, H])
        V, D = np.linalg.eig(B) # eigenvalues and eigenvectors
        V, D = np.real(V), np.real(D)
        ids = np.argsort(V)[::-1] # get sorting indices, in decreasing size
        eig_mat = np.diag(V[ids]) # diagonal matrix of eigenvalues
        D = D[:, ids]
        err_mat = np.zeros(n)+np.inf
        for i in range(1, prob_distr.shape[0]+1):
            err = Embedding._calc_err(dist, D, eig_mat, i) # calculate reprojection error
            if np.isfinite(err):
                err_mat[i] = err
        min_ind = np.argmin(err_mat) # find number of dimenstion minimizing reprojection error
        self.mapping = np.empty((dist.shape[0], min_ind))
        for i in range(dist.shape[0]): # calculate new vectors
            for j in range(int(min_ind)):
                self.mapping[i,j] = D[i,j]*np.sqrt(eig_mat[j,j])


    def save(self, file):
        np.save(file, self.mapping)

    def load(self, file):
        self.mapping = np.load(file)

    def len(self):
        return self.mapping.shape[1]

    def __getitem__(self, ch):
        return self.mapping[Embedding.ARNON_CHARS_IDX[ch], :]

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
    emb.save('emb_model_mds')
    for i in range(emb.mapping.shape[0]):
        print(emb.mapping[i], "'"+Embedding.ARNON_CHARS[i]+"'")
    print(emb.len())
    #print(emb['t'])
