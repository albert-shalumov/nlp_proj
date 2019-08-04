from nltk.tag.crf import CRFTagger
import codecs
import numpy as np
import metrics

class CRF:
    def __init__(self):
        pass

    def prep_data(self, file='data/HaaretzOrnan_annotated.txt'):
        self.data = []
        # print('Preparing data')
        with codecs.open(file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                if line.startswith(u'#') or len(line) == 0:
                    continue
                w = line.split(u' ')[3]
                w = w.replace(u'-', u'')
                self.data.append(list(zip(list(w[::2]), list(w[1::2]))))
        return self

    def shuffle(self, seed=None):
        # Shuffle based on seed
        inds = np.arange(len(self.data))
        np.random.seed(seed)
        np.random.shuffle(inds)
        self.data=[self.data[i] for i in inds]
        return self

    def split(self, valid_ratio=0.1):
        # Split to train and validation based on ratio.
        # If ratio is 0 use all data for training
        num_train = int(len(self.data)*(1-valid_ratio))
        self.train_set = self.data[:num_train]
        self.valid_set = None if valid_ratio==0 else self.data[num_train:]
        return self

    def train(self):
        self.model = CRFTagger(CRF._extract_ftr, verbose=False,
                       training_opt={"c1": 0, "c2": 0, "num_memories": 50, "epsilon": 1e-7, "delta": 1e-8})
        self.model.train(self.train_set, 'word_crf_model')
        return self

    def eval(self):
        conf_mat = np.zeros((len(CRF.VOWELS), len(CRF.VOWELS)))
        valid_word_cons = [[x[0] for x in w] for w in self.valid_set]
        valid_word_vowel = [[x[1] for x in w] for w in self.valid_set]
        predicted = self.model.tag_sents(valid_word_cons)
        predicted = [[x[1] for x in w] for w in predicted]
        for w_ind in range(len(predicted)):
            for vow_ind, pred_vow in enumerate(predicted[w_ind]):
                conf_mat[self.VOWELS_IDX[pred_vow], self.VOWELS_IDX[valid_word_vowel[w_ind][vow_ind]]] += 1
        return conf_mat

    def predict(self, pred_set):
        result = []
        predicted = self.model.tag_sents(pred_set)
        for i, w_cons in enumerate(predicted):
            result.append(''.join(x+y for x, y in w_cons))
        return result

    @staticmethod
    def _len(x):
        return len(x) if isinstance(x, str) else int(x)

    VOWELS = [u'a',u'e',u'u',u'i',u'o',u'*']
    VOWELS_IDX = {x:i for i,x in enumerate(VOWELS)}
    @staticmethod
    def _extract_ftr(tokens, i):
        # print(tokens, i, tokens[i])
        feature_list = []
        feature_list.append("pos="+str(i))
        feature_list.append("cur="+tokens[i])
        feature_list.append("first="+tokens[0])
        feature_list.append("last="+tokens[-1])
        feature_list.append("len="+str(len(tokens)))
        if i > 0:
            feature_list.append("prev="+tokens[i-1])
        if i < (len(tokens)-1):
            feature_list.append("next="+tokens[i+1])
        return feature_list

if __name__ == '__main__':
    print("Word CRF: ")
    for i in range(5):
        crf = CRF()
        #crf.prep_data().shuffle().split(0).train().predict(['ˀnšym', u'nršmym'])
        if 'conf_mat' in locals():
            conf_mat += crf.prep_data().shuffle(None).split(0.1).train().eval()
        else:
            conf_mat = crf.prep_data().shuffle(None).split(0.1).train().eval()
    precision, recall = metrics.MicroAvg(conf_mat)
    f1 = metrics.Fscore(precision, recall, 1)
    print('MicroAvg:',precision,recall,f1)
    precision, recall = metrics.MacroAvg(conf_mat)
    f1 = metrics.Fscore(recall, precision, 1)
    print('MacroAvg:', precision, recall, f1)
    print('AvgAcc:',metrics.AvgAcc(conf_mat))
    conf_mat = metrics.NormalizeConfusion(conf_mat)
    print('ConfMat:\n', np.array_str(conf_mat, max_line_width=300, precision=3))
    print('----------------------------------------------')
