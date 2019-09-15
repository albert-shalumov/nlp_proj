from nltk.tag.crf import CRFTagger
import codecs
import numpy as np
import metrics
from itertools import combinations
import sys

class CRF:
    def __init__(self, config):
        self.ftrs = config['ftrs']
        for ftr in self.ftrs:
            if ftr not in CRF.WORD_FTRS:
                raise Exception('Unknown feature {}. See CRF.CONFIG for supported ones.'.format(CRF.WORD_FTRS))

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

    def train(self, load_model=None):
        _extract_ftr = self._gen_ftr_func()
        self.model = CRFTagger(_extract_ftr, verbose=False,
                       training_opt={"num_memories": 500, "delta": 1e-8})
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
        for sent in pred_set:
            pred_sent = []
            predicted = self.model.tag_sents(sent)
            for i, w_cons in enumerate(predicted):
                pred_sent.append(''.join(x+y for x, y in w_cons))
            result.append(pred_sent)
        return result

    @staticmethod
    def _len(x):
        return len(x) if isinstance(x, str) else int(x)

    VOWELS = [u'a',u'e',u'u',u'i',u'o',u'*']
    VOWELS_IDX = {x:i for i,x in enumerate(VOWELS)}
    WORD_FTRS = ['IS_FIRST', 'IS_LAST', 'IDX', 'VAL', 'PRV_VAL', 'NXT_VAL', 'FRST_VAL', 'LST_VAL', 'SCND_VAL', 'SCND_LST_VAL', 'LEN']

    def _gen_ftr_func(self):
        # Closure
        def _extract_ftr(tokens, i):
            # print(tokens, i, tokens[i])
            feature_list = []
            if 'IS_FIRST' in self.ftrs:
                feature_list.append("is_first="+str(1 if i == 0 else 0))

            if 'IS_LAST' in self.ftrs:
                feature_list.append("is_last="+str(1 if i == (len(tokens)-1) else 0))

            if 'IDX' in self.ftrs:
                feature_list.append("pos="+str(i))

            if 'VAL' in self.ftrs:
                feature_list.append("cur="+tokens[i])

            if 'PRV_VAL' in self.ftrs:
                if i > 0:
                    feature_list.append("prev="+tokens[i-1])

            if 'NXT_VAL' in self.ftrs:
                if i < (len(tokens)-1):
                    feature_list.append("next="+tokens[i+1])

            if 'FRST_VAL' in self.ftrs:
                feature_list.append("first="+tokens[0])

            if 'LST_VAL' in self.ftrs:
                feature_list.append("last="+tokens[-1])

            if 'LEN' in self.ftrs:
                feature_list.append("len="+str(len(tokens)))

            if 'SCND_VAL' in self.ftrs:
                if len(tokens)>1:
                    feature_list.append("scnd="+tokens[1])

            if 'SCND_LST_VAL' in self.ftrs:
                if len(tokens)>1:
                    feature_list.append("scnd_last="+tokens[-2])

            return feature_list
        return _extract_ftr

def search_hparams():
    verbose = False
    with open('crf_word_res.csv','w') as f:
        for num_ftrs in range(len(CRF.WORD_FTRS)):
            num_ftrs += 1
            for ftrs in combinations(CRF.WORD_FTRS, num_ftrs):
                config = {'ftrs':ftrs}
                if 'conf_mat' in locals():
                    del conf_mat
                for i in range(7):
                    crf = CRF(config)
                    if 'conf_mat' in locals():
                        conf_mat += crf.prep_data().shuffle(None).split(0.1).train().eval()
                    else:
                        conf_mat = crf.prep_data().shuffle(None).split(0.1).train().eval()
                res_str = '{};'.format(config)
                print("Configuration = {}: ".format(config))
                precision, recall = metrics.MicroAvg(conf_mat)
                f1 = metrics.Fscore(precision, recall, 1)
                res_str += '{};'.format(f1)
                print('MicroAvg:',precision,recall,f1)
                precision, recall = metrics.MacroAvg(conf_mat)
                f1 = metrics.Fscore(recall, precision, 1)
                res_str += '{};'.format(f1)
                print('MacroAvg:', precision, recall, f1)
                acc = metrics.AvgAcc(conf_mat)
                res_str += '{};'.format(acc)
                print('AvgAcc:',acc)
                f.write(res_str+'\n')
                conf_mat = metrics.NormalizeConfusion(conf_mat)
                if verbose:
                    print('ConfMat:\n', np.array_str(conf_mat, max_line_width=300, precision=4))
                    print('----------------------------------------------')

def check_seeds():
    config = {'ftrs': ('IS_FIRST', 'IS_LAST', 'IDX', 'VAL', 'PRV_VAL', 'NXT_VAL', 'FRST_VAL', 'LST_VAL', 'SCND_VAL', 'SCND_LST_VAL')}
    print("seed, accuracy")
    for seed in range(11):
        if 'conf_mat' in locals():
            del conf_mat
        for iters in range(7):
            crf = CRF(config)
            if 'conf_mat' in locals():
                conf_mat += crf.prep_data().shuffle(seed).split(0.1).train().eval()
            else:
                conf_mat = crf.prep_data().shuffle(seed).split(0.1).train().eval()
        acc = metrics.AvgAcc(conf_mat)
        print(seed, acc)

def print_usage():
    print("Usage:")
    print("crf_word.py [search/seeds]")
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
