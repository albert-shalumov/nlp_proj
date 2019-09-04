from nltk.tag.crf import CRFTagger
import codecs
import numpy as np
import metrics
from itertools import combinations

class CRF:
    def __init__(self):
        self.ftrs = config['ftrs']
        for ftr in self.ftrs:
            if ftr not in CRF.CONFIG:
                raise Exception('Unknown feature {}. See MEMM.CONFIG for supported ones.'.format(CRF.CONFIG))

    def prep_data(self, file='data/HaaretzOrnan_annotated.txt'):
        self.data = []
        with codecs.open(file, encoding='utf-8') as f:
            lines = f.readlines()
            self.data.append([])
            for line in lines:
                line = line.rstrip()
                # Start new sentence
                if line.startswith(u'#'):
                    continue
                if len(line) == 0:
                    if len(self.data[-1]) > 0:
                        self.data.append([])
                    continue
                # Append word to last sentence
                w = line.split(u' ')[3]
                w = w.replace(u'-', u'')
                self.data[-1].append(w)

        # If sentence is empty - remove it
        if len(self.data[-1]) == 0:
            self.data.remove(self.data[-1])
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
        train_set = CRF._fin_data_prep(self.train_set)
        _extract_ftr = self._gen_ftr_func()
        self.model = CRFTagger(_extract_ftr, verbose=False, training_opt={"c1":0,"c2":0, "num_memories":50, "epsilon":1e-7, "delta":1e-8})
        self.model.train(train_set, 'stc_crf_model')
        return self

    def eval(self):
        conf_mat = np.zeros((len(CRF.VOWELS), len(CRF.VOWELS)))
        valid_set = CRF._fin_data_prep(self.valid_set)
        valid_stc_cons = [[x[0] for x in w] for w in valid_set]
        valid_stc_vowel = [[x[1] for x in w] for w in valid_set]
        predicted = self.model.tag_sents(valid_stc_cons)
        predicted = [[x[1] for x in w] for w in predicted]
        for w_ind in range(len(predicted)):
            for vow_ind, pred_vow in enumerate(predicted[w_ind]):
                conf_mat[self.VOWELS_IDX[pred_vow], self.VOWELS_IDX[valid_stc_vowel[w_ind][vow_ind]]] += 1
        return conf_mat

    def predict(self, pred_set):
        data = []
        for sent in pred_set:
            sent_cons = u' '.join(sent)
            for i, w in enumerate(sent):
                w_cons = list(w)
                w_pos = [i]*len(w)
                unif_sent = [sent_cons]*len(w)
                d = list(zip(w_cons, w_pos, unif_sent))
                data.append(d)
        pred = self.model.tag_sents(data)
        result = []
        word_idx = 0
        for sent in pred_set:
            result.append([])
            for word in sent:
                pred_smpl = pred[word_idx]
                w = ''.join([entry[0][0]+entry[-1] for entry in pred_smpl])
                result[-1].append(w)
                word_idx += 1
        return result

    @staticmethod
    def _fin_data_prep(data_set):
        data = []
        for sent in data_set:
            sent_cons = u' '.join([x[::2] for x in sent])
            for i, w in enumerate(sent):
                w_cons = list(w[::2])
                w_pos = [i]*len(w[::2])
                unif_sent = [sent_cons]*len(w[::2])
                d = list(zip(w_cons, w_pos, unif_sent))
                data.append(list(zip(d, list(w[1::2]))))
        return data

    @staticmethod
    def _len(x):
        return len(x) if isinstance(x, str) else int(x)

    VOWELS = [u'a',u'e',u'u',u'i',u'o',u'*']
    VOWELS_IDX = {x:i for i,x in enumerate(VOWELS)}
    CONFIG = ['IS_FIRST', 'IS_LAST', 'IDX', 'VAL', 'PRV_VAL', 'NXT_VAL', 'FRST_VAL', 'LST_VAL', 'SCND_VAL',
              'SCND_LST_VAL', 'LEN', 'STC_LEN']

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



    @staticmethod
    def _extract_stc_ftr(tokens, i):
        feature_list = []
        word_pos = tokens[0][1]
        sent = tokens[0][2].split(' ')
        word = sent[word_pos]

        # Word features
        feature_list += CRF._extract_wrd_ftr(word, i, "_0")

        # Sentence features
        if word_pos == 0:
            feature_list.append('FIRST_WORD')

        if word_pos == (len(sent)-1):
            feature_list.append('LAST_WORD')

        if word_pos > 0:
            feature_list.append("prev_w_last_ch="+sent[word_pos-1][-1])

        return feature_list

    @staticmethod
    def _extract_wrd_ftr(word, i, suff):
        feature_list = []
        feature_list.append("pos{}={}".format(suff,str(i)))
        feature_list.append("cur{}={}".format(suff,word[i]))
        feature_list.append("first{}={}".format(suff,word[0]))
        feature_list.append("last{}={}".format(suff,word[-1]))
        feature_list.append("len{}={}".format(suff,str(len(word))))
        if i > 0:
            feature_list.append("prev{}={}".format(suff,word[i-1]))
        if i < (len(word)-1):
            feature_list.append("next{}={}".format(suff,word[i+1]))
        return feature_list

if __name__ == '__main__':
    verbose = False
    with open('crf_word_res.csv','w') as f:
        for num_ftrs in range(len(CRF.CONFIG)):
            num_ftrs += 1
            for ftrs in combinations(CRF.CONFIG, num_ftrs):
                config = {'ftrs':ftrs}
                if 'conf_mat' in locals():
                    del conf_mat
                for i in range(5):
                    crf = CRF(config)
                    #crf.prep_data().shuffle().split(0).train().predict(['ˀnšym', u'nršmym'])
                    if 'conf_mat' in locals():
                        conf_mat += crf.prep_data().shuffle(0).split(0.1).train().eval()
                    else:
                        conf_mat = crf.prep_data().shuffle(0).split(0.1).train().eval()
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

