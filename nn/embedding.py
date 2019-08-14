# API reference:
# https://radimrehurek.com/gensim/corpora/wikicorpus.html
# Additional references:
# https://github.com/liorshk/wordembedding-hebrew

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from gensim.models.word2vec import LineSentence
from multiprocessing import freeze_support
import logging


def ConvertCorpus(self, wiki_dump):
    def cleanup(text):
        res = text[:]
        char_lut = {
            u'\u05d0': u'\u02c0', u'\u05d1': u'b',
            u'\u05d2': u'g', u'\u05d3': u'd',
            u'\u05d4': u'h', u'\u05d5': u'w',
            u'\u05d6': u'z', u'\u05d7': u'\u1e25',
            u'\u05d8': u'\u1e6d', u'\u05d9': u'y',
            u'\u05da': u'k', u'\u05db': u'k',
            u'\u05dc': u'l', u'\u05dd': u'm',
            u'\u05de': u'm', u'\u05df': u'n',
            u'\u05e0': u'n', u'\u05e1': u's',
            u'\u05e2': u'\u02c1', u'\u05e3': u'p',
            u'\u05e4': u'p', u'\u05e5': u'\u00e7',
            u'\u05e6': u'\u00e7', u'\u05e7': u'q',
            u'\u05e8': u'r', u'\u05e9': u'\u0161',
            u'\u05ea': u't'}
        for i, w in enumerate(text):
            if any(x in w for x in '1234567890'):
                res[i] = u'*'
                continue
            if any(x in w.lower() for x in 'qwertyuiopasdfghjklzxcvbnm'):
                res[i] = u'*'
                continue

            if any(x in w.lower() for x in u'\'`\u05f3\u05f4'):
                print(w)

            res[i] = u''
            for ch in w:
                res[i] += char_lut[ch] if ch in char_lut else ch

        return res

    wiki = WikiCorpus(wiki_dump, lemmatize=False, dictionary={})
    with open('wiki.txt', 'w', encoding="utf-8") as f:
        for i, text in enumerate(wiki.get_texts()):
            f.write("{}\n".format(u" ".join(cleanup(text))))
            # f.flush()
            if i%100 == 0:
                print('{}\r'.format(i), end='')


class Embedding:
    def __init__(self, name, single_train=False):
        self.name = name
        self.model = None
        self.single_train = single_train

    def Train(self):
        raise Exception("Error! Implement the method when inheriting!")

    def Load(self):
        raise Exception("Error! Implement the method when inheriting!")

    def Save(self, filename=None):
        fn = self.name+'_model' if filename is None else filename
        self.model.save(fn)

    def GetVector(self, words):
        return self.model[words]


class FastTextEmb(Embedding):
    def __init__(self, lean=False):
        Embedding.__init__(self, "fasttext", lean)

    def Train(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
        prev_lvl = logging.root.level
        logging.root.setLevel(level=logging.INFO)

        self.model = FastText(LineSentence("wiki.txt"), sg=1, size=150, window=6, min_count=2, iter=15, min_n=1, max_n=6, workers=6)
        if self.single_train:
            self.model.init_sims(replace=True)

        logging.root.setLevel(level=prev_lvl)


class Embedding_: # TODO: use polymorphism to support different embedding algorithms
    def __init__(self, type="fasttext"):
        self.model = None
        if type in ["fasttext",'word2vec']:
            self.type = type
        else:
            raise Exception("Unsupported type {}".format(type))


    def Train(self, new=True):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
        logging.root.setLevel(level=logging.INFO)
        if self.type=='word2vec':
            if not new:
                self.model = Word2Vec.load(self.type+'_model')
            self.model = Word2Vec(LineSentence("wiki.txt"), sg=1, size=150, window=6, min_count=2, workers=6)
        elif self.type=='fasttext':
            if not new:
                self.model = FastText.load(self.type+'_model')
            self.model = FastText(LineSentence("wiki.txt"), sg=1, size=150, window=6, min_count=2, workers=6)
        logging.root.setLevel(logging.CRITICAL)
        self.model.save(self.type + '_model')
        return self

    def LoadModel(self, trim=False):
        if self.type=='word2vec':
            self.model = Word2Vec.load(self.type+'_model')
        elif self.type=='fasttext':
            self.model = FastText.load(self.type+'_model')
        if trim:
            self.model.init_sims(replace=True)
        return self

    def GetEmbedding(self, word):
        return self.model[word]

if __name__ == '__main__':
    freeze_support()
    embed = FastTextEmb(True)
    embed.Train()#.Save()



