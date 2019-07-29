# API reference:
# https://radimrehurek.com/gensim/corpora/wikicorpus.html
#
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from multiprocessing import freeze_support

'''
import multiprocessing

import time

def train(inp = "wiki.he.text",out_model = "wiki.he.word2vec.model"):

    start = time.time()

    model = Word2Vec(LineSentence(inp), sg = 1, # 0=CBOW , 1= SkipGram
                     size=100, window=5, min_count=5, workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)

    print(time.time()-start)

    model.save(out_model)

def getModel(model = "wiki.he.word2vec.model"):

    model = Word2Vec.load(model)

    return model

'''
class Embedding:
    def __init__(self):
        pass

    def Extract(self, wiki_dump):
        def cleanup(text):
            res = text[:]
            for i,w in enumerate(text):
                if any(x in w for x in '1234567890'):
                    res[i]=u'*'
                    continue
                if any(x in w.lower() for x in 'qwertyuiopasdfghjklzxcvbnm'):
                    res[i]=u'*'
                    continue
            return res

        wiki = WikiCorpus(wiki_dump, lemmatize=False, dictionary={})
        with open('wiki.txt', 'w', encoding="utf-8") as f:
            for i,text in enumerate(wiki.get_texts()):

                f.write("{}\n".format(u" ".join(cleanup(text))))
                #f.flush()
                if i%100 == 0:
                    print('{}\r'.format(i),end='')

    def TrainWord2Vec(self, model_fn):
        model = Word2Vec(LineSentence("wiki.txt"), sg=1, size=150, window=6, min_count=3, workers=4)
        model.save(model_fn)

    def TrainFastText(self, model_fn):
        model = FastText(LineSentence("wiki.txt"), sg=1, size=150, window=6, min_count=3, workers=4)
        model.save(model_fn)

if __name__ == '__main__':
    freeze_support()
    embed = Embedding()
    embed.Extract("hewiki-latest-pages-articles.xml.bz2")
    embed.TrainFastText("fastext_model")
    embed.TrainWord2Vec("w2v_model")
