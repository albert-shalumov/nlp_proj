# API reference:
# https://radimrehurek.com/gensim/corpora/wikicorpus.html
# Additional references:
# https://github.com/liorshk/wordembedding-hebrew

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from multiprocessing import freeze_support
import logging

class Embedding:
    def __init__(self):
        pass

    def Extract(self, wiki_dump):
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
            for i,w in enumerate(text):
                if any(x in w for x in '1234567890'):
                    res[i]=u'*'
                    continue
                if any(x in w.lower() for x in 'qwertyuiopasdfghjklzxcvbnm'):
                    res[i]=u'*'
                    continue

                if any(x in w.lower() for x in u'\'`\u05f3\u05f4'):
                    print(w)

                res[i] = u''
                for ch in w:
                    res[i] += char_lut[ch] if ch in char_lut else ch

            return res

        wiki = WikiCorpus(wiki_dump, lemmatize=False, dictionary={})
        with open('wiki.txt', 'w', encoding="utf-8") as f:
            for i,text in enumerate(wiki.get_texts()):
                f.write("{}\n".format(u" ".join(cleanup(text))))
                #f.flush()
                if i%100 == 0:
                    print('{}\r'.format(i),end='')

    def TrainWord2Vec(self, model_fn):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
        logging.root.setLevel(level=logging.INFO)

        model = Word2Vec(LineSentence("wiki.txt"), sg=1, size=150, window=6, min_count=2, workers=6)
        model.save(model_fn)

        logging.root.setLevel(logging.CRITICAL)

    def TrainFastText(self, model_fn):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
        logging.root.setLevel(level=logging.INFO)

        model = FastText(LineSentence("wiki.txt"), sg=1, size=150, window=6, min_count=2, workers=6)
        model.save(model_fn)

        logging.root.setLevel(logging.CRITICAL)

if __name__ == '__main__':
    freeze_support()
    embed = Embedding()
    #embed.Extract("hewiki-latest-pages-articles.xml.bz2")

    embed.TrainWord2Vec("w2v_model")
    embed.TrainFastText("fastext_model")
