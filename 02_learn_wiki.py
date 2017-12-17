import logging
import os.path
import sys
import multiprocessing

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


wd = os.path.expanduser('~/Dropbox/github/py/word2vec_wiki/')

program = os.path.basename(wd + '01_processwiki.py')
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))


text_path = wd + '00_data/wiki.de.text'
model_path = wd + '00_model/wiki.de.word2vec.model'
vector_path = wd + '00_model/wiki.de.word2vec.vector'


model = Word2Vec(LineSentence(text_path), size=400, window=5, min_count=5, workers=3)

# trim unneeded model memory = use (much) less RAM
model.init_sims(replace=True)

model.save(model_path)
model.save_word2vec_format(vector_path, binary=False)



