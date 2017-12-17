import logging
import os.path
import sys
from gensim.corpora import WikiCorpus


wd = os.path.expanduser('~/Dropbox/github/py/word2vec_wiki/')

program = os.path.basename(wd + '01_processwiki.py')
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))



wiki_path = wd + '00_data/dewiki-latest-pages-articles.xml.bz2'
output_path = wd + '00_data/wiki.de.text'

space = " "
i = 0

# Function for replacing Umlauts
def replace_umlauts(text):
    res = text
    res = res.replace(u'ä', 'ae')
    res = res.replace(u'ö', 'oe')
    res = res.replace(u'ü', 'ue')
    res = res.replace(u'Ä', 'Ae')
    res = res.replace(u'Ö', 'Oe')
    res = res.replace(u'Ü', 'Ue')
    res = res.replace(u'ß', 'ss')
    return res

output = open(output_path, 'w')
wiki = WikiCorpus(wiki_path, lemmatize=False, dictionary={})
for text in wiki.get_texts():
    output.write(space.join(text) + "\n")
    i = i + 1
    if (i % 10000 == 0):
        logger.info("Saved " + str(i) + " articles")

output.close()
logger.info("Finished Saved " + str(i) + " articles")



output = open(output_path, 'w')
wiki = WikiCorpus(wiki_path, lemmatize=False, dictionary={})
for text in wiki.get_texts():
    sentence = []
    for word in text:
        word = replace_umlauts(word.decode('utf-8'))
        sentence.append(word)
    output.write(space.join(sentence).encode('utf-8')  + "\n")
    i = i + 1
    if (i % 10000 == 0):
        logger.info("Saved " + str(i) + " articles")


output.close()
logger.info("Finished Saved " + str(i) + " articles")