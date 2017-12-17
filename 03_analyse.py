import os.path
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

# Loading
wd = os.path.expanduser('~/Dropbox/github/py/word2vec_wiki/')
model = Word2Vec.load_word2vec_format(wd + '00_model/wiki.de.word2vec.vector', binary=False)


#---------------------------------------------------#
# Word count
model.vocab["elefant"].count


#---------------------------------------------------#
# Vector representation
model.most_similar(positive=['frau', 'koenig'], negative=['mann'])

model.most_similar(positive=['diktator', 'hitler'], negative=['nazi'])



#---------------------------------------------------#
# Doesn't match
model.doesnt_match('mann hund katze elefant loewe'.split())


#---------------------------------------------------#
# Raw numpy vector of distances
model['mann']


#---------------------------------------------------#
# Most similar words

# Test 1
model.most_similar("mann")
model.most_similar("frau")
model.most_similar("krieg")
model.most_similar("hitler")
model.most_similar("elefant")

# Test Candylabs
model.most_similar("baecker")
model.most_similar("metzger")

model.most_similar("restaurant")
model.most_similar("rechtsanwalt")

model.most_similar("arzt")
model.most_similar("zahnarzt")
model.most_similar("augenarzt")
model.most_similar("hautarzt")

model.most_similar("hotel")
model.most_similar("maler")
model.most_similar("friseur")




#---------------------------------------------------#
# PCA Plots

# configuration
currency = ["schweiz","franken","deutschland","euro","grossbritannien","pounds","japan","yen","russland","rubel","usa","dollar","kroatien","kuna"]
capital  = ["athen","griechenland","berlin","deutschland","ankara","tuerkei","bern","schweiz","hanoi","vietnam","lissabon","portugal","moskau","russland","stockholm","schweden","tokio","japan","washington","usa"]
language = ["deutschland","deutsch","usa","englisch","frankreich","franzoesisch","griechenland","griechisch","norwegen","norwegisch","schweden","schwedisch","polen","polnisch","ungarn","ungarisch"]
# matches = model.most_similar(positive=["Frau"], negative=[], topn=30)
# words = [match[0] for match in matches]


# draw pca plots
draw_words(model, currency, True, True, True, -3, 3, -2, 2, r'$PCA\ Visualisierung:\ W\ddot{a}hrung$')
draw_words(model, capital, True, True, True, -3, 3, -2, 2.2, r'$PCA\ Visualisierung:\ Hauptstadt$')
draw_words(model, language, True, True, True, -3, 3, -2, 1.7, r'$PCA\ Visualisierung:\ Sprache$')



test = ["riskant", "sicher", "aktie", "staatsanleihe"]
draw_words(model, test, True, True, True, -3, 3, -2, 2, r'$PCA\ Visualisierung:\ W\ddot{a}hrung$')

candylabs = ["baecker", "conditor", "restaurant", "cafe", "arzt", "krankenhaus"]

draw_words(model, candylabs, True, True, True, -3, 3, -2, 2)

#---------------------------------------------------#
# PCA Plot Rechtsanwalt

# Get most similar
similar_to_rechtsanwalt = model.most_similar("rechtsanwalt")
# Reduce a little bit
similar_to_rechtsanwalt = [similar_to_rechtsanwalt[i] for i in [0, 1, 2, 3, 5, 6, 8, 9]]

# Extract names and distances from list of list
names = []
distances = []

for i in similar_to_rechtsanwalt:
    names.append(i[0].decode("utf-8"))
    distances.append(i[1])

# Reverse order for horizonal bar
names = list(reversed(names))
distances = list(reversed(distances))
y_pos = np.arange(len(names))

# Horizontal Barplot using mathplotlib
plt.barh(y_pos, distances, align='center', color = '#BED547')
plt.yticks(y_pos, names)
plt.subplots_adjust(left=0.5)
plt.xlabel('Semantische Nähe'.decode('utf-8'))
plt.title('Ähnliche Wörter zu Rechtsanwalt'.decode('utf-8'))
