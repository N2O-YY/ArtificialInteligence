from gensim.models import word2vec

sentences = word2vec.LineSentence('data.txt')
model = word2vec.Word2Vec(sentences,size=256, window=5, min_count=3, workers=4)

model.save('w2v')
