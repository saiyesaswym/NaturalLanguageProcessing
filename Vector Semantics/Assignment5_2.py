from gensim.models import KeyedVectors

#Creating the model using the pretrained embeddings
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=1000000)


#-----------------Predicting the top 10 most similar words based on the model----------

model.most_similar('increase',topn=10)
model.most_similar('up',topn=10)
model.most_similar('agree',topn=10)
model.most_similar('enter',topn=10)
model.most_similar('beautiful',topn=10)
