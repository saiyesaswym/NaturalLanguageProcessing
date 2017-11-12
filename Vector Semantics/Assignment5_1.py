import requests
from gensim.models import KeyedVectors

#Reading the Mikolov's analogy data from the given URL 
response = requests.get('http://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt')

data=response.text
data=data.split('\n')

df = [i.split(' ') for i in data]

df=df[1:]

#Split all the sentences based on the module name 
dfdic={}
for i in df:
    if(i[0]==':'):
        key=i[1]
        dfdic[key]=[]
    else:
        dfdic[key].append(i)

#Filtering out the required 8 modules out of all given modules
reqd = ['capital-world', 'currency', 'city-in-state', 'family',
                      'gram1-adjective-to-adverb', 'gram2-opposite', 'gram3-comparative',
                      'gram6-nationality-adjective']
dfdict = {}
for classes in reqd:
    dfdict[classes] = dfdic[classes]

#-----------PART 1 - USING THE WORD2VEC EMBEDDINGS MODEL---------------

#Creating the model using the pretrained embeddings
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=1000000)

#Testing the model on the Mikolov's analogy dataset

#Creating a dictionary of all the modules to save the respective accuracies
acc_word2vec={'capital-world':0,'currency':0,'city-in-state':0,'family':0,'gram1-adjective-to-adverb':0,
     'gram2-opposite':0,'gram3-comparative':0,'gram6-nationality-adjective':0}

#FOr every module in the test file(8 modules),accuracy is calculated independently
for task in dfdict:
	tempacc=0
	for i in dfdict[task]:
		if all(a in model for a in [i[0],i[1],i[2]]):
			result = model.most_similar(positive=[i[1],i[2]], negative=[i[0]], topn=1)
			if(result[0][0]==i[3]):
				tempacc +=1
	acc_word2vec[task]=(tempacc/len(dfdict[task]))*100
print(acc_word2vec)

#-----------PART 2 - USING THE GLOVE EMBEDDINGS MODEL---------------

#Using the method glove2word2vec, the GLOVE pretrained embedding is converted to word2vec
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.42B.300d.txt'
word2vec_output_file = 'glove.42B.300d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

#model is built using the imported GLOVE word embeddings
from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = 'glove.42B.300d.txt.word2vec'
model_glove = KeyedVectors.load_word2vec_format(filename, binary=False)

acc_glove={'capital-world':0,'currency':0,'city-in-state':0,'family':0,'gram1-adjective-to-adverb':0,
     'gram2-opposite':0,'gram3-comparative':0,'gram6-nationality-adjective':0}

#Accuracy of each module in the test data is determined if the most similar word predicted is same as fourth word 
for task in dfdict:
	tempacc=0
	for i in dfdict[task]:
		if i[0] in model_glove:
			result = model_glove.most_similar(positive=[i[1],i[2]], negative=[i[0]], topn=1)
			if(result[0][0]==i[3]):
				tempacc+=1
	acc_glove[task]=(tempacc/len(dfdict[task]))*100
print(acc_glove)