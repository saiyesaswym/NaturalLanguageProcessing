from gensim.models import KeyedVectors

#Creating the model using the pretrained embeddings
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=1000000)


#-----------Creating two new analogies - Cause and effect / Performer and Action

cause_effect_list = [['sunrise','dawn','sunset','dusk'],
                     ['tired','sleep','hungry','eat'],
                     ['stress','anxiety','war','destruction']]

performer_action_list = [['painter','paint','dancer','dance'],
                         ['teacher','educate','student','learn'],
                         ['artist','paint','scientist','research']]

#Finding the accuracy for the Cause and effect analogy
acc1=0
for i in cause_effect_list:
    if all(a in model for a in [i[0],i[1],i[2]]):
        result = model.most_similar(positive=[i[1],i[2]],negative=[i[0]],topn=1)
        print(result[0])
        if(result[0][0]==i[3]):
            acc1+=1

print(acc1/len(cause_effect_list))


#Finding the accuracy for the Performer and Action analogy
acc2=0
for i in performer_action_list:
    if all(a in model for a in [i[0],i[1],i[2]]):
        result = model.most_similar(positive=[i[1],i[2]],negative=[i[0]],topn=1)
        print(result[0])
        if(result[0][0]==i[3]):
            acc2+=1

print(acc2/len(performer_action_list))
