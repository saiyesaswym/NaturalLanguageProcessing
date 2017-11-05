import pandas as pd
import nltk
from math import log
import collections
import string

#Reading the train and test data
print("#Reading the train and test data")
with open('data/train', 'r') as myfile:
    data_rtrain=myfile.read().splitlines()
#Removing the punctuation
data = [''.join(c for c in s if c not in string.punctuation) for s in data_rtrain]

with open('data/test', 'r') as myfile:
    data_rtest=myfile.read().splitlines()
#Removing the puntuations
data_test = [''.join(c for c in s if c not in string.punctuation) for s in data_rtest]


#Calculating PRIORS of the classes

print("#Calculating the PRIORs")
all_docs=[]
for i in data:
    all_docs.append(i.split()[0])

#Counting the number of documents under each class
classes_count = collections.Counter(all_docs)

priors={}
for i in classes_count:
    priors[i]=classes_count[i]/len(data)  

#Calculating the vocabulary count
listmain=[]
for i in data:
    words = i.split()
    listmain+=words

vocab=[]
for word in listmain:
    if word not in vocab:
        vocab.append(word)



#Counting the number of words under each class in SPEAK dictionary
speak = {}
for i in data:
    word_list = i.split()

    if(speak.get(word_list[0])==None):
        speak[word_list[0]]=len(word_list[1:])
    else:
        speak[word_list[0]]+=len(word_list[1:])

print("Number of classes: "+str(len(speak.keys())))
speakers = list(speak.keys())



#Storing all the words and their counts related to a speaker together in a dictionary
mainlist={}
for i in speakers:
    templist={}
    for j in data:
        words_list=j.split()
        if(i==words_list[0]):
            for k in words_list:
                if(templist.get(k)==None):
                    templist[k]=1
                else:
                    templist[k]+=1
    templist.pop(i)
    mainlist[i]=templist


    
#Function for calculating the accuracy on the test data
def calculateAccuracy(alpha):
    acc=0
    #Iterating through each document in test data
    for line in data_test:
        line_list=line.split()
        p_lineall={}
        word=line_list[0]
        line_list.remove(word)
        #Iterating for all the classes available
        for sp in speakers:
            p_line=0
            #Iterating through all the words in the considered document
            for i in line_list:
                if(mainlist[sp].get(i)!=None):
                    p_word = (mainlist[sp][i]+alpha)/(speak[sp]+(len(vocab)*alpha))
                    p_word = (log(p_word))
                    p_line += p_word
                else:
                    p_word = (alpha)/(speak[sp]+(len(vocab)*alpha))
                    p_word = (log(p_word))
                    p_line += p_word
            #Probability of the line        
            p_line=log(priors[sp])+p_line       
            p_lineall[sp]=p_line

        if(word==max(p_lineall,key=p_lineall.get)):
            acc+=1
    return acc

print("\n#Calculating the accuracy on test data using Add-one or Laplacian smoothing:\n")
acc = calculateAccuracy(1)
print("Positive matches: "+str(acc))
print("Negative matches: "+str(len(data_test)-acc))

print("\nAccuracy of the model is "+str((acc/len(data_test))*100))


print("\n#Calculating the accuracy on test data using different alpha value of 0.1:\n")
acc2 = calculateAccuracy(0.1)
print("Positive matches: "+str(acc2))
print("Negative matches: "+str(len(data_test)-acc2))

print("\nAccuracy of the model is "+str((acc2/len(data_test))*100))
