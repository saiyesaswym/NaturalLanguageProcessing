import pandas as pd
import nltk
from math import log
import collections
from nltk.util import ngrams
import string

#Reading the train and test data
print("#Reading the train and test data")
with open('data/train', 'r') as myfile:
    data_rtrain=myfile.read().splitlines()
#Removing the punctuation
data = [''.join(c for c in s if c not in string.punctuation) for s in data_rtrain]

with open('data/test', 'r') as myfile:
    data_rtest=myfile.read().splitlines()
#Removing the punctuation
data_test = [''.join(c for c in s if c not in string.punctuation) for s in data_rtest]


#Calculating PRIORs
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


#--------------------------------PART A--------------------------

#Counting the number of bigrams under each class into SPEAKBI
print('\nImplementing the Text classification using BIGRAMS')
speakbi = {}
for i in data:
    word_list = i.split()
    word = word_list[0]
    word_list.remove(word)
    word_list=list(set(word_list))
    bisp_list=list(ngrams(word_list,2))
    #bisp_list=list(set(bisp_list))
    if(speakbi.get(word)==None):
        speakbi[word]=len(bisp_list)
    else:
        speakbi[word]+=len(bisp_list)

print("Number of classes: "+str(len(speakbi.keys())))
speakers = list(speakbi.keys())


#Placing all the words related to a speaker together in a dictionary MAINBILIST
mainbilist={}
for i in speakers:
    templist={}
    for j in data:
        words_list=j.split()
        word=words_list[0]
        words_list.remove(word)
        bi_list=list(ngrams(words_list,2))
        if(i==word):
            for k in bi_list:
                if(templist.get(k)==None):
                    templist[k]=1
                else:
                    templist[k]+=1
    mainbilist[i]=templist


#Calculating the accuracy on the test data
def calculateBiAccuracy(alpha):
    acc_bi=0
    for line in data_test:
        line_list=line.split()
        p_lineall={}
        word=line_list[0]
        line_list.remove(word)
        test_list=list(ngrams(line_list,2))
        for sp in speakers:
            p_line=0
            for i in test_list:
                if(mainbilist[sp].get(i)!=None):
                    p_word = (mainbilist[sp][i]+alpha)/(speakbi[sp]+(len(vocab)*alpha))
                    p_word = (log(p_word))
                    p_line += p_word
                else:
                    p_word = (alpha)/(speakbi[sp]+(len(vocab)*alpha))
                    p_word = (log(p_word))
                    p_line += p_word
                    
            p_line=log(priors[sp])+p_line       
            p_lineall[sp]=p_line

        if(word==max(p_lineall,key=p_lineall.get)):
            acc_bi+=1
    return acc_bi


print("\n#Calculating the accuracy on test data using Add-one smoothing:\n")
acc = calculateBiAccuracy(1)
print("Positive matches: "+str(acc))
print("Negative matches: "+str(len(data_test)-acc))

print("\nAccuracy of the model is "+str((acc/len(data_test))*100))



#--------------------------------TRIGRAM--------------------------

#Counting the number of bigrams under each class into SPEAKTRI
print('\nImplementing the Text classification using TRIGRAMS')
speaktri = {}
for i in data:
    word_list = i.split()
    word = word_list[0]
    word_list.remove(word)
    word_list=list(set(word_list))
    trisp_list=list(ngrams(word_list,3))
    if(speaktri.get(word)==None):
        speaktri[word]=len(trisp_list)
    else:
        speaktri[word]+=len(trisp_list)

print("Number of classes: "+str(len(speaktri.keys())))
speakers = list(speaktri.keys())


#Placing all the words related to a speaker together in a dictionary MAINTRILIST
maintrilist={}
for i in speakers:
    templist={}
    for j in data:
        words_list=j.split()
        word=words_list[0]
        words_list.remove(word)
        tri_list=list(ngrams(words_list,3))
        if(i==word):
            for k in tri_list:
                if(templist.get(k)==None):
                    templist[k]=1
                else:
                    templist[k]+=1
    maintrilist[i]=templist


#Calculating the accuracy on the test data
def calculateTriAccuracy(alpha):
    acc_tri=0
    for line in data_test:
        line_list=line.split()
        p_lineall={}
        word=line_list[0]
        line_list.remove(word)
        test_list=list(ngrams(line_list,3))
        for sp in speakers:
            p_line=0
            for i in test_list:
                if(mainbilist[sp].get(i)!=None):
                    p_word = (maintrilist[sp][i]+alpha)/(speaktri[sp]+(len(vocab)*alpha))
                    p_word = (log(p_word))
                    p_line += p_word
                else:
                    p_word = (alpha)/(speaktri[sp]+(len(vocab)*alpha))
                    p_word = (log(p_word))
                    p_line += p_word
                    
            p_line=log(priors[sp])+p_line       
            p_lineall[sp]=p_line

        if(word==max(p_lineall,key=p_lineall.get)):
            acc_tri+=1
    return acc_tri


print("\n#Calculating the accuracy on test data using Add-one smoothing:\n")
acc2 = calculateTriAccuracy(1)
print("Positive matches: "+str(acc2))
print("Negative matches: "+str(len(data_test)-acc2))

print("\nAccuracy of the model is "+str((acc2/len(data_test))*100))


#------------------------------PART B----------------------------------

#Counting the number of words under each class in SPEAK dictionary
speakbin = {}
for i in data:
    word_list = i.split()
    word = word_list[0]
    #Only UNIQUE words are considered from the train set
    word_list = list(set(word_list))
    if(speakbin.get(word)==None):
        speakbin[word]=len(word_list[1:])
    else:
        speakbin[word]+=len(word_list[1:])

print("Number of classes: "+str(len(speakbin.keys())))
speakers = list(speakbin.keys())



#Storing all the words related to a speaker together in a dictionary
mainbinlist={}
for i in speakers:
    templist={}
    for j in data:
        words_list=j.split()
        word = words_list[0]
        #Only UNIQUE words are considered from the train set
        words_list=list(set(words_list))
        if(i==word):
            for k in words_list:
                if(templist.get(k)==None):
                    templist[k]=1
                else:
                    templist[k]+=1
    templist.pop(i)
    mainbinlist[i]=templist


    
#Function for calculating the accuracy on the test data
def calculateAccuracy(alpha):
    acc3=0
    #Iterating through each document in test data
    for line in data_test:
        line_list=line.split()
        p_lineall={}
        word=line_list[0]
        line_list.remove(word)
        line_list=list(set(line_list))
        #Iterating for all the classes available
        for sp in speakers:
            p_line=0
            #Iterating through all the words in the considered document
            for i in line_list:
                if(mainbinlist[sp].get(i)!=None):
                    p_word = (mainbinlist[sp][i]+alpha)/(speakbin[sp]+(len(vocab)*alpha))
                    p_word = (log(p_word))
                    p_line += p_word
                else:
                    p_word = (alpha)/(speakbin[sp]+(len(vocab)*alpha))
                    p_word = (log(p_word))
                    p_line += p_word
            #Probability of the line        
            p_line=log(priors[sp])+p_line       
            p_lineall[sp]=p_line

        if(word==max(p_lineall,key=p_lineall.get)):
            acc3+=1
    return acc3

print("\n#Calculating the accuracy on test data for Binomial model :\n")
acc3 = calculateAccuracy(0.1)
print("Positive matches: "+str(acc3))
print("Negative matches: "+str(len(data_test)-acc3))

print("\nAccuracy of the model is "+str((acc3/len(data_test))*100))

