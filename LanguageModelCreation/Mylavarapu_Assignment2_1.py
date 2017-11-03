
import nltk 
from nltk.corpus import udhr 
from nltk.util import ngrams
import re
import numpy as np 

english = udhr.raw('English-Latin1') 
french = udhr.raw('French_Francais-Latin1') 
italian = udhr.raw('Italian_Italiano-Latin1') 
spanish = udhr.raw('Spanish_Espanol-Latin1')  

#Creating training, development and test samples from datasets

english_train, english_dev = english[0:1000], english[1000:1100]
french_train, french_dev = french[0:1000], french[1000:1100]
italian_train, italian_dev = italian[0:1000], italian[1000:1100]
spanish_train, spanish_dev = spanish[0:1000], spanish[1000:1100]

english_test = udhr.words('English-Latin1')[0:1000] 
french_test = udhr.words('French_Francais-Latin1')[0:1000]
italian_test = udhr.words('Italian_Italiano-Latin1')[0:1000] 
spanish_test = udhr.words('Spanish_Espanol-Latin1')[0:1000]

#------------PROBLEM 1-----------------------


#------------BUILDING MODELS-------------------

#ENGLISH
eng_char=[]

for i in english_train:
    line = i.lower()
    if(line!='\n' and line!=','):
        eng_char.append(line)

#FRENCH
fr_char=[]

for i in french_train:
    line = i.lower()
    if(line!='\n' and line!=','):
        fr_char.append(line)

#SPANISH
sp_char=[]

for i in spanish_train:
    line = i.lower()
    if(line!='\n' and line!=','):
        sp_char.append(line)

#ITALIAN
it_char=[]

for i in italian_train:
    line = i.lower()
    if(line!='\n' and line!=','):
        it_char.append(line)



#-------------BUILDING UNIGRAM MODELS----------------
#ENGLISH
freq_eng_uni=nltk.FreqDist(eng_char)
tot_eng=len(eng_char)


#FRENCH
freq_fr_uni=nltk.FreqDist(fr_char)
tot_fr=len(fr_char)

#SPANISH
freq_sp_uni=nltk.FreqDist(sp_char)
tot_sp=len(sp_char)

#ITALIAN
freq_it_uni=nltk.FreqDist(it_char)
tot_fr=len(it_char)


#--------------BUILDING BIGRAM MODELS-----------------

#ENGLISH

eng_bigram = list(ngrams(eng_char,2))

freq_eng_uni = nltk.FreqDist(eng_char)
freq_eng_bi = nltk.ConditionalFreqDist(eng_bigram)

#FRENCH
fr_bigram = list(ngrams(fr_char,2))

freq_fr_uni = nltk.FreqDist(fr_char)
freq_fr_bi = nltk.ConditionalFreqDist(fr_bigram)

#SPANISH
sp_bigram = list(ngrams(sp_char,2))

freq_sp_uni = nltk.FreqDist(sp_char)
freq_sp_bi = nltk.ConditionalFreqDist(sp_bigram)

#ITALIAN
it_bigram = list(ngrams(it_char,2))

freq_it_uni = nltk.FreqDist(it_char)
freq_it_bi = nltk.ConditionalFreqDist(it_bigram)


#-------------BUILDING TRIGRAM MODELS------------

#ENGLISH
eng_bigram = list(ngrams(eng_char,2))
eng_trigram = list(ngrams(eng_char,3))

#FRENCH
fr_bigram = list(ngrams(fr_char,2))
fr_trigram = list(ngrams(fr_char,3))

#SPANISH
sp_bigram = list(ngrams(sp_char,2))
sp_trigram = list(ngrams(sp_char,3))

#ITALIAN
it_bigram = list(ngrams(it_char,2))
it_trigram = list(ngrams(it_char,3))

#------------PROCESSING THE WORDS IN ENGLISH TEST SET---------
eng_test = ' '.join(list(english_test))
from  nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
test_tokens = tokenizer.tokenize(eng_test.lower())
test_tokens = [item for item in test_tokens if not item.isdigit()]

#-------------ENGLISH Vs FRENCH UNIGRAM MODEL-------
uni_list=[];
for s in test_tokens:
    engprob = 1;
    frprob = 1;
    for i in s:
        engprob = engprob * (freq_eng_uni[i]/tot_eng)
        frprob = frprob * (freq_fr_uni[i]/tot_fr)
    if(engprob>frprob):
        uni_list.append("English")
    else:
        uni_list.append("French")

d = nltk.FreqDist(uni_list)
accuracy_unigram = (d['English']/len(uni_list))*100
print("Accuracy of unigram model for English: "+str(accuracy_unigram))


#-------------ENGLISH Vs FRENCH BIGRAM MODEL-------
bigr_list=[];
for s in test_tokens:
    engprob = 1;
    frprob = 1;
    for i in s:
        if(s.index(i)==0):
            engprob = engprob*((freq_eng_bi[' '][i])/freq_eng_uni[' '])
            frprob = frprob*((freq_fr_bi[' '][i])/freq_fr_uni[' '])
        else:
            bef = s[s.index(i)-1]
            if(bef=='x' or bef=='z'):
                engprob*=1
            else:
                engprob = engprob*((freq_eng_bi[bef][i])/freq_eng_uni[bef])
            if(bef=='k' or bef=='w' or bef=='z'):
                frprob*= 1
            else:
                frprob = frprob*((freq_fr_bi[bef][i])/freq_fr_uni[bef])
    if(engprob>frprob):
        bigr_list.append("English")
    else:
        bigr_list.append("French")

l = nltk.FreqDist(bigr_list)
accuracy_bigram = (l['English']/len(bigr_list))*100
print("Accuracy of bigram model for English: "+str(accuracy_bigram))


#-------------ENGLISH Vs FRENCH TRIGRAM MODEL-------
from collections import defaultdict
d = defaultdict(dict)
temp=[]
for i in eng_trigram:
    if(i not in temp):
        temp.append(i)
        d[i]=1
    else:
        d[i]+=1

f = defaultdict(dict)
temp2=[]
for i in fr_trigram:
    if(i not in temp2):
        temp2.append(i)
        f[i]=1
    else:
        f[i]+=1

tri_list=[];
for s in test_tokens:
    engprob = 1;
    frprob = 1;
    for i in s:
        if(s.index(i)==0):
            engprob= 1
            frprob= 1
        elif(s.index(i)==1):
            prev = s[s.index(i)-1]
            if(d[(' ',prev,i)]=={}):
                engprob*= 1
            else:
                engprob = engprob*(d[(' ',prev,i)]/freq_eng_bi[' '][prev])
            if(f[(' ',prev,i)]=={}):
                frprob*= 1
            else:
                frprob = frprob*(f[(' ',prev,i)]/freq_fr_bi[' '][prev])
        else:
            prev = s[s.index(i)-1]
            bef = s[s.index(i)-2]
            if(bef=='x' or bef=='z' or prev=='x' or prev=='z'):
                engprob*=1
            else:
                if(d[(bef,prev,i)]=={}):
                    engprob*= 1
                else:
                    engprob = engprob*((d[(bef,prev,i)])/(freq_eng_bi[bef][prev]))
            if(bef=='k' or bef=='w' or bef=='z' or prev=='k' or prev=='w' or prev=='z'):
                frprob*= 1
            else:
                if(f[(bef,prev,i)]=={}):
                    frprob*= 1
                else:
                    frprob = frprob*((f[(bef,prev,i)])/(freq_fr_bi[bef][prev]))
    if(engprob>=frprob):
        tri_list.append('English')
    else:
        tri_list.append('French')

m = nltk.FreqDist(tri_list)
accuracy_trigram = (m['English']/len(tri_list))*100
print("Accuracy of trigram model for English: "+str(accuracy_trigram))