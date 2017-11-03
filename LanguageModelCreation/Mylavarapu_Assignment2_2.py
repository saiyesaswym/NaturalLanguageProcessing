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

#SPANISH
freq_sp_uni=nltk.FreqDist(sp_char)
tot_sp=len(sp_char)

#ITALIAN
freq_it_uni=nltk.FreqDist(it_char)
tot_it=len(it_char)


#--------------BUILDING BIGRAM MODELS-----------------

#SPANISH
sp_bigram = list(ngrams(sp_char,2))

freq_sp_uni = nltk.FreqDist(sp_char)
freq_sp_bi = nltk.ConditionalFreqDist(sp_bigram)

#ITALIAN
it_bigram = list(ngrams(it_char,2))

freq_it_uni = nltk.FreqDist(it_char)
freq_it_bi = nltk.ConditionalFreqDist(it_bigram)


#-------------BUILDING TRIGRAM MODELS------------

#SPANISH
sp_bigram = list(ngrams(sp_char,2))
sp_trigram = list(ngrams(sp_char,3))

#ITALIAN
it_bigram = list(ngrams(it_char,2))
it_trigram = list(ngrams(it_char,3))

#------------USING THE SPANISH TEST SET--------------------
#------------PROCESSING THE WORDS IN SPANISH TEST SET---------

sp_test = ' '.join(list(spanish_test))
from  nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
test_tokens = tokenizer.tokenize(sp_test.lower())
test_tokens = [item for item in test_tokens if not item.isdigit()]


#-------------SPANISH Vs ITALIAN UNIGRAM MODEL-------
uni_list=[];
for s in test_tokens:
    spprob = 1;
    itprob = 1;
    for i in s:
        spprob = spprob * (freq_sp_uni[i]/tot_sp)
        itprob = itprob * (freq_it_uni[i]/tot_it)
    if(spprob>itprob):
        uni_list.append("Spanish")
    else:
        uni_list.append("Italian")

d = nltk.FreqDist(uni_list)
accuracy_unigram = (d['Spanish']/len(uni_list))*100
print("Accuracy of unigram model for Spanish: "+str(accuracy_unigram))


#-------------SPANISH Vs ITALIAN BIGRAM MODEL-------
bigr_list=[];
for s in test_tokens:
    spprob = 1;
    itprob = 1;
    for i in s:
        if(s.index(i)==0):
            spprob = spprob*((freq_sp_bi[' '][i])/freq_sp_uni[' '])
            itprob = itprob*((freq_it_bi[' '][i])/freq_it_uni[' '])
        else:
            bef = s[s.index(i)-1]
            if(freq_sp_uni[bef]==0):
                spprob*=1
            else:
                spprob = spprob*((freq_sp_bi[bef][i])/freq_sp_uni[bef])
            if(freq_it_uni[bef]==0):
                itprob*= 1
            else:
                itprob = itprob*((freq_it_bi[bef][i])/freq_it_uni[bef])
    if(spprob>itprob):
        bigr_list.append("Spanish")
    else:
        bigr_list.append("Italian")

l = nltk.FreqDist(bigr_list)
accuracy_bigram = (l['Spanish']/len(bigr_list))*100
print("Accuracy of bigram model for Spanish: "+str(accuracy_bigram))


#-------------ENGLISH Vs FRENCH TRIGRAM MODEL-------
from collections import defaultdict
d = defaultdict(dict)
temp=[]
for i in sp_trigram:
    if(i not in temp):
        temp.append(i)
        d[i]=1
    else:
        d[i]+=1

f = defaultdict(dict)
temp2=[]
for i in it_trigram:
    if(i not in temp2):
        temp2.append(i)
        f[i]=1
    else:
        f[i]+=1

tri_list=[];
for s in test_tokens:
    spprob = 1;
    itprob = 1;
    for i in s:
        if(s.index(i)==0):
            spprob = 1
            itprob = 1
        elif(s.index(i)==1):
            prev = s[s.index(i)-1]
            if(d[(' ',prev,i)]=={}):
                spprob*= 1
            else:
                spprob = spprob*(d[(' ',prev,i)]/freq_sp_bi[' '][prev])
            if(f[(' ',prev,i)]=={}):
                itprob*= 1
            else:
                itprob = itprob*(f[(' ',prev,i)]/freq_it_bi[' '][prev])
        else:
            prev = s[s.index(i)-1]
            bef = s[s.index(i)-2]
            if(freq_sp_bi[bef][prev]==0):
                spprob*=1
            else:
                if(d[(bef,prev,i)]=={}):
                    spprob*= 1
                else:
                    spprob = spprob*((d[(bef,prev,i)])/(freq_sp_bi[bef][prev]))
            if(freq_it_bi[bef][prev]==0):
                itprob*= 1
            else:
                if(f[(bef,prev,i)]=={}):
                    itprob*= 1
                else:
                    itprob = itprob*((f[(bef,prev,i)])/(freq_it_bi[bef][prev]))
    if(spprob>=itprob):
        tri_list.append('Spanish')
    else:
        tri_list.append('Italian')

m = nltk.FreqDist(tri_list)
accuracy_trigram = (m['Spanish']/len(tri_list))*100
print("Accuracy of trigram model for Spanish: "+str(accuracy_trigram))



#------------USING THE ITALIAN TEST SET--------------------
#------------PROCESSING THE WORDS IN ITALIAN TEST SET---------

it_test = ' '.join(list(italian_test))
from  nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
test_it_tokens = tokenizer.tokenize(it_test.lower())
test_it_tokens = [item for item in test_it_tokens if not item.isdigit()]


#-------------SPANISH Vs ITALIAN UNIGRAM MODEL-------
uni_list=[];
for s in test_it_tokens:
    spprob = 1;
    itprob = 1;
    for i in s:
        spprob = spprob * (freq_sp_uni[i]/tot_sp)
        itprob = itprob * (freq_it_uni[i]/tot_it)
    if(spprob>itprob):
        uni_list.append("Spanish")
    else:
        uni_list.append("Italian")

d = nltk.FreqDist(uni_list)
accuracy_unigram = (d['Italian']/len(uni_list))*100
print("Accuracy of unigram model for Italian: "+str(accuracy_unigram))


#-------------SPANISH Vs ITALIAN BIGRAM MODEL-------
bigr_list=[];
for s in test_it_tokens:
    spprob = 1;
    itprob = 1;
    for i in s:
        if(s.index(i)==0):
            spprob = spprob*((freq_sp_bi[' '][i])/freq_sp_uni[' '])
            itprob = itprob*((freq_it_bi[' '][i])/freq_it_uni[' '])
        else:
            bef = s[s.index(i)-1]
            if(freq_sp_uni[bef]==0):
                spprob*=1
            else:
                spprob = spprob*((freq_sp_bi[bef][i])/freq_sp_uni[bef])
            if(freq_it_uni[bef]==0):
                itprob*= 1
            else:
                itprob = itprob*((freq_it_bi[bef][i])/freq_it_uni[bef])
    if(spprob>itprob):
        bigr_list.append("Spanish")
    else:
        bigr_list.append("Italian")

l = nltk.FreqDist(bigr_list)
accuracy_bigram = (l['Italian']/len(bigr_list))*100
print("Accuracy of bigram model for Italian: "+str(accuracy_bigram))


#-------------ENGLISH Vs FRENCH TRIGRAM MODEL-------
from collections import defaultdict
d = defaultdict(dict)
temp=[]
for i in sp_trigram:
    if(i not in temp):
        temp.append(i)
        d[i]=1
    else:
        d[i]+=1

f = defaultdict(dict)
temp2=[]
for i in it_trigram:
    if(i not in temp2):
        temp2.append(i)
        f[i]=1
    else:
        f[i]+=1

tri_list=[];
for s in test_it_tokens:
    spprob = 1;
    itprob = 1;
    for i in s:
        if(s.index(i)==0):
            spprob = 1
            itprob = 1
        elif(s.index(i)==1):
            prev = s[s.index(i)-1]
            if(d[(' ',prev,i)]=={}):
                spprob*= 1
            else:
                spprob = spprob*(d[(' ',prev,i)]/freq_sp_bi[' '][prev])
            if(f[(' ',prev,i)]=={}):
                itprob*= 1
            else:
                itprob = itprob*(f[(' ',prev,i)]/freq_it_bi[' '][prev])
        else:
            prev = s[s.index(i)-1]
            bef = s[s.index(i)-2]
            if(freq_sp_bi[bef][prev]==0):
                spprob*=1
            else:
                if(d[(bef,prev,i)]=={}):
                    spprob*= 1
                else:
                    spprob = spprob*((d[(bef,prev,i)])/(freq_sp_bi[bef][prev]))
            if(freq_it_bi[bef][prev]==0):
                itprob*= 1
            else:
                if(f[(bef,prev,i)]=={}):
                    itprob*= 1
                else:
                    itprob = itprob*((f[(bef,prev,i)])/(freq_it_bi[bef][prev]))
    if(spprob>=itprob):
        tri_list.append('Spanish')
    else:
        tri_list.append('Italian')

m = nltk.FreqDist(tri_list)
accuracy_trigram = (m['Italian']/len(tri_list))*100
print("Accuracy of trigram model for Italian: "+str(accuracy_trigram))
