# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 21:44:28 2016

@author: rakhunzy
"""

# In[]:
#import nltk
#nltk.download()
# In[]:

import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# In[]:

dataset_file = '../data/posts_dump_v1.txt'
dataset_len = 10000000


# In[]:

lines = [''] * dataset_len

i = 0
with open(dataset_file, 'r') as csvfile:
    textreader = csv.reader(csvfile, delimiter=',')
    for num, string in textreader:
        if i % 1000000 == 0:
            print("Reading %d" % i)

        lines[i] = string
        i += 1
print("Done")

# In[]:

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import RussianStemmer

tokenizer = RegexpTokenizer(r'\w+')
lmtzr = RussianStemmer()

# In[]:

def tokenize(sentence):
    return tokenizer.tokenize(sentence.lower())

def flatten(listoflists):
    return [val for sublist in sentences for val in sublist]

# In[]

from multiprocessing import Pool

# In[]
if __name__ == '__main__':
# In[]
    p = Pool(8)
    sentences = p.map(tokenize, tqdm(lines))
    #sentences = [tokenize(s) for s in tqdm(lines)]

    f = open('tokens.txt','w')

    for s in sentences:
        print(','.join(s), file=f)

    f.close() # you can omit in most cases as the destructor will call i


## In[]:
#
#def stem(sentence):
#    return [lmtzr.stem(word) for word in sentence]
#
#stemmed_sentences = [stem(s) for s in tqdm(sentences)]
#
## In[]
#
#i = 115
#print(sentences[i])
#print(stemmed_sentences[i])