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
from multiprocessing import Pool

import task_utils

# In[]:

dataset_file = '../data/posts_dump_v1.txt'
dataset_len = 10000000

# In[]:

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import RussianStemmer

tokenizer = RegexpTokenizer(r'\w+')
lmtzr = RussianStemmer()

# In[]:

def filter_bad_token(token):
    alphabets = task_utils._alphabet_detector.detect_alphabet(token)
    
    # fix words with mixed alphabet 
    token = task_utils.normalize2cyr(token, alphabets)

    # remove suspicious and strange words
    if task_utils.short_latin_word(token, alphabets):
        token = ''
    if task_utils.short_rus_word(token, alphabets):
        token = ''
    if task_utils.is_stop_word(token):
        token = ''
    return token

def tokenize(sentence):
    tokens = tokenizer.tokenize(task_utils.remove_digits(sentence.lower()))

    # remove links
    tokens = list(filter(lambda w : not w.startswith('http'), tokens))
    # coalesce char seq
    tokens = list(map(task_utils.coalesce_char_sequences, tokens))
    
    tokens = list(filter(lambda v: len(v) > 0, map(filter_bad_token, tokens)))
    
    if len(tokens) < 10:
        tokens = []
    else:
        if not task_utils.is_ru(' '.join(tokens[:10])):
            tokens = []    
        
    return tokens     

# In[]
if __name__ == '__main__':
    lines = [''] * dataset_len

    print("Reading")
    i = 0
    with open(dataset_file, 'r') as csvfile:
        textreader = csv.reader(csvfile, delimiter=',')
        for num, string in textreader:
            if i % 1000000 == 0:
                print(i)
    
            lines[i] = string
            i += 1
    print("Done")    
    
    p = Pool(8)
    sentences = p.map(tokenize, tqdm(lines))
    
    f = open('tokens_morethan10_wodgt_filter_ru.txt','w')

    for sentence in tqdm(sentences):
        print(' '.join(sentence), file=f)

    f.close() # you can omit in most cases as the destructor will call i