
# coding: utf-8

# In[ ]:

import nltk
nltk.download()


# In[3]:

import csv
import numpy as np
import matplotlib.pyplot as plt


# In[4]:

dataset_file = '../data/posts_dump_v1.txt'
dataset_len = 10000000


# In[ ]:




# In[26]:

early_break = False #True

line_lengths = np.zeros(dataset_len)
lines = [''] * dataset_len

i = 0
with open(dataset_file, 'r') as csvfile:
    textreader = csv.reader(csvfile, delimiter=',')
    for num, string in textreader:
        #print len(string), string
        #line_lengths[i] = len(string)
       
        if i % 1000000 == 0:
            print("Done %d" % i)
        if early_break and i == 10:
            break
        lines[i] = string
        i += 1
print("Done")


# In[28]:

lines[13]
#fig = plt.Figure()
#plt.hist(line_lengths, bins=100, range=(10,400))
#plt.show()


# In[29]:

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import RussianStemmer


# In[32]:

tokenizer = RegexpTokenizer(r'\w+')
lmtzr = RussianStemmer()
sentences = [tokenizer.tokenize(sentence.lower()) for sentence in lines]
#words = [set(lmtzr.stem(word) for word in tokenizer.tokenize(sentence.lower())) for sentence in lines]


# In[ ]:

#stemmed_sentences = [[lmtzr.stem(word) for word in sentence] for sentence in sentences[:10000]]

import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor()
stemmed_sentences = list(executor.map(lambda sentence: [lmtzr.stem(word) for word in sentence], sentences, chunksize=100000))


# In[ ]:

#list(map(lambda x:print(x), sentences[:100]))


# In[61]:

i = 115
print(sentences[i])
print(stemmed_sentences[i])


# In[ ]:



