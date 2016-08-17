# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 22:33:25 2016

@author: rakhunzy
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 21:44:28 2016

@author: rakhunzy
"""

# In[]:
#import nltk
#nltk.download()
# In[]:

from tqdm import tqdm

# In[]:

dataset_file = 'tokens.txt'
dataset_len = 10000000

# In[]:



# In[]:

from nltk.stem.snowball import RussianStemmer
lmtzr = RussianStemmer()

def stem(sentence):
    return [lmtzr.stem(word) for word in sentence]
# In[]

from multiprocessing import Pool

# In[]
if __name__ == '__main__':
# In[]
    p = Pool(8)
    #sentences = [tokenize(s) for s in tqdm(lines)]

    fo = open('tokens_stem.txt','w')
    
    lines = []

    i = 1
    with open(dataset_file, 'r') as fi:
        for string in fi:
            if i % 1000000 == 0:
                print("Reading %d" % i)
            
            lines.append(string.rstrip().split(','))
           
            if i % 1000000 == 0:
                #break
                sentences = p.map(stem, tqdm(lines))
                lines = []
                for s in sentences:
                    print(','.join(s), file=fo)

            i += 1
            
    print("Done")    
    

    f.close() # you can omit in most cases as the destructor will call i
