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

from tqdm import tqdm

# In[]:

dataset_file = 'tokens_morethan10_wodgt_filter_ru.txt'
dataset_len = 10000000

# In[]:

import pymorphy2
morph = pymorphy2.MorphAnalyzer()

def lemma(sentence):
    return [morph.parse(word)[0].normal_form for word in sentence]
# In[]


# In[]



# In[]

from multiprocessing import Pool

# In[]
if __name__ == '__main__':
# In[]
    p = Pool(8)
    #sentences = [tokenize(s) for s in tqdm(lines)]

    fo = open('tokens_lemma_morethan10_wodgt_filter_ru.txt','w')
    
    lines = []

    i = 1
    with open(dataset_file, 'r') as fi:
        for string in fi:
            if i % 1000000 == 0:
                print("Reading %d" % i)
            
            lines.append(string.rstrip().split(' '))
           
            if i % 1000000 == 0:
                #break
                sentences = p.map(lemma, tqdm(lines))
                lines = []
                for s in sentences:
                    print(' '.join(s), file=fo)

            i += 1
            
    print("Done")    
    

    fo.close() # you can omit in most cases as the destructor will call i
