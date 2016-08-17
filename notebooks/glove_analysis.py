# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 20:56:18 2016

@author: rakhunzy
"""

# In[]

import numpy as np

# In[]

def load_vocab(path='',sfx='',vector_sfx=''):
    vocab_file = path + 'vocab' + sfx + '.txt'
    vectors_file = path + 'vectors' + sfx + vector_sfx + '.txt'

    with open(vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2.0, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)

# In[]

W_norm, vocab, ivocab = load_vocab('../lib/GloVe/', '_morethan5_wodgt', '_100_5_10')

# In[]

def distance(W, vocab, ivocab, input_term):
    for idx, term in enumerate(input_term):
        if term in vocab:
            #print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = W[vocab[term], :] 
            else:
                vec_result += W[vocab[term], :] 
        else:
            print('Word: %s  Out of dictionary!' % term)
            #return
    
    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2.0,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    dist = np.dot(W, vec_norm.T)

    for term in input_term:
        if term in vocab:
            index = vocab[term]
            dist[index] = -np.Inf

    a = np.argsort(-dist)[:10]

    print("                               Word       Cosine distance")
    for x in a:
        print("%35s\t\t%f" % (ivocab[x], dist[x]))



def nearest_vector(W, vocab, ivocab, vec_result):
    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2.0,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    dist = np.dot(W, vec_norm.T)

    #dist[index] = -np.Inf

    a = np.argsort(-dist)[:10]

    for x in a:
        print("%10f\t%s" % (dist[x], ivocab[x]))
        
    return dist
    

# In[]

def mean_vector(W, vocab, ivocab, input_tokens):
    count = 0
    vec_result = []
    for idx, term in enumerate(input_tokens):
        if term in vocab:
            #print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = W[vocab[term], :] 
            else:
                vec_result += W[vocab[term], :]
            count += 1
        else:
            pass
            #print('Word: %s  Out of dictionary!\n' % term)       

    if count > 0:
        return vec_result / count
    else:
        return np.ones(W[0].shape)
        
# In[]
        
#distance(W_norm, vocab, ivocab, 'хер')

dataset_file = 'tokens.txt'
i = 1
with open(dataset_file, 'r') as fi:
    for string in fi:
        if i < 5000001:
            i += 1
            continue
            
        string = string.rstrip().split(',')
        skip = len(string) < 10
    
        if i % 10 == 0: break
        if not skip: 
            print()
            print(' '.join(string))
            mv = distance(W_norm, vocab, ivocab, string)
            #nv = nearest_vector(W_norm, vocab, ivocab, mv)
            i += 1