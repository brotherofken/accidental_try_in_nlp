# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 11:42:57 2016

@author: rakhunzy
"""

# In[]
import time
import numpy as np
import scipy as sp
import sklearn as skl
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
from sklearn.externals import joblib

import matplotlib.pyplot as plt

import pickle

# In[]
def load_vocab(path='', sfx='',vec_sfx=''):
    vocab_file = path + 'vocab' + sfx + '.txt'
    vectors_file = path + 'vectors' + sfx + vec_sfx + '.txt'

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
data_sfx = '_morethan10_wodgt_filter_ru'
#data_sfx = '_lemma_morethan10_wodgt_filter_ru'
W_norm, vocab, ivocab = load_vocab(path='../lib/GloVe/',
                                   sfx=data_sfx,
                                   vec_sfx='_50_10_5')

# In[]

mbk = MiniBatchKMeans(init='k-means++', n_clusters=750, batch_size=10000,
                      max_iter=20000, random_state=42, tol=0.0,
                      reassignment_ratio=0.000001, n_init=20,
                      max_no_improvement=100, verbose=True)
                      
#mbk = MiniBatchKMeans(n_clusters=8, init='k-means++', max_iter=100, 
#                      batch_size=100, verbose=0, compute_labels=True, 
#                      random_state=None, tol=0.0, max_no_improvement=10, 
#                      init_size=None, n_init=3, reassignment_ratio=0.01)  
mbk.fit(W_norm)


# In[]
joblib.dump(mbk, 'models/words' + data_sfx + '_mbkmeans.pkl') 

# In[]
mbk = joblib.load('models/words' + data_sfx + '_mbkmeans.pkl')

# In[]
mbk_means_labels = mbk.labels_
mbk_means_cluster_centers = mbk.cluster_centers_
mbk_means_labels_unique = np.unique(mbk_means_labels)

# In[]

dbs = DBSCAN(eps=0.5, min_samples=3, metric='euclidean', algorithm='auto', 
             leaf_size=50, p=None, n_jobs=-1)
                      
dbs.fit(W_norm[40000:60000])
#print(dbs.labels_)
print(set(dbs.labels_))

for cluster_i in set(dbs.labels_):
    if cluster_i >= 0:
        print([ivocab[i] for i in np.where(dbs.labels_ == cluster_i)[0]])
    
print(np.count_nonzero(dbs.labels_ == -1))

# In[]
from tsne import tsne
#import numpy as np
#from sklearn.manifold import TSNE
#model = TSNE(n_components=2, random_state=0, perplexity=50, verbose=True)
#np.set_printoptions(suppress=True)
#W_norm_tsne = model.fit_transform(W_norm[:10000]) 

W_norm_tsne = tsne(W_norm, no_dims = 2, initial_dims = 50, perplexity = 30.0)

# In[]
plt.scatter(W_norm_tsne[:, 0], W_norm_tsne[:, 1])
plt.show()

# In[]
joblib.dump(mbk, 'models/words' + data_sfx + '_dbscan.pkl') 

# In[]
dbs = joblib.load('models/words' + data_sfx + '_dbscan.pkl')

# In[]

mbk_means_labels = dbs.labels_

# In[]
cluster_sizes = np.array([np.count_nonzero(mbk_means_labels == i) for i in mbk_means_labels_unique])

# In[]
def get_cluster(c):
    return list(filter(lambda i: i[1] == c, enumerate(mbk_means_labels)))


# In[]

word = 'друг'
word_cluster = mbk_means_labels[vocab[word]]
print(word)
print(word_cluster)
cluster = get_cluster(word_cluster)
cluster_words = [ivocab[i] for i, c in cluster]

# In[] Read sentences and calculate features:

#dataset_file = 'tokens_morethan10_wodgt_filter_ru.txt'
dataset_file = 'tokens' + data_sfx + '.txt'
i = 0

feat_vectors = []
feat_indexes = []

X = np.zeros((0, len(mbk_means_labels_unique) + 1))

with open(dataset_file, 'r') as fi:
    for string in fi:
        string = string.rstrip('\n')
        
        feat_vector = np.zeros(len(mbk_means_labels_unique) + 1)
        
        if len(string) > 0:
            tokens = string.split(' ')
            
            
            
            for token in tokens:
                if token in vocab:
                    feat_vector[mbk_means_labels[vocab[token]]] += 1
                else:
                    feat_vector[-1] += 1
            
#            print()
#            print(tokens)
#            print(feat_vector)
            
            feat_vectors.append(feat_vector)

            if len(feat_vectors) == 100000:
                feat_vectors = np.vstack(feat_vectors)
                X = np.vstack((X, feat_vectors))            
                feat_vectors = []
            
            feat_indexes.append(i)
        #if i >= 100: break
        
        i += 1
        
        if i % 100000 == 0:
            print("Reading %d" % i)

feat_vectors = np.vstack(feat_vectors)
X = np.vstack((X, feat_vectors))

# In[]
#X = np.vstack(feat_vectors)
X_indices = np.array(feat_indexes)
#np.save('data/text_feat_vectors.npy', X)
#np.save('data/text_feat_vectors_indices.npy', X_indices)
del feat_vectors
del feat_indexes
# In[]

X = np.load('data/text_feat_vectors.npy')
X_indices = np.load('data/text_feat_vectors_indices.npy')

# In[]
np.clip(X, 0, 5, X)

# In[]

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X_tfidf = transformer.fit_transform(X)

# In[]
X_tfidf = X

# In[]

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=20, random_state=42)
svd.fit(X_tfidf)
#svd.fit(X)

# In[]
print(svd.explained_variance_ratio_)
print(np.sum(svd.explained_variance_ratio_))

# In[]
X_tfidf_tsvd = svd.transform(X_tfidf)

X_tfidf_tsvd = X_tfidf_tsvd/svd.explained_variance_ratio_

# In[]

X_tfidf_tsvd_subset = X_tfidf_tsvd[np.random.choice(len(X_tfidf_tsvd), 10000)]

# In[] Plot
import seaborn as sns
import pandas as pd
sns.set()

sns.pairplot(pd.DataFrame(data=X_tfidf_tsvd_subset[:,:10]))

# In[]
import matplotlib.pyplot as plt
g = sns.PairGrid(pd.DataFrame(data=X_tfidf_tsvd_subset[:,:5]))
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_diag(sns.kdeplot, lw=3, legend=False);

# In[]

text_mbk = MiniBatchKMeans(init='k-means++', n_clusters=500, batch_size=10000,
                      max_iter=1000, random_state=42, reassignment_ratio=0.000001,
                      n_init=10, max_no_improvement=10, verbose=True)
                      
text_mbk.fit(X_tfidf_tsvd)

# In[]
from sklearn.externals import joblib
joblib.dump(text_mbk, 'models/texts' + data_sfx + 'mbkmeans.pkl') 

# In[]
text_mbk = joblib.load('models/texts' + data_sfx + 'mbkmeans.pkl')

# In[]
text_mbk_means_labels = text_mbk.labels_
text_mbk_means_cluster_centers = text_mbk.cluster_centers_
text_mbk_means_labels_unique = np.unique(text_mbk_means_labels)

def get_text_cluster(c):
    return list(filter(lambda i: i[1] == c, enumerate(text_mbk_means_labels)))

# In[]
cluster_sizes = np.array([np.count_nonzero(text_mbk_means_labels == i) for i in text_mbk_means_labels_unique])


# In[]
for cluster_i in text_mbk_means_labels_unique:
    cluster = get_text_cluster(cluster_i)
    cluster = [X_indices[i[0]] for i in cluster]
    print('Checking %d' % cluster_i)
    if 3600727 in cluster:
        print('Found: %d' % cluster_i)
        break

# In[]
cluster = get_text_cluster(0)
cluster = [X_indices[i[0]] for i in cluster]

cluster_texts = []
dataset_file = 'tokens' + data_sfx + '.txt'

i = 0
j = 0
with open(dataset_file, 'r') as fi:
    for string in fi:
        string = string.rstrip('\n')
        
        if i == cluster[j]:
            tokens = string.split(' ')
            cluster_texts.append(' '.join(tokens))
            j += 1
            j = j % len(cluster)
        i += 1
        
        if i % 1000000 == 0:
            print("Reading %d" % i)