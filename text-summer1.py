
# coding: utf-8

# In[8]:


import numpy as np
from nltk.corpus import stopwords
# import nltk
# nltk.download()
from sklearn.cluster import KMeans
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import requests
from gensim.summarization import summarize
import re
import pandas as pd

# In[3]:


# k = number of clusters
# num_runs  = number of times k-means runs with different initial centroids
# num_iter  = number of iterations in each single run
# X = number of sentences X number of distinct features used
def k_mean_clust(X, k, num_runs, num_iter):
    kmean = KMeans(n_clusters = k, n_init = num_runs, max_iter = num_iter).fit(X)
    return kmean


# In[5]:


print(stopwords.words('english'))


# In[16]:


# text  = 'This is Ram.This is shyam'
# cluster_text(text)

def preprocess_text(text):
    text = text.lower()
#     sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)

    return text


# In[9]:


# text  = 'This is Ram.This is shyam'
# cluster_text(text)

def cluster_text(text, k, num_runs, num_iter):
    # print(text[0])
    n = len(text)
    # print(n)

    all_sent = {}
    all_vec = {}
    for i in range (n):
        all_sent['sent'+str(i+1)] = [w.lower() for w in text[i]]

    final_list = all_sent['sent1']
    for i in range(1,n):
        final_list = final_list + all_sent['sent'+str(i+1)]
    # print(final_list)
    all_words = list(set(final_list))
    # print(all_words)
    
    for i in range (0,n):
        all_vec['vector'+str(i+1)] = [0]*len(all_words)

    for i in range (0,n):    
        for w in all_sent['sent'+str(i+1)]:
        #     if w in stopwords:
        #         continue
            all_vec['vector'+str(i+1)][all_words.index(w)] += 1
    # print(all_vec)

    X = np.array(list(all_vec.values()))
    # print(X)

    kmean = k_mean_clust(X, k, num_runs, num_iter)
    return list(kmean.labels_)


# In[17]:


text = "Mr. Smith bought cheapsite.com for 1.5 million dollars, i.e. he paid a lot for it. Did he mind? Adam Jones Jr. thinks he didn't. In any case, this isn't true... Well, with a probability of .9 it isn't."
text = preprocess_text(text)
for sentences in text:
    print(sentences)


# In[18]:


l = [1,3,4,3,5]
l.remove(3)
print(l)

