#!/usr/bin/env python
# coding: utf-8

# # Describing sources

# https://nbviewer.jupyter.org/github/danielfrg/word2vec/blob/master/examples/word2vec.ipynb
# 
# https://github.com/danielfrg/word2vec

# # Function testing

import word2vec


# ## Create text8-phrases file, better for word2vec input according to source

with open("text8") as myfile:
    firstNlines=myfile.readlines()[0:5] #put here the interval you want


word2vec.word2phrase('text8', 'text8-phrases', verbose=True)


# ## Train word2vec model -> create word vectors in binary format

word2vec.word2vec('text8-phrases', 'text8.bin', size=100, verbose=True)


# ## Create vector clusters based on trained model

word2vec.word2clusters('text8', 'text8-clusters.txt', 100, verbose=True)


# ## Predictions

model = word2vec.load('text8.bin')


model.vocab

model.vectors.shape


#retrieve vector of individual words
model['dog'].shape


model['dog'][:10]





model.distance("dog", "cat", "animal")


# In[13]:


# get most similar words from vocab
indexes, metrics = model.similar("cleaning")


# In[14]:


indexes, metrics


# In[15]:


model.vocab[indexes]


# In[16]:


model.generate_response(indexes, metrics)


# In[17]:


model.generate_response(indexes, metrics).tolist()


# In[19]:


#Since we trained the model with the output of word2phrase 
#we can ask for similarity of "phrases", 
#basically compained words such as "Los Angeles"


# In[18]:


indexes, metrics = model.similar('los_angeles')
model.generate_response(indexes, metrics).tolist()


# In[ ]:


# anapgoes can be used to find most common pairs to vocab defined


# In[25]:


# king woman related but not having to do with man

indexes, metrics = model.analogy(pos=['king', 'woman'] , neg=['man'])
model.generate_response(indexes, metrics).tolist()


# In[26]:


indexes, metrics = model.analogy(pos=['king', 'woman'] , neg=['girl'])
model.generate_response(indexes, metrics).tolist()





clusters = word2vec.load_clusters('text8-clusters.txt')



clusters.get_words_on_cluster(90).shape



clusters.get_words_on_cluster(90)[:10]



model.clusters = clusters




indexes, metrics = model.analogy(pos=["paris", "germany"], neg=["russia"])





model.generate_response(indexes, metrics).tolist()






