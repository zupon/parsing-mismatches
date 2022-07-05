#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from tqdm import tqdm
from collections import Counter
import functions as f


# ### Set hyperparameters

# In[2]:


# Select number of similar words to create new word pairs
top_n = 20
# threshold = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0,0]
threshold = 0.0

# Select seed for training data (1-3)
seed = 1

# Select noise method
noise="none"
# noise="freq"
# noise="10_types"
# noise="10_tokens"


# ### Load corpora

# In[3]:


# corpus_A, corpus_B = f.load_corpus(language)


# ### Load vectors

# In[4]:


vectors_dir = "./gd-ga-en-vecs/"
vectors_gd = vectors_dir+"src=gd_tgt=ga/vectors-gd-aligned-w-ga.magnitude"
vectors_ga = vectors_dir+"src=gd_tgt=ga/vectors-ga-aligned-w-gd.magnitude"


# ### Get word-word-relation triples for each corpus

# In[5]:


corpus_gd = "train_data/gd/gd_noise="+str(noise)+"-ud-train.conllu"
corpus_ga = "train_data/ga/ga_noise=none-ud-train.conllu"
# corpus_en = "train_data/en/en_noise=none-ud-train.conllu"


# In[6]:


list_gd, sentences_gd = f.process_training_data(corpus_gd)


# In[7]:


list_ga, sentences_ga = f.process_training_data(corpus_ga)


# In[8]:


# output_A = language+"_A_mismatches"+".tsv"
# output_B = language+"_B_mismatches"+".tsv"
# get list for corpus A data
# list_A, sentences_A = f.process_ud_data(corpus_A, num_sents_A)
# get list for corpus B data
# need to keep process_wsj_data for now for English data
# list_B, sentences_B = f.process_wsj_data(corpus_B, num_sents_B)
# list_B, sentences_B = f.process_ud_data(corpus_B, num_sents_B)


# ### Load text dictionaries

# In[9]:


gd2ga_dict = "gd2ga_dict.txt"
# # gd2en = ""
# gd2ga_dict = {}
# with open(gd2ga) as f:
#     dictionary = f.readlines()
# for item in dictionary:
#     item = item.strip().lower().split("\t")
#     if item[0] in gd2ga_dict:
#         gd2ga_dict[item[0]].append(item[1])
#     else:
#         gd2ga_dict[item[0]] = [item[1]]


# In[10]:


# print(gd2ga_dict)


# ### Pretrained Word Vectors

# #### Create converted files

# In[11]:


conversion_gd = f.get_conversions_bilingual_dict(sentences_gd,
                                                 list_gd, list_ga,
                                                 gd2ga_dict)

# converted_corpus_gd = corpus_gd[:-4]+"_converted_bilingual_dict.conllu"
# f.apply_conversions(corpus_gd, converted_corpus_gd, conversion_gd)


# In[ ]:


conversion_gd = f.get_conversions_bilingual_vectors(sentences_gd,
                                                    list_gd, list_ga,
                                                    vectors_gd, vectors_ga)


# In[ ]:


conversion_gd = f.get_conversions_bilingual_dict_vectors(sentences_gd,
                                                         list_gd, list_ga,
                                                         vectors_gd, vectors_ga,
                                                         gd2ga_dict)


# In[ ]:


conversion_gd = f.get_conversions_bilingual_vectors_vectors(sentences_gd,
                                                            list_gd, list_ga,
                                                            vectors_gd, vectors_ga)


# In[ ]:


# conversion_gd = f.get_conversions_bilingual(sentences_gd, 
#                                             list_gd, 
#                                             list_ga,
#                                             vectors_gd,
#                                             vectors_ga,
#                                             top_n)
# converted_corpus_gd = corpus_gd[:-4]+"_converted_pretrained="+vector_type+".conllu"
# f.apply_conversions(corpus_gd, converted_corpus_gd, conversion_gd)


# #### Create converted files for corpus A and corpus B

# In[ ]:


# converted_corpus_A_pretrained = corpus_A[:-4]+"_converted_pretrained="+vector_type+".conllu"
# converted_corpus_B_pretrained = corpus_B[:-4]+"_converted_pretrained="+vector_type+".conllu"


# In[ ]:


# f.apply_conversions(corpus_A, converted_corpus_A_pretrained, conversion_A_pretrained)
# f.apply_conversions(corpus_B, converted_corpus_B_pretrained, conversion_B_pretrained)


# In[ ]:


conversion_B_BERT = f.get_conversions_pretrained(mismatches_B, 
                                                       sentences_B, 
                                                       list_B, 
                                                       list_A,
                                                       vectors,
                                                       top_n,
                                                       threshold)
converted_corpus_B_BERT = corpus_B[:-4]+"_converted_BERT.conllu"
f.apply_conversions(corpus_B, converted_corpus_B_BERT, conversion_B_BERT)


# In[ ]:




