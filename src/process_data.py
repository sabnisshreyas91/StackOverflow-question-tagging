#!/usr/bin/env python
# coding: utf-8

# In[57]:


import os
import math
import numpy as np
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import re
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from collections import Counter
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
import argparse
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')


# In[58]:


# parser = argparse.ArgumentParser()
# parser.add_argument("source_file_path", help="The path of the 'Train.csv' file inside the 'facebook-recruiting-iii-keyword-extraction' zip file")
# args = parser.parse_args()


# In[59]:


df = pd.read_csv("data\\Train.csv", nrows = 5000000)


# In[60]:


df = df[df.Tags.isnull() == False]


# In[61]:


df['Tags_split'] = df['Tags'].str.split(" ")
df['ct'] = df['Tags_split'].str.len()


# In[69]:


tags_grp = df.groupby('Tags', as_index=False).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False).head(100)
incl_labels = tags_grp['Tags'].tolist()


# In[70]:


# Filter for top 100 labels
df = df[df.Tags.isin(incl_labels)]


# In[74]:


def remove_stop_lemmatize(text):
    text = ' '.join([word for word in text.split()
                      if word not in stop])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


# In[75]:


def normalize(file_str):
    # 1) remove numbers and spacial characters
    file_str = re.sub(r'([^a-zA-Z\s]+?)', '', file_str).replace("\n",' ')
    # 2) Lower case
    file_str = file_str.lower()
    return file_str


# In[76]:


def rem_html_tags(body):
    regex = re.compile('<.*?>')
    return re.sub(regex, '', body)


# In[77]:


df = df[df.Tags.isnull() == False]
df['Title'] = df['Title'].apply(rem_html_tags).apply(normalize).apply(remove_stop_lemmatize)
df['Body'] = df['Body'].apply(rem_html_tags).apply(normalize).apply(remove_stop_lemmatize)


# In[78]:


df['Text'] = df['Title']+" "+df['Body']


# In[79]:


df[['Text']].to_csv("data\\Questions.csv", index=False)
#df[['Title','Body']].to_csv("Questions.csv", index=False)


# In[80]:


labels_lst_of_lst = df.Tags_split.tolist()

labels = (list(set([a for b in labels_lst_of_lst for a in b])))

multilabel_binarizer = MultiLabelBinarizer(classes = labels)
labels_oh = multilabel_binarizer.fit_transform(labels_lst_of_lst)

labels_df = pd.DataFrame(labels_oh, columns = multilabel_binarizer.classes_.tolist())


# In[84]:


labels_df.to_csv("data\\Labels.csv", index=False)


# In[81]:


label_weights = labels_df.sum()/len(labels_df)
label_weights = label_weights.reset_index()
label_weights.rename(columns = {'index':'label',0:'weight'}, inplace=True)
label_weights['weight'] = 1/label_weights['weight']


# In[82]:


#labels_df.to_csv("Labels.csv", index=False)
df[['Tags_split']].to_csv('Labels.csv', index=False)
label_weights.to_csv("data\\label_weights.csv", index=False)

