#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[6]:


DATA_IN_PATH = './clean_data/'
TRAIN_INPUT_DATA = 'train_input.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
TRAIN_CLEAN_DATA = 'train_clean.csv'
DATA_CONFIGS = 'data_configs.json'


trainData = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)
article = list(trainData['article'])
label = np.array(trainData['category'])


# In[8]:


vectorizer = CountVectorizer(analyzer='word', max_features=5000)
trainDataFeatures = vectorizer.fit_transform(article)

trainDataFeatures


# In[18]:


trainInputData, testInputData, trainInputLabel, testInputLabel = train_test_split(trainDataFeatures, label, test_size = 0.2, random_state = 42)


# In[19]:


print('학습 시작')
forest = RandomForestClassifier(n_estimators=100)
forest.fit(trainInputData, trainInputLabel)


# In[20]:


print(forest.score(testInputData, testInputLabel))


# In[1]:


# print(testInputData[0])
# print(testInputLabel[0])

print(testInputLabel[90:100])
print(forest.predict(testInputData[90:100]))
print(forest.predict_proba(testInputData[90:100]))


def predictText(data):
    print(forest.predict(data))
    print(forest.predict_proba(data))

    

