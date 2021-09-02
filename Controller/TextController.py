#!/usr/bin/env python
# coding: utf-8

# In[33]:


import os 
import json
import pandas as pd 
import numpy as np
from konlpy.tag import Okt
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
# from sklearn.model_selection import train_test_split
import numpy as np


# In[4]:


DATA_IN_PATH = './data_in/'
DATA_CONFIGS = 'data_configs.json'

okt=Okt()
stopWords = ['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한']
preproConfigs = json.load(open(DATA_IN_PATH + DATA_CONFIGS, 'r'))

#기존에 학습할떄 사용한 토큰용 사전을 입력한다
tokenizer = Tokenizer()
tokenizer.word_index = preproConfigs['vocab']

def categoryToTarget(text):
    if text == "판매":
        return 1
    elif text == "교환":
        return 2
    elif text == "구입":
        return 3
    elif text == "거래완료":
        return 4
    elif text == "그냥드림":
        return 5
    else:
        return 100


# In[5]:


def preprocessing(text, okt, remove_stopwords= False, stop_words=[]):
    #줄바꿈 문자 삭제
    text = text.replace("\n","")
    wordText = okt.morphs(text, stem=True)
    
    if remove_stopwords:
        wordText = [token for token in wordText if not token in stop_words]
        
    return wordText


# In[64]:


# test = preprocessing('\uc544\uc774\ud3f0 se 1\uc138\ub300 32g \uc2a4\uadf8 (\ucee8\ud2b8\ub9ac\ub77d, \uce74\uba54\ub77c \ubb34\uc74c)\n\n\ucd5c\uc2e0 \ud5e4\uc774\uc2ec \ud3ec\ud568\ud574\uc11c \ub4dc\ub9bd\ub2c8\ub2e4. \uc624\ub298\uae4c\uc9c0\ub3c4 \ud65c\uc131\ud654 \ubb38\uc81c \uc5c6\ub124\uc694.\n\uc0ac\uc6a9\ubc95 \uc544\uc2dc\ub294\ubd84\uc774 \uad6c\ub9e4\ud558\uc168\uc73c\uba74 \uc88b\uaca0\uc2b5\ub2c8\ub2e4.\n\ubc30\ud130\ub9ac \uc131\ub2a5 87%, \ub4e4\ub738 \uc5c6\uc74c, \uc9c0\ubb38\uc778\uc2dd \uc591\ud638, \uc0ac\uc124\uc218\ub9ac\uc774\ub825 \uc5c6\uc74c\n\ube48\ubc15\uc2a4, sword pro \uc54c\ub8e8\ubbf8\ub284 \ubc94\ud37c, \ud22c\uba85\ucf00\uc774\uc2a4, \uc561\ubcf4 3\uc7a5 \uad6c\uc131\n\n\ubb34\ubcf4\uc815 \uc0c1\uc138\uc0ac\uc9c4\nhttps://www.dropbox.com/sh/ukafjbzyab5p13p/AADzOArcCDtRNoTKBnJ-6ffMa?dl=0', okt, remove_stopwords=True, stop_words=stopWords)


# print(textToToken(test))


# In[63]:


def textToToken(text):
    tokenizerArray = tokenizer.texts_to_sequences(text)
    npArray = np.array([])

    for inToken in tokenizerArray:
        if len(inToken) == 1:
            # print(inToken[0])
            # inToken.astype(np.int32)
            npArray = np.append(npArray, np.array(int(inToken[0])))
            npArray.astype(np.int32)
            # print(npArray)
    return npArray


# In[32]:


def test():
    print('ttt!! aabbcc')


