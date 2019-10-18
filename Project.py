#Import Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression #Trying this out first. To check
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer #Im trying the same Tfidf method as implemented in the second assingment, to check if tokenization works
import re, string

#Trying out Keras
import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

#Importing the Dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_labels = pd.read_csv('test_labels.csv')
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

comment_train = train["comment_text"]
comment_test = test["comment_text"]


#Cleaning the Dataset of empty comments

train['comment_text'].fillna("unknown", inplace=True)
test['comment_text'].fillna("unknown", inplace=True)


#tokenization of each words uning keras
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(comment_train))
list_tokenized_train = tokenizer.texts_to_sequences(comment_train)
list_tokenized_test = tokenizer.texts_to_sequences(comment_test)

#YET TO TRY FURTHER PACKAGES OF KERAS. ON HOLD TILL Project_Trial is Done
