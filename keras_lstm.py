import time
start_time = time.time()

import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


#Importing the Dataset
train = pd.read_csv('train.csv').fillna(' ')
test = pd.read_csv('test.csv').fillna(' ')
test_labels = pd.read_csv('test_labels.csv')
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

y = train[label_cols].values

comment_train = train["comment_text"]
comment_test = test["comment_text"]


#train.isnull().any(),test.isnull().any()

#Converting each comment text to indexes
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(comment_train))

tokenized_train = tokenizer.texts_to_sequences(comment_train)
tokenized_test = tokenizer.texts_to_sequences(comment_test)

#Padding short data and shortening longer ones
maxlen = 200
X_train = pad_sequences(tokenized_train, maxlen=maxlen)
X_test = pad_sequences(tokenized_test, maxlen=maxlen)

inp = Input(shape=(maxlen, ))

embed_size = 128
x = Embedding(max_features, embed_size)(inp)

x = LSTM(60, return_sequences=True,name='lstm_layer')(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

batch_size = 32
epochs = 2
model.fit(X_train,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
pred = model.predict(X_test, batch_size = batch_size, verbose = 1)

submission = pd.read_csv("sample_submission.csv")
submission[label_cols] = (pred)
submission.to_csv("submission.csv", index = False)
print("[{}] Completed!".format(time.time() - start_time))
