#Import Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression #Trying this out first. To check
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer #Im trying the same Tfidf method as implemented in the second assingment, to check if tokenization works
import re, string
from scipy.sparse import hstack
from sklearn.model_selection import cross_val_score
from sklearn import svm


#Importing the Dataset
train = pd.read_csv('train.csv').fillna(' ')
test = pd.read_csv('test.csv').fillna(' ')
test_labels = pd.read_csv('test_labels.csv')
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

comment_train = train["comment_text"]
comment_test = test["comment_text"]

#Combining both the comments from train and test, to identify tokens. Ref: Kaggle Notebook
all_comment = pd.concat([comment_train, comment_test])

#Cleaning the Dataset of empty comments
train['comment_text'].fillna("unknown", inplace=True)
test['comment_text'].fillna("unknown", inplace=True)


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_comment)
train_word_features = word_vectorizer.transform(comment_train)
test_word_features = word_vectorizer.transform(comment_test)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_comment)
train_char_features = char_vectorizer.transform(comment_train)
test_char_features = char_vectorizer.transform(comment_test)

print(train_word_features)

#Yet to look into the logic of these below codes  in detail, Havent executed them to understand it completely. Took them up from the notebooks. You can try it to check the validit and understanding
train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']}) 
for class_name in label_cols:
    train_target = train[[class_name]]
    #classifier = LogisticRegression(C=0.1, solver='sag')
    classifier = xgboost.XGBClassifier(random_state=42)
    print(train_target.shape)
    print(train_features.shape)

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=5, scoring='roc_auc'))
    scores.append(cv_score)
    print('Logistic Regression - CV score for class {} is {}'.format(class_name, cv_score))


    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))