# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:07:04 2021

@author: dhemanna
"""
import pandas as pd
import sklearn
import nltk

dfmess = pd.read_csv("SMSSpamCollection",delimiter='\t',names=['label','message'])

import re
from nltk.corpus import stopwords
import nltk
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()

corpus = []
for i in range(0,len(dfmess.message)):
    message = re.sub('[^a-zA-Z]',' ',dfmess.message[i])
    message = message.lower()
    message = message.split()
    message = [lemm.lemmatize(mess) for mess in message if mess not in stopwords.words('english')]
    message = ' '.join(message)
    corpus.append(message)
    
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(corpus).toarray()

y = pd.get_dummies(dfmess.label,drop_first = True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=43)

from sklearn.naive_bayes import MultinomialNB
gnb = MultinomialNB()
gnb_model = gnb.fit(X_train,y_train)
y_train_pred = gnb_model.predict(X_train)
y_test_pred = gnb_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_test_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test_pred,y_test)

# Accuracy score using lemmetization and TFIDF vectorizer is 0.9704035874439462
