# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

dfspam_messages = pd.read_csv("SMSSpamCollection",delimiter="\t" ,names=['Label','msg'])

# Data cleaning and preprocessing

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
corpus = []

for i in range(0,len(dfspam_messages)):
    review = re.sub('[^a-zA-Z]',' ',dfspam_messages.msg[i])
    review = review.lower()    
    review = review.split()    
    
    stopwords_eng = stopwords.words('english')
    review = [ps.stem(word) for word in review if not word in stopwords_eng]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()

Y = pd.get_dummies(dfspam_messages.Label,drop_first=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Training model using naive bayes classifier
# Naive bayes works very well with nlp

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)

y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_pred,y_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred,y_test)

# Accuracy score using stemming and count vectorizer is 0.9811659192825112




