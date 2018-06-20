#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 22:12:39 2018

@author: studyrelated
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk

#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 

#Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter ='\t', quoting=3)

#Collection of words
corpus = []
##cleaning text
for i in range(0,1000):    
    #Not remove all letters from a-z and A-Z
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    #Transform in lowercase string
    review = review.lower()
    #Split into single words
    review = review.split()
    #Create PorterStemmer
    ps = PorterStemmer()    
    #Remove all adjectives, articles, prepositions on stemmed words
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #Join words into string seperated by 2 spaces
    review = ' '.join(review)
    corpus.append(review)

##Creating Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) 
X = cv.fit_transform(corpus).toarray()
#Creating y variable 
y = dataset.iloc[:,1].values

##Applying Logistic Regression Algorithm to dataset
"""
I runned 6 classifiers. a=accuracy%, p=precision%
Random Forest: a=68, p=61
Naive Bayes: a=71, p=77
K-NN: a=65, p=73
SVM 'rbf': a=74, p=74
SVM Kernel: a=74, p=76
Logistic Regression: a=76, p=74 

Either SVM Kernel or Logistic Regression. SVM preferred with larger datasets.

"""

#Create test/traing set from data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state= 0)

#Feature Scaling x_train and x_test (Z-scores)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Create the classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train) #learning correlations to predict new observations

#Predict y based on x_test
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix to evaluate the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Logistic Regression: a=76%, p:74%


