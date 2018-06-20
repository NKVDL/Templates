#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 21:14:42 2018

@author: studyrelated
"""
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Create SVM - Train on training set - I chose a non-linear classification;overall sensitivity,specificity increased
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

#Predict with SVM
y_pred = classifier.predict(X_test) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Applying k-Fold-Cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=10)
accuracies.mean() # = 0.90053021876158679, avg of model accuracy
accuracies.std() # = 0.063889573566262847 , SD 

#Applying Grid Search for best model and parameters
from sklearn.model_selection import GridSearchCV 
parameters = [{'C' : [1,10,100,1000], 'kernel' : ['linear']},#linear
               {'C' : [1,10,100,1000], 'kernel' : ['rbf'], 'gamma': [0.5, 0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9]}] #non-linear 
#Finding C in SVC, optimising parameters,
#with a linear kernel for N parameters 1,10,100 or 1000, or non-linear
grid_search = GridSearchCV(estimator =classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10, #10xcross-validation 
                           n_jobs = -1)                        
grid_search = grid_search.fit(X_train, y_train)                           

#Optimal accuracy of all models
best_accuracy = grid_search.best_score_ #mean of 10 models through 10-fold-cross validation
#Visualise best parameters
best_parameters = grid_search.best_params_ #passes argument of linearity/non_linearity to choose model



# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
