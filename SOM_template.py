#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 18:25:17 2018

@author: NVDL
"""
##### Part 1 - Unsupervised Learning
#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""
Dimensionality reduction on xi with Neural Network. Outliers in a self-organizing map
with the inter neuron Euclidian distance error ---> Fraud detection. Check 2 is to predict potential fraudelent credit card applications.
"""
#Feature scaling - Z-transformation
from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X) 

#Training the Self-Organising Map
#Have minisom.py in directory
from minisom import MiniSom
som = MiniSom(x= 10,
              y= 10, 
              input_len= 15, 
              sigma= 1.0, 
              learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration= 100)

#Plot map to spot outliers
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T) #Mean interneuron distances to difference range values
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

#Looping for every customer to get winning node
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, 
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth =2)
show()

#Finding the fraude
mappings = som.win_map(X) 
frauds = np.concatenate((mappings[(5,9)],mappings[1,7])) 
frauds = sc.inverse_transform(frauds)

"""
Every row is a potential fraudelent customer. Use outcome variable as DV for supervised
learning part. 
"""
#### Supervised Learning
#Creating matrix of features
customers = dataset.iloc[:,1:].values

#Creating dependent variable
#Create vector with 690 zeros
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:#i'th line of dataset or i'th customer, get custemor-ID
            is_fraud[i] = 1 #replace 0 with 1 if they are fraudelent
            

#Feature Scaling of IV variable
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


### Part 2 - ANN
#Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout #apply to all layers if overfitting

#Initialising the ANN - Defining the sequence of layers/a graph
classifier = Sequential()

#Adding input layer and first hidden layer of ANN
classifier.add(Dense(input_dim=15,output_dim=2, init='uniform', activation='relu')) #No. Features, Rectifier activation function
classifier.add(Dropout(p=0.1)) #0.1,0.2 ...% of neurons dropping out, max =0.5

#Adding second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))

#Output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid')) #Binary outcome, sigmoid activation function

#Compiling the ANN w/ Stochastic Gradient Descent='adam',logarithmic loss, accuracy criterium of error update
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)
"""
Classifier is now trained.
"""
### Part 3 - Evaluating the model
#Predicting probability that customer = fraudelent 
y_pred = classifier.predict(customers) #P(Exit|All variables) in percentages
y_pred = np.concatenate((dataset.iloc[:,0:1].values,y_pred), axis=1) #2d array with customer id's and predicted values
#Sort probabilities lowest to highest for ranking
y_pred = y_pred[y_pred[:,1].argsort()]


