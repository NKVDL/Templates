#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 18:45:05 2018

@author: NVDL
"""
"""
Run Part 4 - Gridsearch first to find optimum model and parameters to train 
Artificial Neural Network on. Proceed from Part 1 - Preprocesing. Check if IV have to be encoded
to dummy variables, check test_size and avoid dummy trap. Part 2 - Building ANN, 
add dropouts and optimise while compiling, and fit that function to the data 
to obtain classifier. Use classifier to predict y_test from x_test in Part 3 to evaluate the model's
performance. Compare the two. Lastly, add new entries (business) to predict. 

- Anaconda, Tensorflow, Keras, Python 3.6, Spyder 3.2.8 
"""

###Part 1 Data Preprocessing
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

##Encoding Independent Variables  (dummy var)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#Encode nxn categorical features
onehotencoder = OneHotEncoder(categorical_features = [1]) #encode categorical features on number 0
X = onehotencoder.fit_transform(X).toarray() #transformed the data again

#To avoid dummy trap
X = X[:, 1:]

##Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling of IV_test, IV_train
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### Part 2 - ANN
#Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout #apply to all layers if overfitting

#Initialising the ANN - Defining the sequence of layers/a graph
classifier = Sequential()

#Adding input layer and first hidden layer of ANN
classifier.add(Dense(input_dim=11,output_dim=6, init='uniform', activation='relu')) #Rectifier activation function
classifier.add(Dropout(p=0.1)) #0.1,0.2 ...% of neurons dropping out, max =0.5

#Adding second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))

#Output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid')) #Binary outcome, sigmoid activation function

#Compiling the ANN w/ Stochastic Gradient Descent='adam',logarithmic loss, accuracy criterium of error update
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)
"""
Classifier is now trained.
"""
### Part 3 - Evaluating the model
#Predicting the dependent variables 
y_pred = classifier.predict(X_test) #P(Exit|All variables) in percentages
y_pred = (y_pred > 0.5) #Returns Boolean if sigmoid < or > 0.5

# Validating model w/ the Confusion Matrix Categorical
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""
Accuracy: (1541+141)/(1541+141+264+54)=84% 
Precision: (1541)/(1541+264) =85%
"""
##Predicting a single new observation
"""
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
"""
#A standardized observation (transform) line as a 2D array 
#check dataset for encoded categorical variables
new_prediction = classifier.predict(sc.transform(np.array([[0.0,
                                                            0,
                                                            600, 
                                                            1, 
                                                            40, 
                                                            3, 
                                                            60000,
                                                            2, 
                                                            1, 
                                                            1, 
                                                            50000]])))
#Return Boolean
new_prediction = (new_prediction > 0.5) #Returns Boolean if sigmoid < or > 0.5

### Part 3 Evaluating, Imporving and Tuning the ANN
# Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score

##Building variable from classifier
def build_classifier():
    #Initialising the ANN - Defining the sequence of layers/a graph
    classifier = Sequential()
    # Adding input layer and first hidden layer of ANN
    classifier.add(Dense(input_dim=11,output_dim=6, init='uniform', activation='relu')) #Rectifier activation function
    #Adding second hidden layer
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    #Output layer
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid')) #Binary outcome, sigmoid activation function
    #Compiling the ANN w/ Stochastic Gradient Descent,logarithmic loss, accuracy criterium of error update
    classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

##Building ANN-classifier with K-fold cross validation
classifier = KerasClassifier(build_fn= build_classifier, batch_size = 10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv =10, n_jobs=-1) #returns 10 accuracies of K-fold cross validation
#Asses means of 10 models
mean = accuracies.mean()
variance = accuracies.std()

### Part 4 - Improving the ANN -Finding best hyperparameters(GridSearch)
#-Dropout regularization to reduce overfitting if needed-Part 2
##Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

##Building variable from ANN for Grid-Search - 
def build_classifier(optimizer, init,activation, loss): #create new argument for hyperparameter
    #Initialising the ANN - Defining the sequence of layers/a graph
    classifier = Sequential()
    # Adding input layer and first hidden layer of ANN
    classifier.add(Dense(input_dim=11,output_dim=6, init=init, activation=activation)) #Rectifier activation function
    #Adding second hidden layer
    classifier.add(Dense(output_dim=6, init=init, activation=activation))
    #Output layer
    classifier.add(Dense(output_dim=1, init=init, activation=activation)) #Binary outcome, sigmoid activation function
    #Compiling the ANN w/ parameters input var 'optimizer',logarithmic loss, accuracy criterium of error update
    classifier.compile(optimizer=optimizer,loss=loss, metrics=['accuracy'])
    return classifier

##Building ANN-classifier 
classifier = KerasClassifier(build_fn= build_classifier)
#Create dictionary of all hyperparameters to search for optimisation of the model
#Define grid_search model parameters - Pick on face validity
parameters = {'batch_size':[25,32],
              'epochs': [100,300],
              'optimizer': ['adam',
                            'rmsprop',
                            'Adagrad',
                            'Adadelta',
                            'Adamax',
                            'Nadam', 
                            'TFOptimizer',],
              'init' : ['uniform',
                        'lecun_uniform', 
                        'normal', 
                        'zero',
                        'glorot_normal', 
                        'glorot_uniform', 
                        'he_normal', 
                        'he_uniform'],
              'activation':['softmax', 
                            'softplus',
                            'softsign',
                            'relu', 
                            'tanh', 
                            'sigmoid', 
                            'hard_sigmoid',
                            'linear'],
              'loss':['mean_squared_error', 
                      'mean_absolute_error',
                      'mean_absolute_percentage_error', 
                      'mean_squared_logarithmic_error', 
                      'squared_hinge','hinge', 
                      'categorical_hinge',
                      'logcosh',
                      'categorical_crossentropy', 
                      'sparse_categorical_crossentropy', 
                      'binary_crossentropy'],
                      'kullback_leibler_divergence',
                      'poisson',
                      'cosine_proximity'}
#Create grid_search function for optimal parameters
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
#Fit the grid_search function to the training set
grid_search = grid_search.fit(X_train, y_train)
#Return the selection of best parameters & accuracies
best_parameters= grid_search.best_params_
best_accuracy = grid_search.best_score_
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))

"""
Adjust if needed: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
"""












