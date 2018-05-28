
#install theano keras tensorflow

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")

X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# two categorical data Gender and geography
labelencoder_Xcountry = LabelEncoder()
X[:, 1] = labelencoder_Xcountry.fit_transform(X[:, 1])

labelencoder_Xgender = LabelEncoder()
X[:, 2] = labelencoder_Xgender.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train ,X_test,Y_train,Y_test = train_test_split(X, Y ,test_size = 0.2,random_state= 0)

#features scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)  

#done with preprocessing

#building ANN
# import keras
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#applying k-fold crossvalidations for better and more accurate accuracies

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6,kernel_initializer = 'uniform',activation ='relu' ,input_dim=11 ))
    classifier.add(Dense(units=6,kernel_initializer = 'uniform',activation ='relu'))
    classifier.add(Dense(units=1,kernel_initializer = 'uniform',activation ='sigmoid'))
    classifier.compile(optimizer = "adam",loss= "binary_crossentropy",metrics = ['accuracy'] ) 
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
#accuracies = cross_val_score(estimator = classifier, X=X_train, y=Y_train, cv=10 ,n_jobs=1)
    
import tensorflow as tf
with tf.device('/device:GPU:0'):
    accuracies = cross_val_score(estimator=classifier, X = X_train, y = Y_train, cv = 10,n_jobs= 1 )
    # -*- coding: utf-8 -*-
mean = accuracies.mean()
variance = accuracies.std()

#dropout regularisations to reduce overfitting if needed

#tunning Ann
#use gridSearch CV for hyperparameter tuning
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

#applying k-fold crossvalidations for better and more accurate accuracies

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6,kernel_initializer = 'uniform',activation ='relu' ,input_dim=11 ))
    classifier.add(Dense(units=6,kernel_initializer = 'uniform',activation ='relu'))
    classifier.add(Dense(units=1,kernel_initializer = 'uniform',activation ='sigmoid'))
    classifier.compile(optimizer = optimizer,loss= "binary_crossentropy",metrics = ['accuracy'] ) 
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
#building a dictionary to carry all parameters for gridsearch

parameters = {'batch_size' : [25,32],
             'epochs' : [100,500],
             'optimizer' : ['adam','rmsprop']}

#code the gridSearchCV Object
import tensorflow as tf
grid_search = GridSearchCV(estimator=classifier, param_grid = parameters, scoring = 'accuracy', cv=10)
with tf.device('/device:GPU:0'):
    grid_search = grid_search.fit(X_train,Y_train)
    
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_


    




