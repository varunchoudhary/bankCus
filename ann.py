
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

# choose rectifier function in  hidden layer and sigmoid in output layers as activation function

classifier = Sequential()
#first hidden layer
classifier.add(Dense(units=6,kernel_initializer = 'uniform',activation ='relu' ,input_dim=11 ))
#second hidden layer
classifier.add(Dense(units=6,kernel_initializer = 'uniform',activation ='relu'))
#output layer
classifier.add(Dense(units=1,kernel_initializer = 'uniform',activation ='sigmoid'))

#applying stocastic gradient decent
# best optyimiser to use is adam optimiser and loss function if the out put is binary the " binary" logarithmic else categorical _crossEntropy

classifier.compile(optimizer = "adam",loss= "binary_crossentropy",metrics = ['accuracy'] ) 

classifier.fit(X_train,Y_train, batch_size = 10,nb_epoch=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
accuracy = (1539+145)/2000
print("accuracy of predictions is :  ",accuracy)

q=sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]]))
y_p = classifier.predict(q)
y_p = (y_p > 0.5)
print(y_p)
 
