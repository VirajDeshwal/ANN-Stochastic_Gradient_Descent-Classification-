#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 01:35:00 2018

@author: virajdeshwal
"""


import pandas as pd

file = pd.read_csv('Churn_Modelling.csv')
X = file.iloc[:,3:13].values
y = file.iloc[:,13].values
'''We have to encode the categorical data. As our Independent variable contains the string.'''

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


from sklearn.preprocessing import StandardScaler
#Scaling the X_set to be in the same range.
scaling = StandardScaler()

x_train = scaling.fit_transform(x_train)
x_test = scaling.fit_transform(x_test)


'''Now lets work with Keras'''
#Import our model

from keras.models import Sequential
from keras.layers import Dense

#Initializing the model
model = Sequential()
#First Layer
model.add(Dense(output_dim =6 , init ='uniform', activation='relu', input_dim  =11))
#Second Hidden Layer
model.add(Dense(output_dim =6 , init ='uniform', activation='relu'))
#Output layer with sigmoid function to give output as a probability
model.add(Dense(output_dim =1 , init ='uniform', activation='sigmoid'))

#compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])
#train the model
model.fit(x_train, y_train, batch_size=10, nb_epoch=10)


y_pred = model.predict(x_test)
#TRUE if prob>0.5|| false if prob <0.5
y_pred = (y_pred>0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import confusion_matrix

#show the true positive and false positive through the confusion matrix.
conf_matrix = confusion_matrix(y_test, y_pred)
print('\n\n print the confusion matrix for true and false prediction rate.\n\n')
print(conf_matrix)



from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print('\n\n\n Hence the accuracy of the GaussianNB is:',accuracy)
print('\n\n Done :)')

