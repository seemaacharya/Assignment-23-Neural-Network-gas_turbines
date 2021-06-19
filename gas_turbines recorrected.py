# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 19:40:53 2021

@author: DELL
"""
#Importing the libraries
import pandas as pd
import numpy as np
import tensorflow as tf
#loading the dataset
dataset = pd.read_csv("gas_turbines.csv")
dataset.head()

#Checking the missing values
dataset.isna().sum()
#There is no missing values in the dataset

#Splitting into X(independent variables) and Y (dependent variable)
x=dataset.iloc[:,:-2]
y=dataset.iloc[:,-2:]
print(x.shape, y.shape)

#train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#usage of StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)



#using Neural Network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Initializing ANN
model= Sequential()
model.add(Dense(units=10,activation="relu",kernel_initializer="he_uniform",input_dim=9))
model.add(Dense(units=2))
#compile the model
model.compile(optimizer="adam",loss="mae",metrics=["accuracy"])
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(10, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mae', optimizer='adam')
	return model


#Fit the model


history=model.fit(x_train,y_train,validation_split=0.33,epochs=150,batch_size=10)

#predict the model
y_pred=model.predict(x_test)
y_pred

#Evaluate the model
scores=model.evaluate(x_test,y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#Accuracy=100%


#Visualize training history
#list all the data in history
model.history.history.keys()

#summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

#Summarize the history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
































