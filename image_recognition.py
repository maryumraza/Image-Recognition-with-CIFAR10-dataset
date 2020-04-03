# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:44:57 2020

@author: uni tech
"""


import numpy as np
import pandas as pd
from tensorflow import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten,Activation,  Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils 
import matplotlib.pyplot as plt
import cv2


# Loading the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()




X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# Normalizing 
X_train /= 255
X_test /= 255



# Converting iamges to grayscale
X_train_gray = np.zeros(X_train.shape[:-1])
for i in range(X_train.shape[0]):
    X_train_gray[i] =  cv2.cvtColor( X_train[i], cv2.COLOR_BGR2GRAY) 



X_test_gray = np.zeros(X_test.shape[:-1])
for i in range(X_test.shape[0]):
    X_test_gray[i] =  cv2.cvtColor( X_test[i], cv2.COLOR_BGR2GRAY)
     

X_train_gray = X_train_gray.reshape(X_train.shape[0],32,32,1)
X_test_gray = X_test_gray.reshape(X_test.shape[0], 32,32,1)


    
# One hot encoding outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)




# Defining callbacks function    
early_stoppings = EarlyStopping(monitor='val_loss',
                                patience = 5,
                                verbose = 1,
                                restore_best_weights = True)   





# Defining the model and adding layers to it
model = Sequential()
model.add(Conv2D(32, input_shape=(32, 32, 1), kernel_size=(4,4),padding='same', activation='relu'))
model.add(BatchNormalization())


model.add(Conv2D(64,kernel_size=(4,4),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())


model.add(Conv2D(64,kernel_size=(4,4),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())


model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Dense(10, activation='softmax'))



print(model.summary())





# model compilation
adam = Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer = adam, loss='categorical_crossentropy',  metrics=['accuracy'])


# Model training
model.fit(X_train_gray, y_train ,batch_size=500, epochs=10, validation_split=0.2 , callbacks=[early_stoppings])


# Saving the model
from keras.models import load_model 
model.save('image_recognition_model.h5')






















