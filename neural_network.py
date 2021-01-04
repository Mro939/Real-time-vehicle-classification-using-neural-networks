
"""
Created on Fri Dec 11 12:32:12 2020

@author: miguel.r
"""
import numpy as np
import os
from mlxtend.evaluate import confusion_matrix
import matplotlib.pyplot as plt
import mlxtend
from mlxtend.plotting import plot_confusion_matrix

# Keras
from tensorflow.python import keras
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import models, layers, regularizers
from keras.models import Sequential




def train_test_split(features_path, lables_path):

    features = np.load(features_path+'.npy')
    labels = np.load(lables_path+'.npy')
    print('Dimensiones del array features: ', features.shape)
    print('Dimensiones del array labels: ', labels.shape)


    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=0, stratify=labels, shuffle='True')
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.1, random_state=0, stratify=y_train, shuffle='True')

    print("The dimensions of X train are: ", np.shape(x_train))
    print("The dimensions of X validation are: ", np.shape(x_validation))
    print("The dimensions of X test are: ", np.shape(x_test))

    print("Labels classes", np.unique(y_train))
    n_samples = len(y_train)
    print("Number of training samples: " + str(n_samples))

    order = np.array(range(n_samples))
    np.random.shuffle(order)
    x_train = x_train[order]
    y_train = y_train[order]

    lb = LabelEncoder()
    y_train_lb = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test_lb = np_utils.to_categorical(lb.fit_transform(y_test))
    y_validation_lb = np_utils.to_categorical(lb.fit_transform(y_validation))

    print('Label encoder classes:', y_train_lb[1])

    return x_train, x_test, x_validation, y_train_lb, y_test_lb, y_validation_lb




def neural_network(x_train, x_test, x_validation, y_train_lb, y_test_lb, y_validation_lb, epochs, batch_size):

    #-----------------------------------NEURAL NETWORK ARCHITECTURE.-------------------------------------------

    dim = np.shape(x_train)[1]
    num_labels = y_train_lb.shape[1]
    act_functions = ['relu', 'sigmoid', 'softmax', 'softplus', 'tanh', 'exponential']

    # 4-layers

    model = Sequential()

    #1st (input layer)
    model.add(Dense(512, input_shape=(dim,)))
    model.add(Activation(act_functions[1]))
    model.add(Dropout(0.2))

    #2nd
    model.add(Dense(256))
    model.add(Activation(act_functions[1]))
    model.add(Dropout(0.2))

    #3rd
    model.add(Dense(256))
    model.add(Activation(act_functions[1]))
    model.add(Dropout(0.2))

    #4th
    model.add(Dense(128))
    model.add(Activation(act_functions[1]))
    #model.add(Dropout(0.2))

    #output layer.
    model.add(Dense(num_labels, activation='sigmoid')) 

    #Compiling and summary.
    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam')  
    history=model.fit(x_train, y_train_lb, batch_size=batch_size, epochs=epochs, validation_data=(x_validation,y_validation_lb))
    model.summary()
    
    return history

x_train, x_test, x_validation, y_train_lb, y_test_lb, y_validation_lb = train_test_split()

