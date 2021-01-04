#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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



def neural_network(x_train, x_test, x_validation, y_train_lb, y_test_lb, y_validation_lb, epochs, batch_size):

    #-----------------------------------NEURAL NETWORK ARCHITECTURE.-------------------------------------------

    dim = np.shape(x_train)[1]
    num_labels = y_train_lb.shape[1]
    functions = ['relu', 'sigmoid', 'softmax', 'softplus', 'tanh', 'exponential']

    # 4-layers architecture.

    model = Sequential()

    model.add(Dense(256, input_shape=(dim,)))
    model.add(Activation(functions[1]))
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(Activation(functions[1]))
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(Activation(functions[1]))
    #model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(Activation(functions[1]))
    #model.add(Dropout(0.2))

    #OUTPUT LAYER:
    model.add(Dense(num_labels, activation='sigmoid')) # softmax for categorical and sigmoid for binary

    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam')  # (Posibles: loss=binary_crossentropy, categorical_crossentropy)
    history=model.fit(x_train, y_train_lb, batch_size=batch_size, epochs=epochs, validation_data=(x_validation,y_validation_lb))
    model.summary()

    #--------------------------------Método rápido para evaluar el modelo.--------------------------------------

    print('Evaluación sobre el test dataset.')
    results = model.evaluate(x_test, y_test_lb, batch_size=10)
    print('Test loss & Accuracy: ', results)

    return history


def results(model, x_test,y_test, y_test_lb):

    result = model.predict(x_test)
    cnt = 0
    prediction=[]

    for i in range(len(y_test_lb)):

        if(np.amax(result[i])<0.5): #np.amax: Return the maximum of an array or maximum along an axis.

            pred = np.argmax(result[i]) #np.argmax:Returns the indices of the maximum values along an axis.
            prediction = np.append(prediction, pred)
        else:
                pred = np.argmax(result[i])
                prediction = np.append(prediction, pred)

                if np.argmax(y_test_lb[i])==pred:
                    cnt+=1

    acc = str(round(cnt*100/float(len(y_test)),2))
    print("Precisión del modelo: " + acc + "%")

def confusion_matrix(y_test, prediction):


    cm = confusion_matrix(y_target=y_test,
                      y_predicted=prediction,
                      binary=False)

    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True,figsize=(7, 7),
                                class_names=list(np.unique(y_test)))

    plt.title('Matriz de confusión')
    plt.show()

def plots(history):

    accu = history.history['binary_accuracy']
    val_accu = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accu))

    #Training and validation accuracy.
    plt.plot(epochs, accu, 'darkred', label='Training accuracy')
    plt.plot(epochs, val_accu, 'cornflowerblue', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    #Training and validation loss.
    plt.plot(epochs, loss, 'darkred', label='Training Loss')
    plt.plot(epochs, val_loss, 'cornflowerblue', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.title('Training and validation loss')

    plt.show()
