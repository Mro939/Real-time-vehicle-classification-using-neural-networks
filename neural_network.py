
"""
This file contents MLP_neural_network neural network architecture where the number of layers, 
the activation functions and applied dropout to each layer,the number of epochs and bach size can be modified. 
The result will a summary of the model and the result after fitting it with the testing dataset saved
in the history variable.
"""

import numpy as np

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed
from keras.layers import Convolution2D, MaxPooling2D, MaxPooling1D, Conv1D
from keras.optimizers import Adam, SGD
from keras.utils import np_utils



def MLP_neural_network(x_train, x_test, x_validation, 
                                       y_train_lb, y_test_lb, y_validation_lb, 
                                       epochs, batch_size):

    # NEURAL NETWORK ARCHITECTURE.

    dim = np.shape(x_train)[1]
    num_labels = y_train_lb.shape[1]
    act_functions = ['relu', 'sigmoid', 'softmax', 'softplus', 'tanh', 'exponential']

    # 4-layers
    model = Sequential()
    # 1st (input layer)
    model.add(Dense(512, input_shape=(dim,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # 2nd
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # 3rd
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # 4th
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # output layer.
    model.add(Dense(num_labels, activation='sigmoid'))

    # Compiling and summary.
    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam')
    history = model.fit(x_train, y_train_lb, batch_size=batch_size, epochs=epochs, validation_data=(x_validation, y_validation_lb))
    print('Models summary: ')
    model.summary()

    return history, model

