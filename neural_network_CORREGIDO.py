"""
This file contents MLP_neural_network neural network architecture where the number of layers, 
the activation functions and applied dropout to each layer,the number of epochs and bach size can be modified. 
The result will a summary of the model and the result after fitting it with the testing dataset saved
in the history variable.
"""

import numpy as np
from datetime import datetime 

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed
from keras.layers import Convolution2D, MaxPooling2D, MaxPooling1D, Conv1D
from keras.optimizers import Adam, SGD
from keras.utils import np_utils



def MLP_neural_network(x_train, x_test, y_train, y_test, 
                                       num_epochs, num_batch_size, num_labels):

    # NEURAL NETWORK ARCHITECTURE.

    dim = np.shape(x_train)[1]
    act_functions = ['relu', 'sigmoid', 'softmax', 'softplus', 'tanh', 'exponential']

    # Construct model 
    model = Sequential()

    model.add(Dense(256, input_shape=(52,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(num_labels))
    model.add(Activation('sigmoid'))
    
    # Compiling and summary.
    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam')
    start = datetime.now()

    history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)


    duration = datetime.now() - start
    print("Training completed in time: ", duration)    
    print("Model's summary: ")
    model.summary()
    
    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1])

    return history, model


