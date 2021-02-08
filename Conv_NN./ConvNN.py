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
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from datetime import datetime 


def Convolutional_NN(x_train, x_test, y_train, y_test,
                                    num_epochs, num_batch_size, num_labels,
                                    num_rows, num_columns, num_channels):
    

    filter_size = 2

    # Construct model 
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(num_labels, activation='sigmoid'))
    
    # Compiling and summary.
    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam')
    print("Model's summary: ")
    model.summary()
    # Calculate pre-training accuracy 
    score = model.evaluate(x_test, y_test, verbose=1)
    accuracy = 100*score[1]
    print("Pre-training accuracy: %.4f%%" % accuracy)
    
    start = datetime.now()
    history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)    
    
    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1])

    return history, model


