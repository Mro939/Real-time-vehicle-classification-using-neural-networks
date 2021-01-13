# Vehicle class classification

Real time vehicle classification from mic using a multilayer perceptron neural network (MLP) with keras.

Steps to follow:

1. Use feature_extractor.py for extract the audio features (mel frequency cepstral coefficients and chroma spectrum) and saved them in the desired path. This file will create two numpy array.
2. Take a look at the neural_network.py to study the MLP arquitecture and make some change on it if you want to try with others configurations. 
3. Use the evaluateModel.py to train the machine learning model and test. Here you can change the size of the training dataset on the train-test split function and see some plots of the validation accuracy and loss function along the different epochs of training as well as confusion matrix.
