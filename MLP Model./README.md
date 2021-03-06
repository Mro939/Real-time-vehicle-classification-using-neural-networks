# Vehicle class classification

Real time vehicle classification from mic using a multilayer perceptron neural network (MLP) with keras.

## Continous classification from mic:

Steps to follow:

1. Use [feature_extractor.py](https://github.com/Mro939/Real-time-vehicle-classification-using-neural-networks/blob/main/features_extractor.py) for extract the audio features (mel frequency cepstral coefficients and chroma spectrum) and saved them in the desired path. This file will create two numpy array.
2. Take a look at the [neural_network.py](https://github.com/Mro939/Real-time-vehicle-classification-using-neural-networks/blob/main/neural_network.py) to study the MLP arquitecture and make some change on it if you want to try with others configurations. 
3. Use the [evaluateModel.py](https://github.com/Mro939/Real-time-vehicle-classification-using-neural-networks/blob/main/modelEvaluation.py) to train the machine learning model and test. Here you can change the size of the training dataset on the train-test split function and see some plots of the validation accuracy and loss function along the different epochs of training as well as confusion matrix.
4. Use [vehicle_classifier-from_mic.py](https://github.com/Mro939/Real-time-vehicle-classification-using-neural-networks/blob/main/vehicle_classifier-from_mic.py) for classify a sequence of vehicle sounds captured from mic using python library PyAudio.
