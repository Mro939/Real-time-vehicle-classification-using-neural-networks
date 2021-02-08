import numpy as np
import os
import pandas as pd

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical

from mlxtend.plotting import plot_confusion_matrix
from neural_network_CORREGIDO import MLP_neural_network
from mlxtend.evaluate import confusion_matrix
import matplotlib.pyplot as plt
from keras.utils import np_utils

lb = LabelEncoder()


def train_test(dataframe_path):

    #Loading features and labels dataframe
    featuresdf = pd.read_pickle(dataframe_path)
    
    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())
    yy = to_categorical(lb.fit_transform(y))
    
    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42, stratify=yy)

    print("The dimensions of X train dataset are: ", np.shape(x_train))
    print("The dimensions of X test dataset are: ", np.shape(x_test))
    n_samples = len(y_train)
    print("Number of training samples: " + str(n_samples))
    dim = np.shape(x_train)[1]
    num_labels = yy.shape[1]

    return x_train, x_test, y_train, y_test, dim, num_labels


def results(model, x_test, y_test):
    
    result = model.predict(x_test)
    cnt = 0
    prediction = []
    for i in range(len(y_test)):

        if(np.amax(result[i])<0.5):
            pred = np.argmax(result[i])
            prediction = np.append(prediction, pred)
        else:
                pred = np.argmax(result[i])
                prediction = np.append(prediction, pred)
                if np.argmax(y_test[i])==pred:
                    cnt+=1

    acc = str(round(cnt*100/float(len(y_test)),2))
    print("Model's accuracy: " + acc + "%")

    
    return prediction, acc

def conf_matrix(y_test, prediction):


    cm = confusion_matrix(y_target=y_test,
                      y_predicted=prediction,
                      binary=False)

    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True,figsize=(7, 7),
                                class_names=list(np.unique(y_test)))

    plt.title('Confusion Matrix')
    plt.figure()
    plt.show()
    
    #Uncoment the above line for save the image:
    #plt.savefig('/Users/miguel.r/Desktop/confusion_matrix.pdf')

def model_plots(history):

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
    plt.legend()
    plt.show()

    
# Loading features and Train/Test Split:

dataframe_path = '/Users/miguel.r/Desktop/featuresdf.pkl'

x_train, x_test, y_train, y_test, dim, num_labels = train_test(dataframe_path)
print('Label encoder classes:', y_train[1])


# Training the model:

history, model = MLP_neural_network(x_train, x_test, y_train, y_test, 50, 3, num_labels)


# Quick method to evaluate the model from test dataset:

prediction, acc = results(model, x_test,y_test)

# Reverse Label Encoder:

prediction = lb.inverse_transform(prediction.astype(int))

y_test_lb = []
for i in range (len(y_test)):
    if (np.argmax(y_test, axis=1)[i]) == 1:
        y_test_lb.append('pesado')
    if (np.argmax(y_test, axis=1)[i]) == 0:
        y_test_lb.append('ligero')
        
print('Prediction classes: ', np.unique(prediction))

# Confusion matrix and model history plots:

conf_matrix(y_test_lb, prediction)
model_plots(history)
  
# Saving the model:
saving_path = '/Users/miguel.r/Desktop/'
model_json = model.to_json()
with open(saving_path+"_acc_"+acc+".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(saving_path+"_acc_"+acc+".h5")
