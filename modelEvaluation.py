
"""
This file split the saved data (features and labels) into three datasets: training, testing and validation. 
The size of the subsets can be changed in the code.
"""

import numpy as np
import os

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold


from mlxtend.plotting import plot_confusion_matrix
from neural_network import MLP_neural_network
from mlxtend.evaluate import confusion_matrix
import matplotlib.pyplot as plt
from keras.utils import np_utils



def train_test(features_path, lables_path, test_size, val_size):

    #Loading features and labels
    features = np.load(features_path)
    labels = np.load(labels_path)
    #print('Dimensiones del array features: ', features.shape)
    #print('Dimensiones del array labels: ', labels.shape)


    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, 
                                                                                            random_state=1, stratify=labels, shuffle='True')
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = val_size, 
                                                                                            random_state=1, stratify=y_train, shuffle='True')

    print("The dimensions of X train dataset are: ", np.shape(x_train))
    print("The dimensions of X validation dataset are: ", np.shape(x_validation))
    print("The dimensions of X test dataset are: ", np.shape(x_test))
    n_samples = len(y_train)
    print("Number of training samples: " + str(n_samples))
    dim = np.shape(x_train)[1]

    #Randomm shuffle of data:
    order = np.array(range(n_samples))
    np.random.shuffle(order)
    x_train = x_train[order]
    y_train = y_train[order]

    return x_train, x_test, x_validation, y_train, y_test, y_validation, dim


def results(model, x_test, y_test, y_test_lb):

    result = model.predict(x_test)
    cnt = 0
    prediction = []
    for i in range(len(y_test_lb)):

        if(np.amax(result[i])<0.5):
            pred = np.argmax(result[i])
            prediction = np.append(prediction, pred)
        else:
                pred = np.argmax(result[i])
                prediction = np.append(prediction, pred)
                if np.argmax(y_test_lb[i])==pred:
                    cnt+=1

    acc = str(round(cnt*100/float(len(y_test)),2))
    print("Model accuracy: " + acc + "%")
    
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



#Loading features and Train/Test Split:

features_path = '/Users/miguel.r/Desktop/NN vehiculos/metodo_MFCCS+CHROMA/MFCCS+CHROMA features/features_27-11-2020, Hora: 14, Min: 56.npy'
labels_path = '/Users/miguel.r/Desktop/NN vehiculos/metodo_MFCCS+CHROMA/MFCCS+CHROMA features/labels_27-11-2020, Hora: 14, Min: 56.npy'

x_train, x_test, x_validation, y_train, y_test, y_validation, dim = train_test(features_path, labels_path, 0.1, 0.1)

lb = LabelEncoder()
y_train_lb = np_utils.to_categorical(lb.fit_transform(y_train))
y_test_lb = np_utils.to_categorical(lb.fit_transform(y_test))
y_validation_lb = np_utils.to_categorical(lb.fit_transform(y_validation))
print('Number of label encoder classes:', len(np.unique(y_train)))

#-----------Training the model:-------------------------------------

history, model = MLP_neural_network(x_train, x_test, x_validation, y_train_lb, y_test_lb, y_validation_lb, 2, 100)
print(history)



#-----------Quick method to evaluate the model from test dataset:-------------------------------------

prediction, acc = results(model, x_test,y_test, y_test_lb)

#-----------Reverse Label Encoder.-------------------------------------

prediction = lb.inverse_transform(prediction.astype(int))
print('Prediction classes: ', np.unique(prediction))


#----------Confusion matrix and model history plots:-------------------------------------

conf_matrix(y_test, prediction)
model_plots(history)

    
#--------------Saving the model:-------------------------------------
saving_path = '/Users/miguel.r/Desktop/'
model_json = model.to_json()
with open(saving_path+"model_acc_"+acc+".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(saving_path+"model_acc_"+acc+".h5")
