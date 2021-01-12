""" 
This file generates two numpy arrays that will contain the features 
and labels of each loaded audio file. The features selected in this 
case are the mel frecuency cepstral coefficients and the chroma spectrogram.
"""
# Audio analysis
import librosa
# Others
import numpy as np
import glob
import os
from datetime import datetime

def features_extraction(audio_file, hop_length):
    
    # Loading audio file:
    X, sample_rate = librosa.load(audio_file)
    
    # Compute mel frecuency cepstral coefficients and chroma spectrogram.
    mfccs = np.array(librosa.feature.mfcc(y=X, sr=sample_rate, 
                                          hop_length=hop_length, n_mfcc=20).T)
    chroma = np.array(librosa.feature.chroma_stft(y=X, sr=sample_rate,
                                                  hop_length=hop_length).T)
    
    ext_features = np.hstack([mfccs, chroma])
    
    return ext_features


def audio_files_analysis(path, saving_path, hop_length):
    # MFCCS dimension = 20
    
    # CHROMA dimension = 12
    DIM = 32

    extracted_features = np.empty([0, DIM])
    labels = []
    clases = ['light', 'heavy']
    file_counter = 0
    file_ext = '*.wav'

    for clase in clases:
        features=np.empty([0, DIM])
        for file in glob.glob(os.path.join(path, clase, file_ext)):
            ext_features = features_extraction(file, hop_length)
            file_counter += 1
            print(file_counter, clase + ': ', file)
            features = np.vstack([features, ext_features])  
            
        for i in range(np.shape(features)[0]):
            labels.append(clase)

        extracted_features = np.row_stack([extracted_features, features])
        print('Features array shape: ', extracted_features.shape)
        print('Labels array shape: ', np.shape(labels))

    # Saving features and labels in numpy array format.
    np.save(saving_path+'features.npy', extracted_features)
    np.save(saving_path+'labels.npy',labels)


    # Uncomment the followingcode lines for adding date format when saving features and labels
    # now=datetime.now()
    # format=now.strftime('%d-%m-%Y, Hora: %H, Min: %M')
    # np.save(saving_path+format+'.npy', extracted_features)
    # np.save(saving_path+format+'.npy', extracted_labels)  

# -------------------------------Execute features extraction. Saved in saving_path:----------------------------------
path = '/Users/miguel.r/Desktop/NN vehiculos/Audios vehiculos grabados/Todos'
saving_path = '/Users/miguel.r/Desktop/NN vehiculos/metodo_MFCCS+CHROMA/MFCCS+CHROMA features'

path_test = '/Users/miguel.r/Desktop/NN vehiculos/Audios vehiculos grabados/testing'
saving_path_test = '/Users/miguel.r/Desktop/'

audio_files_analysis(path_test, saving_path_test, 512)
