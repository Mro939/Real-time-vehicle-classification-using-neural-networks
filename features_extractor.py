""" 
This file generates two numpy arrays that contain the features (mel frecuency cepstral coefficients and chroma spectrogram)
and labels for each loaded audio sample. The arrays will be saved in the desired path in a .npy format.
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
    mfccs = np.array(librosa.feature.mfcc(y=X, sr=sample_rate, hop_length=hop_length, n_mfcc=20).T)
    chroma = np.array(librosa.feature.chroma_stft(y=X, sr=sample_rate, hop_length=hop_length).T)
    ext_features = np.hstack([mfccs, chroma])
    
    return ext_features


def audio_files_analysis(path, saving_path, hop_length):
    
    # np.shape(mfccs)[1] = 20
    # np.shape(chroma)[1] = 12
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

    # Saving features and labels as numpy arrays.
    np.save(saving_path+'features.npy', extracted_features)
    np.save(saving_path+'labels.npy',labels)


    # Uncomment the following lines for adding a date format when saving features and labels.
    # now=datetime.now()
    # format=now.strftime('%d-%m-%Y, Hora: %H, Min: %M')
    # np.save(saving_path+format+'.npy', extracted_features)
    # np.save(saving_path+format+'.npy', extracted_labels)  


path = '/Users/miguel.r/Desktop/project/Audios/all'
saving_path = '/Users/miguel.r/Desktop/project/MFCCS+CHROMA features'
audio_files_analysis(path_test, saving_path_test, 512)
