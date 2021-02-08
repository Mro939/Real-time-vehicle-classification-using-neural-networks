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
import pandas as pd
from datetime import datetime

def extract_features(file_name):

    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, hop_length=512, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, hop_length=512)
        chromascaled = np.mean(chroma.T,axis=0)
        F = [*mfccsscaled, *chromascaled]

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return F


def audio_files_analysis(path, saving_path):

    features = []
    clases = ['ligero', 'pesado']
    file_counter = 0
    file_ext = '*.wav'

# Iterate through each sound file and extract the features
    for clase in clases:
        for file in glob.glob(os.path.join(path, clase, file_ext)):
            file_counter += 1
            print(file_counter, clase + ': ', file)

            class_label = clase
            data = extract_features(file)
            features.append([data, class_label])

    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

    # Saving features as a numpy array.
    featuresdf.to_pickle(saving_path+"featuresdf.pkl")
    #np.save(saving_path+'features.npy', featuresdf)


    # Uncomment the followingcode lines for adding date format when saving features and labels
    # now=datetime.now()
    # format=now.strftime('%d-%m-%Y, Hora: %H, Min: %M')
    # np.save(saving_path+format+'.npy', extracted_features)
    # np.save(saving_path+format+'.npy', extracted_labels)

# Features extraction, results saved in saving_path.
path = '/Users/miguel.r/Desktop/CITSEM/NN vehiculos/Audios grabados/Todos'
saving_path = '/Users/miguel.r/Desktop/NN vehiculos/metodo_MFCCS+CHROMA/MFCCS+CHROMA features'

path_test = '/Users/miguel.r/Desktop/NN vehiculos/Audios vehiculos grabados/testing'
saving_path_test = '/Users/miguel.r/Desktop/'

audio_files_analysis(path, saving_path_test)
