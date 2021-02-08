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

def extract_features_refined(file_name, max_pad_len):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, hop_length=512, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccs


def audio_files_analysis(path, saving_path):

    features = []
    clases = ['light', 'heavy']
    file_counter = 0
    file_ext = '*.wav'

# Iterate through each sound file and extract the features. Lenght pad = 1000
    for clase in clases:
        for file in glob.glob(os.path.join(path, clase, file_ext)):
            file_counter += 1
            print(file_counter, clase + ': ', file)

            class_label = clase
            data = extract_features_refined(file, 1000)
            features.append([data, class_label])

    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
    print('The features extraction has finished.')
    # Saving features as a numpy array.
    featuresdf.to_pickle(saving_path+"featuresdf_refined.pkl")


    # Uncomment the followingcode lines for adding a date format at the saved file name. 
    # now=datetime.now()
    # format=now.strftime('%d-%m-%Y, Hora: %H, Min: %M')
    # np.save(saving_path+format+'.npy', extracted_features)
    # np.save(saving_path+format+'.npy', extracted_labels)

# Execute features extraction:
path = '/Users/miguel.r/Desktop/CITSEM/NN vehiculos/Audios grabados/Todos'
saving_path = '/Users/miguel.r/Desktop/'

audio_files_analysis(path, saving_path)

