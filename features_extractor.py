#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:54:20 2020

@author: miguel.r
"""
import librosa
import numpy as np
import glob
import os
from datetime import datetime


def features_extraction(audio_file,hop_length):
    X,sample_rate=librosa.load(audio_file)
    mfccs=np.array(librosa.feature.mfcc(y=X,sr=sample_rate,hop_length=hop_length,n_mfcc=20).T)
    chroma=np.array(librosa.feature.chroma_stft(y=X, sr=sample_rate,hop_length=hop_length).T)
    return mfccs,chroma


def audio_files_analysis(path,saving_path,mfccs_coeff):

    #dim_MFCCS = 20
    #dim_CHROMA = 12
    #DIM = dim_CHROMA+dim_MFCCS
    DIM=32

    extracted_features = np.empty([0,DIM])
    extracted_labels = np.empty(0)
    clases = ['ligero','pesado']
    file_counter=0
    file_ext='*.wav'

    for clase in clases:
        features=np.empty([0,DIM])
        for file in glob.glob(os.path.join(path, clase, file_ext)):
            mfccs,chroma = features_extraction(file, 512)
            file_counter +=1
            print(file_counter, clase + ': ', file)
            ext_features = np.hstack([mfccs,chroma])
            features = np.vstack([features, ext_features])

            label=[]
            for i in range(features.shape[0]):
                label.append(clase)

        extracted_features = np.row_stack([extracted_features, features])
        extracted_labels = np.append(extracted_labels, label)
        print('Features shape: ', extracted_features.shape)
        print('Labels shape: ', extracted_labels.shape)

    #Date format.
    now=datetime.now()
    format=now.strftime('%d-%m-%Y, Hora: %H, Min: %M')

    np.save(saving_path+format+'.npy', extracted_features)
    np.save(saving_path+format+'.npy', extracted_labels)
    
    return features, labels
