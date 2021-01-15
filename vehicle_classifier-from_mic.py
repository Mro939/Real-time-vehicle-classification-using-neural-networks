
"""
Real Time processing and vehicle classfication from mic.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import IPython
import os
import pyaudio
from sklearn.metrics import mean_squared_error



# Loading the model:

Path = r'/Users/miguel.r/Desktop/NN vehiculos/'
modelSTFT_path = "metodo_III/Resultados/stft_18-11-2020, Hora: 17, Min: 55_acc_88.89"
model_path = "metodo_MFCCS+CHROMA/Resultados/mfccs+chroma_512-512-512-128_epochs300-35027-11-2020, Hora: 15, Min: 08_acc_93.86"

# Model reconstruction:
with open(Path + model_path + '.json', 'r') as f:
    model = model_from_json(f.read())

model.load_weights(Path + model_path + '.h5')

# Label Encoder:
lb = LabelEncoder()
lb.fit_transform(['light', 'heavy'])  

# Auxiliars functions
def plotAudio(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,10))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    ax.margins(2, -0.1)
    plt.show()

def plotAudio2(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    plt.show()

def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr-mn)/(mx-mn)

def vehicle_predictor(X,RATE,frame_length, hop_length):
    
    clip, index = librosa.effects.trim(X, top_db=20, frame_length=frame_length, hop_length=hop_length)
    #Features:
    mfccs = np.array(librosa.feature.mfcc(y=X, sr=RATE, hop_length=hop_length, n_mfcc=20).T) 
    chroma = np.array(librosa.feature.chroma_stft(y=X, sr=RATE, hop_length=hop_length).T)
    # Normalize:
    mfccs_norm = minMaxNormalize(mfccs)
    chroma_norm = minMaxNormalize(chroma)
    features = np.hstack([mfccs,chroma])
    # Prediction from model
    result = model.predict(features)
    predictions = [np.argmax(y) for y in result]
    print('Potential vehicle class ', lb.inverse_transform([predictions[0]])[0])
    prediction = lb.inverse_transform([predictions[0]])[0]
        
    plotAudio2(clip)
    
    return prediction


# Parameters

CHUNKSIZE = 22050
RATE = 22050 # 22050 Hz of sampling rate 

# Initializing portaudio:
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

# Difining a noise window:
data = stream.read(10000, exception_on_overflow=False)
noise_sample = np.frombuffer(data, dtype=np.float32)
print("Loud sample:")
plotAudio2(noise_sample)

# Loud threshold:
loud_threshold = np.sqrt(np.mean(noise_sample**2))
# loud_threshold = np.mean(np.abs(silence_sample)) * 10 
print("Loud threshold =", loud_threshold)

audio_buffer = []
counter_light = 0
counter_heavy = 0

while(True):
    data = stream.read(CHUNKSIZE, exception_on_overflow=False)
    current_window = np.frombuffer(data, dtype=np.float32)

    # Reducing noise:
    current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)

    if(audio_buffer==[]):
        audio_buffer = current_window
        
    else:
        if(np.mean(np.abs(current_window))<loud_threshold):
            print('Inside SILENCE reign...')

        else:
            print("Inside LOUD reign...")
            audio_buffer = np.concatenate((audio_buffer,current_window))
            prediction =vehicle_predictor(np.array(audio_buffer), RATE, 512, 256)
            audio_buffer = [] # Buffer reboot
            
            # Setting a specific class vehicle as final prediction when detecting 3 times in a row the same potential class.
            if(prediction == 'heavy'):
                counter_heavy +=1
                #print('counter pesado: ', counter_heavy)
                if(counter_heavy == 2):
                    print('Vehicle class: HEAVY!')
                    counter_heavy = 0
            else:
                counter_light +=1
                #print('counter ligero: ', counter_light)
                if(counter_light == 2):
                    print('Vehicle class: LIGHT!')
                    counter_light = 0


# Closing Stream:
stream.stop_stream()
stream.close()
p.terminate()
