from am_analysis import am_analysis as ama
import numpy as np
import matplotlib.pyplot as plt 

import os
import pandas as pd
from scipy import signal

import librosa
import numpy as np
import scipy.io as sio

from scipy.signal import periodogram, welch
from scipy.signal import find_peaks
from scipy.stats import kurtosis
from scipy.stats import skew
import librosa
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features
import parselmouth

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
path1 = '/media/amrgaballah/Backup_Plus/Internship_exp/Exp_'
folder_name = ['3/door_knock', '3/no_door_knock', '4/Fire_alarm', '4/no_fire_alarm', '5/car_horn', '5/Siren']

for j in folder_name:
    print(path1+j + '/')
    path = path1+j + '/'
    output_cols = ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10',
                   'feat_11', 'feat_12', 'feat_13', 'feat_14', 'mfcc_1','mfcc_2', 'mfcc_3','mfcc_4', 'mfcc_5', 'mfcc_6',
                   'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'melbank_1', 'mel_2', 'mel_3',
                   'mel_4', 'mel_5', 'mel_6', 'mel_7', 'mel_8', 'mel_9', 'mel_10', 'mel_11', 'mel_12',  'mel_2', 'mel_3',
                   'mel_4', 'mel_5', 'mel_6', 'mel_7', 'mel_8', 'mel_9', 'mel_10', 'mel_11', 'mel_12',  'mel_2', 'mel_3',
                   'mel_4', 'mel_5', 'mel_6', 'mel_7', 'mel_8', 'mel_9', 'mel_10', 'mel_11', 'mel_12',  'mel_2', 'mel_3',
                   'mel_4', 'mel_5', 'mel_6', 'mel_7', 'mel_8', 'mel_9', 'mel_10', 'mel_11', 'mel_12','tonnet_1','tonnet_2', 
                   'tonnet_3','tonnet_4', 'tonnet_5', 'tonnet_6','zcr','tempo', 'feat_18', 'feat_20',
                   'feat_23', 'feat_25','label']
    feat_audio = np.empty((0, 87))
    for f in os.listdir(path):
        print(f)
        br, sr = librosa.load(os.path.join(path, f), sr = None)
       
        ## pitch
        snd = parselmouth.Sound(os.path.join(path, f))
        pitch = snd.to_pitch()

        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values==0] = np.nan
       
        x = pitch_values[~np.isnan(pitch_values)]
       
        if x.shape[0] == 0:
            feat_1 = feat_2 = feat_3=feat_4 =feat_5 =0
        else: 
            ## min 
            feat_1 = np.min(x)

            feat_1= np.asarray(feat_1)
            ## max 
            feat_2 = np.max(x)

            ## mean 
            feat_3 = np.mean(x)

            ##range
            feat_4 = feat_2-feat_1

            #stddev 
            feat_5 = np.std(x)


           

        ###RMS energy
        rms_ = librosa.feature.rms(y=br)
        print(rms_.shape)


        ## min 
        feat_6 = np.min(rms_, axis=1)

        ## max 
        feat_7 = np.max(rms_, axis=1)

        ## mean 
        feat_8 = np.mean(rms_, axis=1)

        ## range 
        feat_9 = feat_7-feat_6

        #stddev 
        feat_10 = np.std(rms_, axis=1)
       


        ####Formants
        sound = Waveform(path=os.path.join(path, f))
        formants = sound.formants()
       
        feat_11 = formants.get('f1')

        feat_12 = formants.get('f2')

        feat_13 = formants.get('f3')

        feat_14 = formants.get('f4')



        #### MFCC
        mfcc_ = librosa.feature.mfcc(y=br, n_mfcc=13,sr=sr)
        mfcc_ = np.mean(mfcc_, axis = 1)
        mfcc_ = mfcc_[:, None]
        
        ####Melbank
        S = librosa.feature.melspectrogram(y=br, sr=sr, n_mels=12,fmax=8000)
        S = np.mean(S,axis = 1)
        S = S[:, None]
      
        ###Chroma features
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)
        chroma = np.mean(chroma, axis = 1)
       

        chroma_cq = librosa.feature.chroma_cqt(y=br, sr=sr)
        chroma_cq = np.mean(chroma_cq, axis = 1)
        

        chroma_cens = librosa.feature.chroma_cens(y=br, sr=sr)
        chroma_cens = np.mean(chroma_cens, axis = 1)
        

        ### tonet features
        y = librosa.effects.harmonic(y=br)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz = np.mean(tonnetz, axis = 1)
      
        #### zero crossing feature


        zcrs = librosa.feature.zero_crossing_rate(br)
        zcr = np.sum(zcrs, axis = 1)
      
        #### musical features

        ###Rhythm features

        hop_length = 512
        oenv = librosa.onset.onset_strength(y=br, sr=sr, hop_length=hop_length)

        # Estimate the global tempo for display purposes
        tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                                   hop_length=hop_length)[0]
        



        ####Beat and tempo
        onset_env = librosa.onset.onset_strength(y, sr=sr,
                                                 aggregate=np.median)
        av = librosa.util.normalize(onset_env)



        ## mean 
        feat_18 = np.mean(av, axis=0)


        #stddev 
        feat_20 = np.std(av, axis=0)


        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)


        ## mean 
        feat_23 = np.mean(pulse, axis=0)


        #stddev 
        feat_25 = np.std(pulse, axis=0)



       

        final_f = np.hstack((np.array([feat_1])[:, None], np.array([feat_2])[:, None], np.array([feat_3])[:, None], np.array([feat_4])[:, None], np.array([feat_5])[:, None], feat_6[:, None], feat_7[:, None], feat_8[:, None], feat_9[:, None], feat_10[:, None],np.array([feat_11])[:, None], np.array([feat_12])[:, None], np.array([feat_13])[:, None], np.array([feat_14])[:, None],
                             mfcc_.T, S.T, chroma[:, None].T, chroma_cq[:, None].T,chroma_cens[:, None].T,tonnetz[:, None].T,zcr[:, None], np.array([tempo])[:, None].T,np.array([feat_18])[:, None],np.array([feat_20])[:, None], np.array([feat_23])[:, None], np.array([feat_25])[:, None]))



        feat_audio = np.vstack((feat_audio, final_f))
    
    n= feat_audio.shape[0]
  
    X0 = np.zeros((n,1))
    # X0 = np.ones((n,1))
  
    feat = np.hstack((feat_audio,X0))
    

    df=pd.DataFrame(feat)
    df.to_csv(path+j +'_feat2.csv',index=None)
