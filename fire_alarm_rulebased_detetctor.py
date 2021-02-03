###### In this code, I have implemented rule for selecting a area of intrest by observing abrupt changes in RMSE energy.
###### Algorithm:
###### Step 1: calculate the rmse over frame_length = 2048 and hop_length = 512
###### Step 2: Based on rmse value make  a rule for detetction of region of intrest for fire alarm fetaures. Algorithm for selection of starting and ending time of fire alarm when it crosses zero, raised a flag for strating featur extraction get a start time and then stop flag as soon as it goes to negative value then the end time of fire alarm beep. 
###### Step 3: Once we get a region of interest, calculate features for Fire alarm : pitch, harmonics, harmonic ratio, Spectral centroid, spectral bandwidth, spectral flatness, duration, entropy of energy, skewness and kurtosis of energy.

import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
import librosa
import pandas as pd
import numpy
from scipy.signal import find_peaks
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.stats import entropy
path = '/media/amrgaballah/Backup_Plus/Internship_exp/final_wavfiles_sr/Fire_alarm/'
feat_audio = np.empty((0,133))
for filename_ in os.listdir(path):
    #### filename and parameters
    filename = os.path.join(path,filename_)


    x, fs = librosa.load(filename_, sr= None)
    sr =fs
    frames_all = range(len(x))
    hop_length = 512
    frame_length = 2048
    #### Step 1: calculate the rmse over frame_length = 2048 and hop_length = 512
    #### peak_env and rmse
    peak_env = numpy.array([max(abs(x[i:i+frame_length]))for i in range(0, len(x), hop_length)])
    rmse = librosa.feature.rms(x, frame_length=frame_length, hop_length=hop_length, center=True)
    rmse = (rmse-np.mean(rmse))/np.std(rmse)
    print(rmse.shape)
    frames = range(len(peak_env))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
   
   

   ###### Step 2: Based on rmse value make  a rule for detetction of region of intrest for fire alarm fetaures. Algorithm for selection of starting and ending time of fire alarm when it crosses zero, raised a flag for strating featur extraction get a start time and then stop flag as soon as it goes to negative value then the end time of fire alarm beep. 
    ### list of array where rmse values are greater than 0
    rmse_pos = np.where(np.array([rmse.T])>=0)[1]
    print(rmse_pos.shape)
    t_new = t[rmse_pos]
    print(t_new.shape)
    start_time = [] 
    end_time = []
    for index in range(len(rmse_pos)-1):

        if index == 0:
            start_time.append(t_new[index])

        elif rmse_pos[index] != rmse_pos[index+1]-1:
            end_time.append(t_new[index])
            start_time.append(t_new[index+1])

    end_time.append(t_new[-1]) ### to append last end time
    print(len(start_time), len(end_time))


    ### total number of knock
    print('total number of beeps', len(start_time))

    ###### Step 3: Once we get a region of interest, calculate features for Fire alarm : pitch, harmonics, harmonic ratio, Spectral centroid, spectral bandwidth, spectral flatness, duration, entropy of energy, skewness and kurtosis of energy.
    ####  get samples from start time and end time from audio signal
    t_w = librosa.samples_to_time(frames_all, sr=sr)
    time_l = list(t_w)
    time_l1 = [round(num,5) for num in time_l]
    
    for i1, j1 in zip(start_time,end_time):


        out = np.where((round(i1,5)<(time_l1)) & ((time_l1)<round(j1,5)))
        f = x[out]
        print(f)
        

        S_ = np.abs(librosa.stft(f))
        cent = librosa.feature.spectral_centroid(y=f, sr=sr)
        print(cent.shape)
        feat_1 = np.mean(cent, axis=1)
        feat_2 = np.std(cent, axis=1)
        S, phase = librosa.magphase(librosa.stft(y=f))
        centroid = librosa.feature.spectral_centroid(S=S)
        print(centroid.shape)
        feat_3 = np.mean(centroid, axis=1)
        feat_4 = np.std(centroid, axis=1)
        bw = librosa.feature.spectral_bandwidth(S=S)
        print(bw.shape)
        feat_5 = np.mean(bw, axis=1)
        feat_6 = np.std(bw, axis=1)
        contrast = librosa.feature.spectral_contrast(S=S_, sr=sr)
        print(contrast.shape)
        feat_7 = np.mean(contrast, axis=1)
        feat_8 = np.std(contrast, axis=1)
        flatness = librosa.feature.spectral_flatness(y=f)
        print(flatness.shape)
        feat_9 = np.mean(flatness, axis=1)
        feat_10 = np.std(flatness, axis=1)
        flat =librosa.feature.spectral_flatness(S=S)
        print(flat.shape)
        feat_11 = np.mean(flat, axis=1)
        feat_12 = np.std(flat, axis=1)


        chroma = librosa.feature.chroma_stft(S=S_, sr=sr)
        print(chroma.shape)
        feat_13 = np.mean(chroma, axis=1)
        feat_14 = np.std(chroma, axis=1)
        chroma_cens = librosa.feature.chroma_cens(y=f, sr=sr)
        print(chroma_cens.shape)
        feat_15 = np.mean(chroma_cens, axis=1)
        feat_16 = np.std(chroma_cens, axis=1)
        chroma_cq = librosa.feature.chroma_cqt(y=f, sr=sr)
        print(chroma_cq.shape)
        feat_17 = np.mean(chroma_cq, axis=1)
        feat_18 = np.std(chroma_cq, axis=1)
        mel = librosa.feature.melspectrogram(y=f, sr=sr, n_mels=26)
        print(mel.shape)
        feat_17 = np.mean(mel, axis=1)
        feat_18 = np.std(mel, axis=1)
        zcrs = librosa.feature.zero_crossing_rate(f)
        feat_19 = np.sum(zcrs, axis = 1)

        duration = j1-i1
        feat_20 = np.array([duration])
        print('dur',duration)
        ####strength of beat across subband and full band , tempo and pulse
        onset_env = librosa.onset.onset_strength(f, sr=sr)

        av = librosa.util.normalize(onset_env)



        ## mean 
        feat_21 = np.mean(av, axis=0)

        feat_21 = np.array([feat_21])
        print(feat_5.shape)

        #stddev 
        feat_22 = np.std(av, axis=0)
        feat_22 = np.array([feat_22])


        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        print(tempo)
        feat_23 = tempo
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
        ## mean 
        feat_24 = np.mean(pulse, axis=0)
        feat_24= np.array([feat_24])

        #stddev 
        feat_25 = np.std(pulse, axis=0)
        feat_25 = np.array([feat_25])

        #### skewness and kurtosis of energy contour
        rms = librosa.feature.rms(S=S)
        rms_ = rms.T
        print(rms)
        feat_26 = kurtosis(rms_)
        feat_26 = skew(rms_)

        feat_27 = entropy(rms_)
        
        
        
        print(feat_1.shape, feat_2.shape, feat_3.shape, feat_4.shape, feat_5.shape, feat_6.shape, feat_7.shape, feat_8.shape, feat_9.shape, feat_10.shape,feat_11.shape, feat_12.shape, feat_13.shape, feat_14.shape, feat_15.shape, feat_16.shape, feat_17.shape, feat_18.shape, feat_19.shape, feat_20.shape,
                             feat_21.shape, feat_22.shape, feat_23.shape, feat_24.shape, feat_25.shape, feat_26.shape, feat_27.shape)
        final_f = np.hstack((feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7, feat_8, feat_9, feat_10,feat_11, feat_12, feat_13, feat_14, feat_15, feat_16, feat_17, feat_18, feat_19, feat_20,
                             feat_21, feat_22, feat_23, feat_24, feat_25, feat_26, feat_27))
        final_f = final_f[:,None]
        print(final_f.shape)
        feat_audio = np.vstack((feat_audio, final_f.T))

    n= feat_audio.shape[0]

    X0 = np.zeros((n,1))
    # X0 = np.ones((n,1))

    feat = np.hstack((feat_audio,X0))


    df=pd.DataFrame(feat)
    df.to_csv('/media/amrgaballah/Backup_Plus/Internship_exp/feat_firealarm.csv',index=None)
