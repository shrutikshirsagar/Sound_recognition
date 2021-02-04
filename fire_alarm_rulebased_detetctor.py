###### In this code, I have implemented rule for selecting a area of intrest by observing abrupt changes in RMSE energy.
###### Algorithm:
###### Step 1: calculate the rmse over frame_length = 2048 and hop_length = 512
###### Step 2: Based on rmse value make  a rule for detetction of region of intrest for fire alarm fetaures. Algorithm for selection of starting and ending time of fire alarm when it crosses zero, raised a flag for strating featur extraction get a start time and then stop flag as soon as it goes to negative value then the end time of fire alarm beep. 
###### Step 3: Once we get a region of interest, calculate features for Fire alarm : pitch, harmonics, harmonic ratio, Spectral centroid, spectral bandwidth, spectral flatness, duration, entropy of energy, skewness and kurtosis of energy.

from __future__ import division
from am_analysis import am_analysis as ama
import numpy as np

import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  
import os
import iracema
import numpy as np
import pysptk
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
import librosa
import numpy as np
import pysptk
from scipy.io import wavfile
import pandas as pd
import numpy
from scipy.signal import find_peaks
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.stats import entropy
feat_audio = np.empty((0,206))
path = '/media/amrgaballah/Backup_Plus/Internship_exp/final_wavfiles_sr/Fire_alarm/'
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
        f_ = x[out]
        f = y[out]
       
        if f.shape[0]<=2000:
            continue

        #### pitch using SWIPE pysptk toolbox
        f0 =  pysptk.swipe(f_.astype(np.float64), fs=fs, hopsize=512, min=60, max=3500, otype="f0")

        feat_1 = np.mean(np.array([f0]), axis=1)
        feat_2 = np.std(np.array([f0]), axis=1)
        #### pitch using RAPT  pysptk toolbox
        f01 =  pysptk.rapt(f_.astype(np.float32), fs=fs, hopsize=512, min=60, max=3500, otype="f0")

        feat_3 = np.mean(np.array([f01]), axis=1)
        feat_4 = np.std(np.array([f01]), axis=1)
        #### pitch using HPS IRACEMA toolbox
        a2 = iracema.Audio(f, 44100)
        window_size, hop_size = 2048, 512
        fft = iracema.spectral.fft(a2, window_size, hop_size)
        pitch = iracema.pitch.hps(fft,minf0=60, maxf0=3500)
        f0_ = pitch.data

        feat_5 = np.mean(np.array([f0_]), axis=1)
        feat_6 = np.std(np.array([f0_]), axis=1)
        #### skewness, kurtosis, entropy and ZCR of energy contour
        S, phase = librosa.magphase(librosa.stft(f))
        S_ = np.abs(librosa.stft(f))
        rms = librosa.feature.rms(S=S)
        rms_ = rms.T
       
        feat_7 = kurtosis(rms_)
        feat_8 = skew(rms_)

        feat_9 = entropy(rms_)
    #     zcrs_rmse = librosa.feature.zero_crossing_rate(rms_)
    #     feat_10 = np.sum(zcrs_rmse, axis = 1)
        ##### duration fetaures 
        duration = j1-i1
        feat_11 = np.array([duration])
       

        ##### ZCR on time series audio
        zcrs = librosa.feature.zero_crossing_rate(f)
        feat_12 = np.sum(zcrs, axis = 1)
    #     #### harmonics related fetaures
        harmonics = iracema.harmonics.extract(fft, pitch, nharm=12)
    #     h_f = harmonics['frequency'].data
    #     feat_13 = np.mean(np.array([h_f]), axis=1)
    #     feat_14 = np.std(np.array([h_f]), axis=1)
       ### high frquency content
        hfc = iracema.features.hfc(fft, method='energy')
        hfc_ = hfc.data
        feat_15 = np.mean(np.array([hfc_]), axis=1)
        feat_16 = np.std(np.array([hfc_]), axis=1)


        ##### harmonic energy
        h_mag = harmonics['magnitude']
        h_e = iracema.features.harmonic_energy(h_mag)
        he_ = h_e.data
        feat_17 = np.mean(np.array([he_]), axis=1)
        feat_18 = np.std(np.array([he_]), axis=1)
        ##### noisyness
        noisyness = iracema.features.noisiness(fft, h_mag)
        noisyness_ = noisyness.data
        feat_19 = np.mean(np.array([noisyness_]), axis=1)
        feat_20 = np.std(np.array([noisyness_]), axis=1)

  
        onset_env = librosa.onset.onset_strength(f, sr=sr)

        av = librosa.util.normalize(onset_env)



        ## mean 
        feat_29 = np.mean(av, axis=0)

        feat_29 = np.array([feat_29])
        print(feat_5.shape)

        #stddev 
        feat_30 = np.std(av, axis=0)
        feat_30 = np.array([feat_30])


        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        print(tempo)
        feat_31 = tempo
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
        ## mean 
        feat_32 = np.mean(pulse, axis=0)
        feat_32= np.array([feat_32])

        #stddev 
        feat_33 = np.std(pulse, axis=0)
        feat_33 = np.array([feat_33])

        h = librosa.effects.harmonic(y=f)
        tonnetz = librosa.feature.tonnetz(y=h, sr=sr)
        feat_34 = np.mean(tonnetz, axis = 1)
        feat_35 = np.std(tonnetz, axis=1)
        chroma = librosa.feature.chroma_stft(S=S_, sr=sr)
        print(chroma.shape)
        feat_36 = np.mean(chroma, axis=1)
        feat_37 = np.std(chroma, axis=1)
        chroma_cens = librosa.feature.chroma_cens(y=f, sr=sr)
        print(chroma_cens.shape)
        feat_38 = np.mean(chroma_cens, axis=1)
        feat_39 = np.std(chroma_cens, axis=1)
        chroma_cq = librosa.feature.chroma_cqt(y=f, sr=sr)
        print(chroma_cq.shape)
        feat_40 = np.mean(chroma_cq, axis=1)
        feat_41 = np.std(chroma_cq, axis=1)
        mel = librosa.feature.melspectrogram(y=f, sr=sr, n_mels=26)
        print(mel.shape)
        feat_42 = np.mean(mel, axis=1)
        feat_43 = np.std(mel, axis=1)
        ##### spectral related fetaures


        ### centroid of time series data

        cent = librosa.feature.spectral_centroid(y=f, sr=sr)
        print(cent.shape)
        feat_44 = np.mean(cent, axis=1)
        feat_45= np.std(cent, axis=1)

        ### spectral centroid 
        centroid = librosa.feature.spectral_centroid(S=S)
        feat_46 = np.mean(centroid, axis=1)
        feat_47 = np.std(centroid, axis=1)
        print(feat_3, feat_4)


        bw = librosa.feature.spectral_bandwidth(S=S)
        print(bw.shape)
        feat_48 = np.mean(bw, axis=1)
        feat_49 = np.std(bw, axis=1)
        contrast = librosa.feature.spectral_contrast(S=S_, sr=sr)
        print(contrast.shape)
        feat_50 = np.mean(contrast, axis=1)
        feat_51 = np.std(contrast, axis=1)
        flatness = librosa.feature.spectral_flatness(y=f)
        print(flatness.shape)
        feat_52 = np.mean(flatness, axis=1)
        feat_53 = np.std(flatness, axis=1)
        flat =librosa.feature.spectral_flatness(S=S)
        print(flat.shape)
        feat_54 = np.mean(flat, axis=1)
        feat_55 = np.std(flat, axis=1)
        rolloff_85 = librosa.feature.spectral_rolloff(y=f, sr=sr, roll_percent=0.85)
        feat_56 = np.mean(rolloff_85, axis=1)
        feat_57 = np.std(rolloff_85, axis=1)
        rolloff_05 = librosa.feature.spectral_rolloff(y=f, sr=sr, roll_percent=0.05)
        feat_58 = np.mean(rolloff_05, axis=1)
        feat_59 = np.std(rolloff_05, axis=1)
   

        ####
        spectral_spread = iracema.features.spectral_spread(fft)
        spectral_spread_ = spectral_spread.data
        feat_61 = np.mean(np.array([spectral_spread_]), axis=1)
        feat_62 = np.std(np.array([spectral_spread_]), axis=1)
   
        spectral_flux = iracema.features.spectral_flux(fft)
        spectral_flux_ = spectral_flux.data
        feat_67 = np.mean(np.array([spectral_flux_]), axis=1)
        feat_68 = np.std(np.array([spectral_flux_]), axis=1)
        spectral_entropy = iracema.features.spectral_entropy(fft)
        spectral_entropy_ = spectral_entropy.data

        feat_69 = np.mean(np.array([spectral_entropy_]), axis=1)
        feat_70 = np.std(np.array([spectral_entropy_]), axis=1)
        spectral_energy = iracema.features.spectral_energy(fft)
        spectral_energy_ = spectral_energy.data
        feat_71 = np.mean(np.array([spectral_energy_]), axis=1)
        feat_72 = np.std(np.array([spectral_energy_]), axis=1)
        fs=sr  #sampling frequency
        w_size =  int(0.004 * fs)   # window size in seconds

        w_shift = int(0.002 * fs)   # window overlap

        #compute short time fourier transform
        rfft_spect_h = ama.strfft_spectrogram(f, fs, w_size, w_shift, win_function = 'hamming' )
        power_spect_h = sum(sum(rfft_spect_h['power_spectrogram']))[0] * rfft_spect_h['freq_delta'] * rfft_spect_h['time_delta']

        feat = np.empty((0, 12))

        for ix in range(0,rfft_spect_h['power_spectrogram'].shape[0]):
            psd=(rfft_spect_h['power_spectrogram'][ix,:,0])
            psd=psd/np.sum(psd)
            a = rfft_spect_h['freq_axis']

            b = np.where(a<5000) 
            feat_73 = psd[b]
            feat_73 = np.sum(feat_73)

            c = np.where((a >5000) & (a<10000))
            feat_74 = psd[c]
            feat_74 = np.sum(feat_74)

            d = np.where((a >10000) & (a<15000))
            feat_75 = psd[d]
            feat_75 = np.sum(feat_75)
            feat_76=feat_73/(feat_74+feat_75)

            feat_77=feat_74/(feat_73+feat_75)

            feat_78=feat_75/(feat_73+feat_74)
            peaks, properties = find_peaks(psd)
            a1 = psd[peaks]
            a1=np.sort(a1)
            if len(a1)>=3:
                first, second, third=a1[-1], a1[-2], a1[-3]
            elif len(a1)==2:
                first,second,third=a1[-1], a1[-2], np.min(psd)
            elif len(a1)==1:
                first,second,third=a1[-1], np.min(psd), np.min(psd)
            elif len(a1)==0:
                first,second,third=np.max(psd), np.min(psd), np.min(psd)
            feat_79=first/second
            feat_80=first/third
            feat_81 = kurtosis(psd)
            feat_82 = skew(psd) 
            feat_83 = -np.sum(psd*np.log(psd))
            feat_84 = a[np.argmax(psd)]
            feat_c=np.hstack((feat_73, feat_74, feat_75, feat_76, feat_77, feat_78, feat_79, feat_80, feat_81, feat_82, feat_83,feat_84))
      
            feat=np.vstack((feat,feat_c))

        df1 = np.sum(feat, axis = 0)

        df1 = df1[None,:]

      

        final_f = np.hstack((feat_1[None,:], feat_2[None,:], feat_3[None,:], feat_4[None,:], feat_5[None,:], feat_6[None,:], feat_7[None,:], feat_8[None,:], feat_9[None,:], feat_11[None,:], feat_12[None,:],  feat_15[None,:], feat_16[None,:], feat_17[None,:], feat_18[None,:], feat_19[None,:], feat_20[None,:],
                                   feat_29[None,:], feat_30[None,:], feat_31[None,:], feat_32[None,:], feat_33[None,:], feat_34[None,:], feat_35[None,:], feat_36[None,:], feat_37[None,:], feat_38[None,:], feat_39[None,:], feat_40[None,:], feat_41[None,:], feat_42[None,:], feat_43[None,:], feat_44[None,:], feat_45[None,:], feat_46[None,:], feat_47[None,:], feat_48[None,:], feat_49[None,:], feat_50[None,:],
                                feat_51[None,:], feat_52[None,:], feat_53[None,:], feat_54[None,:], feat_55[None,:], feat_56[None,:], feat_57[None,:], feat_58[None,:], feat_59[None,:],
                             feat_61[None,:], feat_62[None,:], feat_67[None,:], feat_68[None,:], feat_69[None,:],  feat_70[None,:],feat_71[None,:], feat_72[None,:], df1))
       
        feat_audio = np.vstack((feat_audio, final_f))

   

n= feat_audio.shape[0]


X00 = np.ones((n,1))

feat_rule = np.hstack((feat_audio,X00))


df_final=pd.DataFrame(feat_rule)
df_final.to_csv('/media/amrgaballah/Backup_Plus/Internship_exp/feat_firealarm_rulebased_all.csv',index=None)   
