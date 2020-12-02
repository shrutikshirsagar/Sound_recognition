from __future__ import division
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
def FeatureSpectralFlatness(X, f_s):

    norm = X.mean(axis=0, keepdims=True)
    norm[norm == 0] = 1

    X = np.log(X + 1e-20)

    vtf = np.exp(X.mean(axis=0, keepdims=True)) / norm

    return (vtf)
path = '/media/amrgaballah/Backup_Plus/Internship_exp/Exp_2/VAD/no_baby_cry/'
output_cols = ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10','feat_11c1', 'feat_11c2', 'feat_11c3', 'feat_11c4', 'feat_11c5','feat_11c6', 'feat_11c7', 'feat_12c1', 'feat_12c2', 'feat_12c3', 'feat_12c4', 'feat_12c5','feat_12c6', 'feat_12c7',
               'feat_13c1', 'feat_13c2', 'feat_13c3', 'feat_13c4', 'feat_13c5','feat_13c6', 'feat_13c7', 'feat_14c1', 'feat_14c2', 'feat_14c3', 'feat_14c4', 'feat_14c5','feat_14c6', 'feat_14c7',
               'feat_15c1', 'feat_15c2', 'feat_15c3', 'feat_15c4', 'feat_15c5','feat_15c6', 'feat_15c7', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20',
'feat_21', 'feat_22', 'feat_23', 'feat_24', 'feat_25', 'feat_26', 'feat_27', 'feat_28', 'feat_29', 'feat_30', 'bnd_pwr_0_25','bnd_pwr_25_50','bnd_pwr_50_150','entropy','p1_p2','p1_p3','kurt','sknew','flatness','peak_f','ratio_25', 'ratio_50','ratio_150', 'label']
feat_audio = np.empty((0, 73))
for f in os.listdir(path):
    print(f)
    br, sr = librosa.load(os.path.join(path, f), sr = None)
    if librosa.get_duration(y=br, sr=sr) == 0.0:
        continue
    cent =  librosa.feature.spectral_centroid(y=br, sr=sr)
    print(cent.shape)

    ## min 
    feat_1 = np.min(cent, axis=1)


    ## max 
    feat_2 = np.max(cent, axis=1)

    ## mean 
    feat_3 = np.mean(cent, axis=1)

    ##range
    feat_4 = feat_2-feat_1

    #stddev 
    feat_5 = np.std(cent, axis=1)


    bw =  librosa.feature.spectral_bandwidth(y=br, sr=sr)

    ## min 
    feat_6 = np.min(bw, axis=1)

    ## max 
    feat_7 = np.max(bw, axis=1)

    ## mean 
    feat_8 = np.mean(bw, axis=1)

    ## range 
    feat_9 = feat_7-feat_6

    #stddev 
    feat_10 = np.std(bw, axis=1)


    S = np.abs(librosa.stft(br))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    # min 
    feat_11 = np.min(contrast, axis=1)

    ## max 
    feat_12 = np.max(contrast, axis=1)

    ## mean 
    feat_13 = np.mean(contrast, axis=1)

    ## range 
    feat_14 = feat_12-feat_11

    #stddev 
    feat_15 = np.std(contrast, axis=1)



    flatness = librosa.feature.spectral_flatness(y=br)
    ## min 
    feat_16 = np.min(flatness, axis=1)

    ## max 
    feat_17 = np.max(flatness, axis=1)

    ## mean 
    feat_18 = np.mean(flatness, axis=1)

    ## range 
    feat_19 = feat_17-feat_16

    #stddev 
    feat_20 = np.std(flatness, axis=1)




    roll_85 = librosa.feature.spectral_rolloff(y=br, sr=sr, roll_percent=0.85)
    # min 
    feat_21 = np.min(roll_85, axis=1)

    ## max 
    feat_22 = np.max(roll_85, axis=1)

    ## mean 
    feat_23 = np.mean(roll_85, axis=1)

    ## range 
    feat_24 = feat_22-feat_21

    #stddev 
    feat_25 = np.std(roll_85, axis=1)



    roll_20 = librosa.feature.spectral_rolloff(y=br, sr=sr, roll_percent=0.20)
    ## min 
    feat_26 = np.min(roll_20, axis=1)

    ## max 
    feat_27 = np.max(roll_20, axis=1)

    ## mean 
    feat_28 = np.mean(roll_20, axis=1)

    ## range 
    feat_29 = feat_17-feat_16

    #stddev 
    feat_30 = np.std(roll_20, axis=1)


    feat_1 = feat_1[None,:]
 
    feat_2 = feat_2[None,:]
   
    feat_3 = feat_3[None,:]
  
    feat_4 = feat_4[None,:]
   
    feat_5 = feat_5[None,:]
    
    feat_6 = feat_6[None,:]
   
    feat_7 = feat_7[None,:]
   
    feat_8 = feat_8[None,:]
   
    feat_9 = feat_9[None,:]
   
    feat_10 = feat_10[None,:]
  


    feat_11 = feat_11[None,:]
   
    feat_12 = feat_12[None,:]
 
    feat_13 = feat_13[None,:]

    feat_14 = feat_14[None,:]
    
    feat_15 = feat_15[None,:]
    
    feat_16 = feat_16[None,:]
    
    feat_17 = feat_17[None,:]
   
    feat_18 = feat_18[None,:]
    
    feat_19 = feat_19[None,:]
    
    feat_20 = feat_20[None,:]
    


    feat_21 = feat_21[None,:]
   
    feat_22 = feat_22[None,:]
    
    feat_23 = feat_23[None,:]
   
    feat_24 = feat_24[None,:]
    
    feat_25 = feat_25[None,:]
    
    feat_26 = feat_26[None,:]
    
    feat_27 = feat_27[None,:]
    
    feat_28 = feat_28[None,:]
    
    feat_29 = feat_29[None,:]
   
    feat_30 = feat_30[None,:]
    


    br=(br-np.mean(br))/np.std(br)
    fs=sr  #sampling frequency
    w_size =  int(0.004 * fs)   # window size in seconds
  
    w_shift = int(0.002 * fs)   # window overlap
    
    #compute short time fourier transform
    rfft_spect_h = ama.strfft_spectrogram(br, fs, w_size, w_shift, win_function = 'hamming' )
    power_spect_h = sum(sum(rfft_spect_h['power_spectrogram']))[0] * rfft_spect_h['freq_delta'] * rfft_spect_h['time_delta']

    feat = np.empty((0, 13))
    for ix in range(0,rfft_spect_h['power_spectrogram'].shape[0]):
        psd=(rfft_spect_h['power_spectrogram'][ix,:,0])
        psd=psd/np.sum(psd)
        a = rfft_spect_h['freq_axis']

        b = np.where(a<5000) 
        feat_31 = psd[b]
        feat_31 = np.sum(feat_31)
    #     feat_P1=np.vstack((feat_P1,feat_1))
        #print(feat_1)

        c = np.where((a >5000) & (a<10000))
        feat_32 = psd[c]
        feat_32 = np.sum(feat_32)
    #     feat_P2=np.vstack((feat_P2,feat_2))
        #print(feat_2)

        d = np.where((a >10000) & (a<15000))
        feat_33 = psd[d]
        feat_33 = np.sum(feat_33)
    #     feat_P3=np.vstack((feat_P3,feat_3))
        #print(feat_3)

        feat_34 = -np.sum(psd*np.log(psd))
    #     feat_P4=np.vstack((feat_P4,feat_4))
        #print(feat_4)






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


    #     print(first, second, third)
        feat_35=first/second
    #     feat_P5=np.vstack((feat_P5,feat_5))
    #     print(feat_5)
        feat_36=first/third
    #     feat_P6=np.vstack((feat_P6,feat_6))
    #     print(feat_6)
        feat_37 = kurtosis(psd)
    #     feat_P7=np.vstack((feat_P7,feat_7))
    #     print(feat_7)
        feat_38 = skew(psd) 
    #     feat_P8=np.vstack((feat_P8,feat_8))
    #     print(feat_8)




        feat_39 = FeatureSpectralFlatness(psd, fs)
    #     feat_P10=np.vstack((feat_P10,feat_10))
    #     print(feat_10)
        feat_40 = a[np.argmax(psd)]
    #     feat_P11=np.vstack((feat_P11,feat_11))
        #in band vs out of band ratio for feat 1
        feat_41=feat_31/(feat_32+feat_33)
    #     feat_P12=np.vstack((feat_P12,feat_12))
        #in band vs out of band ratio for feat 1
        feat_42=feat_32/(feat_31+feat_33)
    #     feat_P13=np.vstack((feat_P13,feat_13))
        #in band vs out of band ratio for feat 1
        feat_43=feat_33/(feat_31+feat_32)



        feat_c=np.hstack((feat_31, feat_32, feat_33, feat_34, feat_35, feat_36, feat_37, feat_38, feat_39, feat_40, feat_41,feat_42,feat_43))
       
        feat=np.vstack((feat,feat_c))
    
    df1 = np.sum(feat, axis = 0)
    df1 = df1[None,:]
   

    final_f = np.hstack((feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7, feat_8, feat_9, feat_10,feat_11, feat_12, feat_13, feat_14, feat_15, feat_16, feat_17, feat_18, feat_19, feat_20,
                         feat_21, feat_22, feat_23, feat_24, feat_25, feat_26, feat_27, feat_28, feat_29, feat_30, df1))
  
    feat_audio = np.vstack((feat_audio, final_f))
print(feat_audio.shape)
n= feat_audio.shape[0]
print(n)
X0 = np.zeros((n,1))
#X0 = np.ones((n,1))
print(X0.shape)
feat = np.hstack((feat_audio,X0))
print(feat.shape)

df=pd.DataFrame(feat,columns=output_cols)
df.to_csv('/media/amrgaballah/Backup_Plus/Internship_exp/Exp_2/VAD/no_baby_cry.csv',index=None)

