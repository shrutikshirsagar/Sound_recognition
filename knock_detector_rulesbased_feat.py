###### In this code, I have implemented rule for selecting a area of intrest by observing abrupt changes in delta enrgy and peaks of delta energy.
###### Algorithm:
###### Step 1: calculate the rmse over frame_length = 2048 and hop_length = 512
###### Step 2: calculate the first derivative of rmse (delta energy)
###### Step 3: Find peaks of delta energy and ZCR of delta energy
###### Step 4: Remove abrupt changes in energy by looking at consecutive numbers in ZCR
###### Step 5: Algorithm for selection of starting and ending time of knock iterate each peak in ZCR, if peak is found in range of ZCR then get start and end time of knock.
###### Step 6: Once we get a region of interest, calculate features for doorknock : Spectral centroid, spectral bandwidth, spectral flatness, duration, entropy of energy, skewness and kurtosis of energy.

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
feat_audio = np.empty((0, 133))
output_cols =['cent_sample_m','cent_sample_s','cent_stft_m','cent_stft_s', 'bw_m', 'bw_s','contrast1_m', 'contrast2_m','contrast3_m','contrast4_m','contrast5_m','contrast6_m','contrast7_m','contrast1_s',
'contrast2_s','contrast3_s','contrast4_s','contrast5_s','contrast6_s','contrast7_s','flat_sam_m','flat_sam_s','flat_stft_m','flat_stft_s','feat_11c1', 'feat_11c2', 'feat_11c3', 'feat_11c4', 'feat_11c5','feat_11c6', 'feat_11c7', 'feat_11c8', 'feat_11c9', 'feat_11c10', 'feat_11c11', 'feat_11c12','feat_11c1', 'feat_11c2', 'feat_11c3', 'feat_11c4', 'feat_11c5','feat_11c6', 'feat_11c7', 'feat_11c8', 'feat_11c9', 'feat_11c10', 'feat_11c11', 'feat_11c12','feat_11c1', 'feat_11c2', 'feat_11c3', 'feat_11c4', 'feat_11c5','feat_11c6', 'feat_11c7', 'feat_11c8', 'feat_11c9', 'feat_11c10', 'feat_11c11', 'feat_11c12','feat_11c1', 'feat_11c2', 'feat_11c3', 'feat_11c4', 'feat_11c5','feat_11c6', 'feat_11c7', 'feat_11c8', 'feat_11c9', 'feat_11c10', 'feat_11c11', 'feat_11c12', 'mel_1','mel_2', 'mel_3','mel_4', 'mel_5', 'mel_6', 'mel_7', 'mel_8', 'mel_9', 'mel_10', 'mel_11', 'mel_12',  'mel_13', 'mel_14','mel_15', 'mel_16', 'mel_17', 'mel_18', 'mel_19', 'mel_20', 'mel_21', 'mel_22',  'mel_23', 'mel_24','mel_25', 'mel_26','mel_1','mel_2', 'mel_3','mel_4', 'mel_5', 'mel_6', 'mel_7', 'mel_8', 'mel_9', 'mel_10', 'mel_11', 'mel_12',  'mel_13', 'mel_14','mel_15', 'mel_16', 'mel_17', 'mel_18', 'mel_19', 'mel_20', 'mel_21', 'mel_22',  'mel_23', 'mel_24','mel_25', 'mel_26','zcr','dur','strn_m','stre_s','tempo','pulse_m''pulse_s','rms_kur','rms_skew','rms_en','labels']
path = '//media/amrgaballah/Backup_Plus/Internship_exp/google_audioset_sr/Firealarm/'
for filename_ in os.listdir(path):
    #### Step 1: calculate the rmse over frame_length = 2048 and hop_length = 512
    #### filename and parameters
    filename = os.path.join(path,filename_)
    x, fs = librosa.load(filename, sr= None)
    sr =fs
    frames_all = range(len(x))
    hop_length = 512
    frame_length = 2048

    #### peak_env and rmse
    peak_env = numpy.array([max(abs(x[i:i+frame_length]))for i in range(0, len(x), hop_length)])
    rmse = librosa.feature.rms(x, frame_length=frame_length, hop_length=hop_length, center=True)
    rmse = (rmse-np.mean(rmse))/np.std(rmse)
    frames = range(len(peak_env))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    ##### Step 2: calculate the first derivative of rmse (delta energy)
    ### first derivative of rmse
    A = np.diff(rmse)
    a = np.zeros((1,1))
    delta_peak = np.hstack((A,a))
    delta_peak = delta_peak.T
    delta_peak = delta_peak[:,0]
    ###### Step 3: Find peaks of delta energy and ZCR of delta energy
    #### find peaks of delta energy
    peaks, _ = find_peaks(delta_peak, height=0, distance=5, prominence=1,width=0)
   
    ### find zcr of delta rmse
    ZCR = numpy.where(numpy.diff(numpy.sign(delta_peak)))[0]
    ###### Step 4: Remove abrupt changes in energy by looking at consecutive numbers in ZCR
    ### false alarm if ZCR is too fast therefore we need to delete consecutive ZCR (it also represent the abrupt changes has occured in enrgy contour)

    zcr_del2 = []
    zcr_del1 = []
    for nums in range(1, len(ZCR)-1): 
        if ZCR[nums] == (ZCR[nums -1]+1) and ZCR[nums] == (ZCR[nums -2]+2):
     
            zcr_del2.append(nums -1)
            zcr_del2.append(nums -2)
        if ZCR[nums] == (ZCR[nums -1]+1):

            zcr_del1.append(nums-1)
    zcr_new = ZCR[zcr_del2]
    zero_crossings1 = [x for x in ZCR if x not in zcr_new]
    zcr_new1 = ZCR[zcr_del1]
    zero_crossings = [x for x in zero_crossings1 if x not in zcr_new1]
    #### first zero crossing indicate first knock
    start_time = [] 
    end_time = []
    if ZCR[0] == 0 and ZCR[1]>=4:
        start_time.append(t[ZCR[0]])
        end_time.append(t[ZCR[1]])
        zero_crossings_up= zero_crossings[2:] ### to remove first two zcr
    else:
        zero_crossings_up= zero_crossings

     ### updated zero crossing list
    ###### Step 5: Algorithm for selection of starting and ending time of knock iterate each peak in ZCR, if peak is found in range of ZCR then get start and end time of knock.
    ### algorithm for selection of strating and ending time of knock
    #### first take peaks and check the range in ZCR, if peak is find in ZCR then get start and end time
    for i in range(len(peaks)):
        for j in range(i, len(zero_crossings_up)):
   
            
            if peaks[i]==zero_crossings_up[-1] or peaks[i]==zero_crossings_up[-1]+1 or peaks[i]==zero_crossings_up[-1]+2:
                break
  
            elif peaks[i] == zero_crossings_up[j]+2 and peaks[i] ==zero_crossings_up[j+1]+1 and peaks[i] ==zero_crossings_up[j]and zero_crossings_up[j+2]-zero_crossings_up[j+1]>zero_crossings_up[j+1]
zero_crossings_up[j]:
         
                start_time.append(t[zero_crossings_up[j+1]])
                end_time.append(t[zero_crossings_up[j+2]])
                break
            elif peaks[i] == zero_crossings_up[j]+2 and peaks[i] ==zero_crossings_up[j+1]+1 and zero_crossings_up[j+2]-zero_crossings_up[j+1]>zero_crossings_up[j+1]-zero_crossings_up[j]:

                start_time.append(t[zero_crossings_up[j+1]])
                end_time.append(t[zero_crossings_up[j+2]])
                break
            elif peaks[i] == zero_crossings_up[j]+2:
                
  
                start_time.append(t[zero_crossings_up[j]])
                end_time.append(t[zero_crossings_up[j+1]])
                break

            elif peaks[i] == zero_crossings_up[j]+1:
  
                start_time.append(t[zero_crossings_up[j]])
                end_time.append(t[zero_crossings_up[j+1]])
                break
            elif peaks[i] == zero_crossings_up[j]:
  
                start_time.append(t[zero_crossings_up[j]])
                end_time.append(t[zero_crossings_up[j+1]])
                break
    ### total number of knock
    print('total number of knocks', len(start_time))


    ####  get samples from start time and end time from audio signal
    t_w = librosa.samples_to_time(frames_all, sr=sr)
    time_l = list(t_w)
    time_l1 = [round(num,5) for num in time_l]

    for i1, j1 in zip(start_time,end_time):
        ###### Step 6: Once we get a region of interest, calculate features for doorknock :
        ##### based on the audio time get teh smaples for calculating features
        out = np.where((round(i1,5)<(time_l1)) & ((time_l1)<round(j1,5)))
        f = x[out]
      
        S_ = np.abs(librosa.stft(f))
        #### centroid fetaure on samples and then mean and standard deviation from sample's centroid
        cent = librosa.feature.spectral_centroid(y=f, sr=sr)
     
        feat_1 = np.mean(cent, axis=1)
        feat_2 = np.std(cent, axis=1)
        S, phase = librosa.magphase(librosa.stft(y=f))
        #### centroid fetaure on STFT spectrogram and then mean and standard deviation from centroid
        centroid = librosa.feature.spectral_centroid(S=S)
     
        feat_3 = np.mean(centroid, axis=1)
        feat_4 = np.std(centroid, axis=1)
        #### Spectral bandwidth fetaure on STFT spectrogram and then mean and standard deviation from spectral bandwidth
        bw = librosa.feature.spectral_bandwidth(S=S)
      
        feat_5 = np.mean(bw, axis=1)
        feat_6 = np.std(bw, axis=1)
        #### Spectral contrast fetaure on STFT spectrogram and then mean and standard deviation from spectral contrast
        contrast = librosa.feature.spectral_contrast(S=S_, sr=sr)
     
        feat_7 = np.mean(contrast, axis=1)
        feat_8 = np.std(contrast, axis=1)
        #### flatness fetaure on amples and then mean and standard deviation from flatness
        flatness = librosa.feature.spectral_flatness(y=f)
     
        feat_9 = np.mean(flatness, axis=1)
        feat_10 = np.std(flatness, axis=1)
        #### Spectral flatness fetaure on STFT spectrogram and then mean and standard deviation from spectral flatness
        flat =librosa.feature.spectral_flatness(S=S)
    
        feat_11 = np.mean(flat, axis=1)
        feat_12 = np.std(flat, axis=1)

        #### chroma fetaure on STFT spectrogram and then mean and standard deviation from chroma
        chroma = librosa.feature.chroma_stft(S=S_, sr=sr)
       
        feat_13 = np.mean(chroma, axis=1)
        feat_14 = np.std(chroma, axis=1)
        #### chroma centroid fetaure on samples and then mean and standard deviation from chroma centroid
        chroma_cens = librosa.feature.chroma_cens(y=f, sr=sr)
    
        feat_15 = np.mean(chroma_cens, axis=1)
        feat_16 = np.std(chroma_cens, axis=1)
        #### Mel spectrogram fetaure and then mean and standard deviation from mel spectrogram
        mel = librosa.feature.melspectrogram(y=f, sr=sr, n_mels=26)
     
        feat_17 = np.mean(mel, axis=1)
        feat_18 = np.std(mel, axis=1)
        #### zero crossing rate fetaure
        zcrs = librosa.feature.zero_crossing_rate(f)
        feat_19 = np.sum(zcrs, axis = 1)
        ##### duration of knock feature 
        duration = j1-i1
        feat_20 = np.array([duration])
       
        ####strength of beat across  full band and then mean and standard deviation from it
        onset_env = librosa.onset.onset_strength(f, sr=sr)
       
        av = librosa.util.normalize(onset_env)
        ## mean 
        feat_21 = np.mean(av, axis=0)
        feat_21 = np.array([feat_21])
        #stddev 
        feat_22 = np.std(av, axis=0)
        feat_22 = np.array([feat_22])
        ##### tempo
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
      
        feat_23 = tempo
        ##### pulse and mean and standard deviation from it
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
        ## mean 
        feat_24 = np.mean(pulse, axis=0)
        feat_24= np.array([feat_24])

        #stddev 
        feat_25 = np.std(pulse, axis=0)
        feat_25 = np.array([feat_25])
      
        ##### energy contour
        rms = librosa.feature.rms(S=S)
        rms_ = rms.T
        #### kurtosis of energy contour
        feat_26 = kurtosis(rms_)
        #### skewness of energy contour
        feat_26 = skew(rms_)
        #### entropy of energy contour
        feat_27 = entropy(rms_)
        #### stacking all features
        final_f = np.hstack((feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7, feat_8, feat_9, feat_10,feat_11, feat_12, feat_13, feat_14, feat_15, feat_16, feat_17, feat_18, feat_19, feat_20,
                             feat_21, feat_22, feat_23, feat_24, feat_25, feat_26, feat_27))
        final_f = final_f[:,None]
        final_f = final_f.T
        
        feat_audio = np.vstack((feat_audio, final_f))

n= feat_audio.shape[0]
X0 = np.ones((n,1))
feat = np.hstack((feat_audio,X0))
df=pd.DataFrame(feat,columns=output_cols)
df.to_csv('//media/amrgaballah/Backup_Plus/Internship_exp/google_audio_sr_feat/Firealarm_feat_rulebased.csv',index=None)                       
