import matplotlib.pyplot as plt
import librosa
import numpy as np
import librosa.display
from surfboard.sound import Waveform
import numpy as np
import os
import pandas as pd
path = '//media/shruti/Data/Internship_data/Experiment/Indoor_16Khz/Fire_alarm/'
final_f = np.empty((0, 35))
for f in os.listdir(path):
    filename = os.path.join(path,f)
    print(filename)
    y, sr = librosa.load(filename, sr=None)
#     try: 
#         y = np.asarray(y)

#         if y.shape[1] == 2:
#             y = np.mean(y, axis=1)
#     except:
#         y = y
#     try: 
        
#         if y.shape[1] == 2:
#             continue
#     except:
#         y = y
    print(y.shape)
    sound = Waveform(path=filename, sample_rate=44100)



    ## F1 F2 F3 F4 
    formants = sound.formants()
    F1 = np.asarray([formants['f1']])
    F1 = F1[:,None]
    F2 = np.asarray([formants['f2']])
    F2 = F2[:,None]
    F3 = np.asarray([formants['f3']])
    F3 = F3[:,None]
    F4 = np.asarray([formants['f4']])
    F4 = F4[:,None]
   
    #### statistics over harmonics


    ### F1/F2
    H1 = F2 - F1

    ### F1/F3

    H2 = F3-F1
    ### F1/F4
    H3 = F4 - F1

    ### F2/F3

    H4 = F3 - F2
    ###F2/F4

    H5 = F4 - F2
    ### F3/F4
    H6 = F4 - F3
  
    ### stratistics over F0
    ### F0
    # f0_contour = sound.f0_contour()
    # print(f0_contour)

    
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=5000)
    F0 = pitches[np.nonzero(pitches)]
    # np.set_printoptions(threshold=np.nan)

    F0 = F0[:, None]
   
    ## mean
    mean_F0 = np.mean(F0, axis = 0)
    mean_F0 = mean_F0[:,None]
    print(mean_F0)
    ## std dev
    std_F0 = np.std(F0, axis = 0)
    std_F0 = std_F0[:,None]
    print(std_F0)
    ## min
    min_F0 = np.min(F0, axis = 0)
   
    ## max
    max_F0 = np.max(F0, axis = 0)
    
    ## range
    range_F0 = max_F0-min_F0
    range_F0 = range_F0[:,None]
    min_F0 = min_F0[:,None]
    print(min_F0)
    max_F0 = max_F0[:,None]
    print(max_F0)
    print(range_F0)
    rms_i = librosa.feature.rms(y=y)
    
    ### stratistics over Intensity

    ## mean
    mean_rms = np.mean(rms_i, axis = 1)
   
    ## std dev
    std_rms = np.std(rms_i, axis = 1)
   
    ## min
    min_rms = np.min(rms_i, axis = 1)
    
    ## max
    max_rms = np.max(rms_i, axis = 1)
    

    ## range
    range_rms = max_rms-min_rms
    
    mean_rms = mean_rms[:,None]
    std_rms = std_rms[:,None]
    min_rms = min_rms[:,None]
    max_rms = max_rms[:,None]
    range_rms = range_rms[:,None]
    ### Spectral fetaures
    
    ### spectral centroid

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    ### spectral bandwidth
     ## mean
    mean_cent = np.mean(cent, axis = 1)
    
    ## std dev
    std_cent = np.std(cent, axis = 1)
    
    ## min
    min_cent = np.min(cent, axis = 1)
    
    ## max
    max_cent = np.max(cent, axis = 1)
   

    ## range
    range_cent = max_cent-min_cent
    mean_cent = mean_cent[:,None]
    std_cent = std_cent[:,None]
    min_cent = min_cent[:,None]
    max_cent = max_cent[:,None]
    range_cent = range_cent[:,None]
    
    
    
    
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
   
    ### spectral bandwidth
    ## mean
    mean_spec_bw = np.mean(spec_bw, axis = 1)
   
    ## std dev
    std_spec_bw = np.std(spec_bw, axis = 1)
  
    ## min
    min_spec_bw = np.min(spec_bw, axis = 1)
    
    ## max
    max_spec_bw = np.max(spec_bw, axis = 1)
   
    ## range
    range_spec_bw = max_spec_bw-min_spec_bw
   
    mean_spec_bw = mean_spec_bw[:,None]
    std_spec_bw = std_spec_bw[:,None]
    min_spec_bw = min_spec_bw[:,None]
    max_spec_bw = max_spec_bw[:,None]
    range_spec_bw = range_spec_bw[:,None]
    
    
    S = np.abs(librosa.stft(y))
#     contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
  
#      ## mean
#     mean_contrast = np.mean(contrast, axis = 1)
   
#     ## std dev
#     std_contrast = np.std(contrast, axis = 1)
  
#     ## min
#     min_contrast= np.min(contrast, axis = 1)
 
#     ## max
#     max_contrast = np.max(contrast, axis = 1)
   

#     ## range
#     range_contrast = max_contrast-min_contrast
    
#     mean_contrast = mean_contrast[:,None]
#     std_contrast = std_contrast[:,None]
#     min_contrast = min_contrast[:,None]
#     max_contrast = max_contrast[:,None]
#     range_contrast = range_contrast[:,None]
    
    ### Spectral skewness

    flatness = librosa.feature.spectral_flatness(y=y)
   
    ## mean
    mean_flatness = np.mean(flatness, axis = 1)
   
    ## std dev
    std_flatness = np.std(flatness, axis = 1)
    
    ## min
    min_flatness= np.min(flatness, axis = 1)
    
    ## max
    max_flatness = np.max(flatness, axis = 1)
   

    ## range
    range_flatness = max_flatness-min_flatness
    mean_flatness = mean_flatness[:,None]
    std_flatness = std_flatness[:,None]
    min_flatness = min_flatness[:,None]
    max_flatness = max_flatness[:,None]
    range_flatness = range_flatness[:,None]
    print( F1.shape, F2.shape, F3.shape, F4.shape, H1.shape, H2.shape, H3.shape, H4.shape, H5.shape, H6.shape) 
    print(mean_F0.shape, std_F0.shape, min_F0.shape, max_F0.shape, range_F0.shape) 
    print(mean_rms.shape, std_rms.shape, min_rms.shape, max_rms.shape, range_rms.shape) 
    print(mean_cent.shape, std_cent.shape, min_cent.shape, max_cent.shape, range_cent.shape)  
    print(mean_spec_bw.shape, std_spec_bw.shape, min_spec_bw.shape, max_spec_bw.shape, range_spec_bw.shape)
      
    print(mean_flatness.shape, std_flatness.shape, min_flatness.shape, max_flatness.shape, range_flatness.shape)
    
    
    
    
    feat = np.hstack((F1, F2, F3, F4, H1, H2, H3, H4, H5, H6, mean_F0, std_F0, min_F0, max_F0, range_F0, mean_rms, std_rms, min_rms, max_rms, range_rms, mean_cent, std_cent, min_cent, max_cent, range_cent,  mean_spec_bw, std_spec_bw, min_spec_bw, max_spec_bw, range_spec_bw, mean_flatness, std_flatness, min_flatness, max_flatness, range_flatness))
    print(feat.shape)
    final_f = np.vstack((final_f, feat))
print(final_f.shape)
df=pd.DataFrame(final_f)
df.to_csv('//media/shruti/Data/Internship_data/Experiment/Features/Fire_alarm_proposed_feat.csv',index=None)
    ### Spectral roll off

#     # Approximate maximum frequencies with roll_percent=0.99
#     rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
#     print(rolloff.shape)
#     # Approximate minimum frequencies with roll_percent=0.01
#     rolloff_min = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.01)
#     print(rolloff_min.shape)

    ##polynomial
    
#     p0 = librosa.feature.poly_features(S=S, order=0)
#     p1 = librosa.feature.poly_features(S=S, order=1)
#     p2 = librosa.feature.poly_features(S=S, order=2)

#     ## zero crossing rate
#     ZCR = librosa.feature.zero_crossing_rate(y)
#     print(ZCR.shape)
      
