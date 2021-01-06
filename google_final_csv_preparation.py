import os
import numpy as np
import pandas as pd
exp_num = [1,2,3,4,5,6]
for num in exp_num:
    
    

    path = '/media/amrgaballah/Backup_Plus/Internship_exp/google_audioset_detector_all_results/feat_3/LR/Exp_'+ str(num)+ '/'
    out = '/media/amrgaballah/Backup_Plus/Internship_exp/google_audioset_detector_all_results/feat_3/LR/'
    output_cols = ['feat', 'Smoke_detector', 'Civildefense_siren', 'Ambulance', 'Policecar', 'Fireengine',
                 'Door_knock', 'Firealarm', 'Doorbell', 'Ding-dong', 'carhorn' , 'Baby_cry']
#     index_row = ['spectral_centroid','spectral_spread', 'spectral_crest', 'spectral_flux', 'spectral_rollof', 'psd_related_feat', 'feat_1']

    index_row = ['TEE_medium', 'TEE_iqr', 'TEE_iqr_normal', 'TEE_mean', 'TEE_all', 'Autocorrelation', 'STFT_mag', 'STFT_power', 'Harmonics', 'ERB_fft','ERB_game', 'feat_3']
    final = np.empty((12, 0))
    for filename in os.listdir(path):
        print(filename)
        df_ = pd.read_csv(os.path.join(path,filename)).values
        df_ = df_[:,-1][:,None]
        print(df_.shape)
        final = np.hstack((final, df_))

    print(final.shape)
    index_row = np.array([index_row]). T
    print(index_row.shape)
    final = np.hstack((index_row, final))
    print(final.shape)
    df=pd.DataFrame(final,columns=output_cols)
    
    df.to_csv(out + 'exp_' + str(num) + '.csv', index = None)


