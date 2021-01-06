clc;
clear all;
close all;

%baby_crying, Door_bell Door_knock, Fire_alarm, car_horn, Siren
path_in = '//media/amrgaballah/Backup_Plus/Internship_exp/google_audioset_features/feat_3/Emergency_vehicle/'
path_out = '//media/amrgaballah/Backup_Plus/Internship_exp/google_audioset_features/feat_3_final/Emergency_vehicle/'
if(isempty(dir(path_out)))
    mkdir(path_out);
end
strFiles = strcat(path_in, '*_stat.mat');
% For each audio file in audioFolder add noise at using the snr parameter 
F = dir(strFiles);

%audioFolder ='/home/shruti/Desktop/Segan/enhancement/cauchi_EURASIP_2015_SC/cauchi_EURASIP_2015_SC/functions/park_5dB/sim/boston1/angry/'
feat_fin = zeros(1,364)
feat_fin = num2cell(feat_fin)
for iFile = 1:length(F)
    filename = fullfile(path_in, F(iFile).name)
    disp(filename)
    ans = load(filename)

    f_1 = struct2cell(ans.ALLDESCSTATS_s.TEE_median)
    f_2 = struct2cell(ans.ALLDESCSTATS_s.TEE_iqr)
%     f_2 = f_2(end-1)
    f_3 = struct2cell(ans.ALLDESCSTATS_s.TEE_iqr_normal)
%     f_3 = f_3(end-1)
    f_4 = struct2cell(ans.ALLDESCSTATS_s.TEE_mean)
%     f_4 = f_4(end-1)
    f_5 = struct2cell(ans.ALLDESCSTATS_s.AS_median)
    f_6 = struct2cell(ans.ALLDESCSTATS_s.AS_iqr)
    f_7 = struct2cell(ans.ALLDESCSTATS_s.AS_iqr_normal)
    f_8 = struct2cell(ans.ALLDESCSTATS_s.AS_mean)
    f_9 = struct2cell(ans.ALLDESCSTATS_s.STFTmag_median)
    f_9 = f_9(3:end)
    f_10 = struct2cell(ans.ALLDESCSTATS_s.STFTmag_iqr)
    f_10 = f_10(3:end)
    f_11 = struct2cell(ans.ALLDESCSTATS_s.STFTmag_iqr_normal)
    f_11 = f_11(3:end)
    f_12 = struct2cell(ans.ALLDESCSTATS_s.STFTmag_mean)
    f_12 = f_12(3:end)
    f_13 = struct2cell(ans.ALLDESCSTATS_s.STFTpow_median)
    f_13 = f_13(3:end)
    f_14= struct2cell(ans.ALLDESCSTATS_s.STFTpow_iqr)
    f_14 = f_14(3:end)
    f_15 = struct2cell(ans.ALLDESCSTATS_s.STFTpow_iqr_normal)
    f_15 = f_15(3:end)
    f_16 = struct2cell(ans.ALLDESCSTATS_s.STFTpow_mean)
    f_16 = f_16(3:end)
    f_17 = struct2cell(ans.ALLDESCSTATS_s.Harmonic_median)
    f_18= struct2cell(ans.ALLDESCSTATS_s.Harmonic_iqr)
    f_19 = struct2cell(ans.ALLDESCSTATS_s.Harmonic_iqr_normal)
    f_20= struct2cell(ans.ALLDESCSTATS_s.Harmonic_mean)
    f_21 = struct2cell(ans.ALLDESCSTATS_s.ERBfft_median)
    f_21 = f_21(3:end)
    f_22 = struct2cell(ans.ALLDESCSTATS_s.ERBfft_iqr)
    f_22 = f_22(3:end)
    f_23 = struct2cell(ans.ALLDESCSTATS_s.ERBfft_iqr_normal)
    f_23 = f_23(3:end)
    f_24 = struct2cell(ans.ALLDESCSTATS_s.ERBfft_mean)
    f_24 = f_24(3:end)
    f_25 = struct2cell(ans.ALLDESCSTATS_s.ERBgam_median)
    f_25 = f_25(3:end)
    f_26 = struct2cell(ans.ALLDESCSTATS_s.ERBgam_iqr)
    f_26 = f_26(3:end)
    f_27 = struct2cell(ans.ALLDESCSTATS_s.ERBgam_iqr_normal)
    f_27 = f_27(3:end)
    f_28 = struct2cell(ans.ALLDESCSTATS_s.ERBgam_mean)
    f_28 = f_28(3:end)
    final = [f_1;f_2;f_3;f_4;f_5;f_6;f_7;f_8;f_9;f_10;f_11;f_12;f_13;f_14;f_15;f_16;f_17;f_18;f_19;f_20;f_21;f_22;f_23;f_24;f_25;f_26;f_27;f_28]
    final2 = final.'
%     X = zeros(final2)
%     X = ones
    feat_fin = vertcat(feat_fin, final2)
    
    
end
%final_r = zeros(length(feat_fin),1)
% final_r  = 3.*ones(length(feat_fin),1)
[m,n] = size(feat_fin)
final_r  = ones(m,1)
feat_fin = cell2mat(feat_fin)
% final_r = num2cell(final_r)
feat_fin = real(feat_fin)
feat_fin = horzcat(feat_fin, final_r)
save(fullfile(path_out,'Emergency_vehicle.mat'), 'feat_fin')
