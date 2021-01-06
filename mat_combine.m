clc;
clear all;
close all;

path_out = '//media/amrgaballah/Backup_Plus/Internship_exp/Exp_4/'
if(isempty(dir(path_out)))
    mkdir(path_out);
end
x1 = load('//media/amrgaballah/Backup_Plus/Internship_exp/Exp_4/fire_alarm_feat3.mat');
x2 = load('/media/amrgaballah/Backup_Plus/Internship_exp/Exp_4/no_fire_alarm_feat3.mat');
% x3= load('/media/amrgaballah/Backup_Plus/Internship_exp/Indoor_feat3_final_all/Door_knock/Door_knock_feat_fin.mat');
% x4 = load('/media/amrgaballah/Backup_Plus/Internship_exp/Indoor_feat3_final_all/Fire_alarm/Fire_alarm_feat_fin.mat');


% Concatenate
% feat_fin = [x1.feat_fin(2:end,:);x2.feat_fin(2:end,:);x3.feat_fin(2:end,:);x4.feat_fin(2:end,:)];
feat_fin = [x1.feat_fin(2:end,:);x2.feat_fin(2:end,:)];
% feat_fin = cell2mat(feat_fin)
% tmp=squeeze(cell2mat(feat_fin));
% [maxRealVal,imxR] = real(tmp);
% maxVal=tmp(imxR);
save(fullfile(path_out,'exp4_feat3.mat'), 'feat_fin')