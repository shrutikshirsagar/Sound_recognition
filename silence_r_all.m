clc;
clear all;
close all
path_in = '//media/amrgaballah/Backup_Plus/Internship_exp/Exp_1/no_doorbell_new/'
path_out ='//media/amrgaballah/Backup_Plus/Internship_exp/Exp_1/no_doorbell_new_sr/'
if(isempty(dir(path_out)))
    mkdir(path_out);
end
strFiles = strcat(path_in, '*.wav');
% For each audio file in audioFolder add noise at using the snr parameter 
F = dir(strFiles);

for iFile = 1:length(F)
    filename = fullfile(path_in, F(iFile).name)
    [data, fs] = audioread(filename);
%     Convert mono to stereo
%     if (size(data, 2)==2)
%         data = mean(data')';
%         data1 = data'
%     end
%     if (fs ~= 44100)
%         data = resample(data1,fs,44100);
%         data = data'
%     end
    fs=44100
    f_dur=0.1;
    f_len=f_dur*fs;
    N = length(data);
    no_frames=floor(N/f_len);
    new_data = zeros(N,1);
    count=0;
    frame = zeros(f_len,1);
    for k=1:no_frames
        frame=data((k-1)*f_len+1:f_len*k);
        
        max_val=max(frame);
        if(max_val>0.005)
            count=count+1;
            new_data((count-1)*f_len+1:count*f_len)=frame;
        end
    end
    
    new_data1 = nonzeros(new_data);
    out_f = fullfile(path_out, F(iFile).name)
    try
        audiowrite(out_f,new_data1,44100)
    catch
        continue
    end
end
    