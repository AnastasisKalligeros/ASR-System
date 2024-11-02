clc
clear

%sample frequency in hz
s=6000;
%lowest allowed frequency in hz
l=40;

[recording,frequency]=audioread('eight_seven_four_nine_one.wav');
rec_resampled=resample(recording,s,frequency);
rec_resampled=highpass(rec_resampled,l,s);

energy=zeros(size(rec_resampled));
fourier_frequencies=zeros(size(rec_resampled));

%window size=5
w=600;
%we will calculate the energy of the soundwave and the fourier frequencies in windows of length w
for i=1:size(rec_resampled,1)
    if(i+w>size(rec_resampled,1))
        for j=i:size(rec_resampled,1)
            energy(i)=energy(i)+rec_resampled(j)^2;
        end
        fourier_frequencies(i)=sum(abs(fft(rec_resampled(i:size(rec_resampled,1)),w)));
    else
        for j=i:i+w
            energy(i)=energy(i)+rec_resampled(j)^2;
        end
        fourier_frequencies(i)=sum(abs(fft(rec_resampled(i:w),w)));
    end
    fourier_frequencies(i)=sum(abs(fft(rec_resampled(i),w)));
end

%smoothing out the fourier frequencies by applying the movmean function
fourier_frequencies=movmean(fourier_frequencies,600);

figure
subplot(3,1,1)
plot(rec_resampled)
xlabel("time")
ylabel("frequency")
title("Original Soundwave")
subplot(3,1,2)
plot(energy)
title("Energy of Signal(L=600)")
subplot(3,1,3)
plot(fourier_frequencies)
title("Average Fourier Frequencies")


%logical arrays of all the positions that fit the classification criteria
foreground_sounds=(energy>0.0005).*(fourier_frequencies>10);

%splitting the single soundwave into multiple smaller ones
[labels,regions]=bwlabel(foreground_sounds);

%each word/digit to be classified is one cell
foreground_sounds_cells={};
for i=1:regions
    foreground_sounds_cells{i}=rec_resampled(labels(:,1)==i);
end


%making the soundwaves into entries for the svms
entries={};
for i=1:length(foreground_sounds_cells)
    sp=abs(spectrogram(foreground_sounds_cells{i}));
    sp=imresize(sp,[250,10]);
    sp=reshape(sp,[],1);
    entries{i}=rescale(sp);
end




load("classifiers.mat");
predictions={};
for i=1:length(entries)
    pred=1;
    [~,sc]=predict(classifiers{1},entries{i}');
    min=abs(sc(1));
    for j=2:length(classifiers)
        [p,sc]=predict(classifiers{j},entries{i}');
        if(abs(sc(1)<min) && p==1)
            min=abs(sc(1));
            pred=j;
            
        end
    end
    predictions{i}=pred;
end

fprintf("Predicted Output:\n");
for i=1:length(predictions)
    fprintf("%d\t",predictions{i}-1);
end
fprintf("\n");




















