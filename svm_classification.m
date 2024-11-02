clc
clear

recs=dir("recordings\*");

%resampling frequency
s=6000;
%lowest allowed frequency
l=40;

x=zeros(3000,2500);
Y=zeros(3000,10);
for i=3:length(recs)
    [r,freq]=audioread(strcat(recs(i).folder,"\",recs(i).name));
    r=resample(r,s,freq);
    r=highpass(r,l,s);
    sp=spectrogram(r);
    sp=abs(sp);
    %rescaling the output so that all outputs are uniform
    sp=imresize(sp,[250,10]);
    x(i-2,:)=reshape(sp,[],1);
    Y(i-2,str2num(recs(i).name(1))+1)=1;
end



x=rescale(x);
%each svm number corresponds to each number being trained on this machine
svm0=fitcsvm(x,Y(:,1));
svm1=fitcsvm(x,Y(:,2));
svm2=fitcsvm(x,Y(:,3));
svm3=fitcsvm(x,Y(:,4));
svm4=fitcsvm(x,Y(:,5));
svm5=fitcsvm(x,Y(:,6));
svm6=fitcsvm(x,Y(:,7));
svm7=fitcsvm(x,Y(:,8));
svm8=fitcsvm(x,Y(:,9));
svm9=fitcsvm(x,Y(:,10));


%collecting all svms in one structure
classifiers={svm0,svm1,svm2,svm3,svm4,svm5,svm6,svm7,svm8,svm9};
%saving to file
save("classifiers.mat","classifiers");

