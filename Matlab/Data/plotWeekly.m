close all; clear all;
% Input
loadDir = 'Weekly';
loadName = 'LicePerFish';
loadExt = 'mat';
loadStr = [loadDir,'/',loadName,'.',loadExt];

% Input
outDir = 'Figures';mkDir([cd,'/',outDir]);
% outName = 'LicePerFish';
% outExt = 'mat';
% outStr = [loadDir,'/',loadName,'.',loadExt];

data=load(loadStr);

dates=data.dates;
farms=data.farms;
lpf=data.dataLPF;
tt=data.dataTT;
treatments=data.utreatments;
% TT=double(logical(sum(tt,3,"omitnan")));
% TT=double(~isnan(tt));
indMech=contains(treatments,"mekanisk");
indMed=contains(treatments,"medikamentell");
indFish=contains(treatments,"rensefisk");
TT=zeros(size(tt,1),size(tt,2),3);
TT(:,:,1)=sum(tt(:,:,indMech),3,"omitnan")>=1;
TT(:,:,2)=sum(tt(:,:,indMed),3,"omitnan")>=1;
TT(:,:,3)=sum(tt(:,:,indFish),3,"omitnan")>=1;
TT(TT==0)=nan;

lpfFilled=fillmissing(lpf,'previous',1);
region='Trøndelag';
iFarm=contains(farms,region);
farmsRegion=farms(iFarm);
lpfRegion=lpf(:,iFarm,:);
ttRegion=TT(:,iFarm,:);
lpfRegionFilled=lpfFilled(:,iFarm);

% fig=figure();hold on;
% plot(1:length(dates),lpfRegion(:,1:5),'-')
% plot(1:length(dates),.5.*ones(length(dates),1),'k-')
% plot(1:length(dates),lpfFilled(:,iFarm))

% fig=figure();hold on;
% [X,Y]=meshgrid(1:length(dates),1:size(lpfRegion,2));
% % surf(X,Y,lpfRegion','EdgeColor','none','LineStyle','none');
% surf(X,Y,lpfRegionFilled(:,:,1)','EdgeColor','none','LineStyle','none');
% % heatmap(1:length(dates),1:size(lpfRegion,2),lpfRegionFilled')
% % scatter3(X,Y,lpfRegion');
% xlabel('dates')
% ylabel('farms')
% zlabel('lice per salmon')
% title(['Region: ',region])
% view(3);

%% Plot LPF for a specific Farm <- may not be in partial farm
iFarm=5;

fig=figure();hold on;
timeInd=1:length(dates);

% LPF data
plot(timeInd,squeeze(lpfRegion(:,iFarm,1:end-1)),'-')
legendEntry={'Female lice per fish','Moving lice per fish','Stuck lice per fish'};
% legendEntry={'Female lice per fish','Moving lice per fish','Stuck lice per fish','Has fish'};

% Periods with "has fish"
fishInd=lpfRegion(:,iFarm,end);fishInd(isnan(fishInd))=0;
fishTmp=10.*double(fishInd(:))';
fishInd=timeInd;
area(fishInd,fishTmp,FaceColor=[.1,.1,.1],FaceAlpha=.05,EdgeColor="none")
legendEntry{end+1}={'Has probably fish'};

plot(1:length(dates),.5.*ones(length(dates),1),'k-')
legendEntry{end+1}='Threshold';
plot(1:length(dates),squeeze(ttRegion(:,iFarm,1)),'*')
plot(1:length(dates),squeeze(ttRegion(:,iFarm,2:end)),'.')
% legendEntry=cat(2,legendEntry,treatments');
legendEntry=cat(2,legendEntry,["Mech","Med","Fish"]);
legend(legendEntry,'Interpreter','none')
title("Farm "+ farmsRegion(iFarm),'Interpreter','none')
ylim([0,2])

%%

fig=newFigure();hold on;
[X,Y]=meshgrid(1:length(dates),1:size(lpfRegion,2));
Xvec=X(:);
Yvec=Y(:);
mech=squeeze(ttRegion(:,:,1))';mech=mech(:);
med=squeeze(ttRegion(:,:,2))';med=med(:);
fish=squeeze(ttRegion(:,:,3))';fish=fish(:);
% s1=scatter3(Xvec,Yvec,mech,'blue');
% s2=scatter3(Xvec,Yvec,med,'yellow');
% s3=scatter3(Xvec,Yvec,fish,'red');
s1=scatter(Xvec,Yvec,36.*mech,'bx');
s2=scatter(Xvec,Yvec,36.*med,'yo');
s3=scatter(Xvec,Yvec,36.*fish,'r.');
xlabel('weeks')
ylabel('farms')
zlabel('treatment')
title(['Region: ',region])
legend('Mechanical','Medicine','Cleaner Fish');
% view(3);
exportgraphics(fig,'Figures/treatments.pdf')

%% Auxiliary
function [C,Icat]=detectSegments(M)
index=find(~isnan(M));
idx=find(diff(index)~=1);
A=[idx(1);diff(idx);numel(index)-idx(end)];
tmp=[1:length(M)]';
I=mat2cell(tmp(~isnan(M)),A,1);
C=mat2cell(M(~isnan(M)),A,1);
Icat={I{1}};
for i=2:length(I)
    Icurr=I{i};
    Iprev=Icat{end};
    if abs(Iprev(end)-Icurr(1))<10
        Icat{end}=cat(1,Iprev,Icurr);
    else
        Icat{end+1}=Icurr;
    end
end
% celldisp(C)
end
function flag=mkDir(dir)
    if ~exist(dir,"dir")
        flag=mkdir(dir);
    else
        flag=1;
    end
end