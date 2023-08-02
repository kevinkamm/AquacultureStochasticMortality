close all; clear all;
% Input
loadDir = 'Weekly';
loadName = 'LicePerFish';
loadExt = 'mat';
loadStr = [loadDir,'/',loadName,'.',loadExt];

% Input
outDir = 'WeeklyAligned';mkDir([cd,'/',outDir]);
outName = 'LicePerFish';
outExt = 'mat';
outStr = [outDir,'/',outName,'.',outExt];

data=load(loadStr);

dates=data.dates;
farms=data.farms;
lpf=data.dataLPF;
tt=data.dataTT;
treatments=data.utreatments;

weeks=computeWeeks(dates);

indMechAny=contains(treatments,"mekanisk");
indMechFull=contains(treatments,"mekanisk")&contains(treatments,"hele");
indMed=contains(treatments,"medikamentell");
indFish=contains(treatments,"rensefisk");
TT=zeros(size(tt,1),size(tt,2),4);
TT(:,:,1)=1.*(sum(tt(:,:,indMechAny),3,"omitnan")>=1);
TT(:,:,2)=1.*(sum(tt(:,:,indMed),3,"omitnan")>=1);
TT(:,:,3)=1.*(sum(tt(:,:,indFish),3,"omitnan")>=1);
TT(:,:,4)=1.*(sum(tt(:,:,indMechFull),3,"omitnan")>=1);
TT(TT==0)=nan;

region='Trøndelag';
iFarm=contains(farms,region);
farmsRegion=farms(iFarm);
lpfRegion=lpf(:,iFarm,:);
ttRegion=TT(:,iFarm,:);

% farmsRegion=farms;
% lpfRegion=lpf;
% ttRegion=TT;


fullFarmData = cell(length(farmsRegion),1);
partialFarmData = cell(length(farmsRegion),1);
for iFarm=1:length(farmsRegion)
    ind=detectSegments(lpfRegion(:,iFarm,1));
    out=cell(length(ind),6);
    outP=cell(0,7);
    for iSeg=1:length(ind)
        currSegTotal=ind{iSeg};
        
        out{iSeg,1}=currSegTotal;
        out{iSeg,2}=dates(currSegTotal);
        out{iSeg,3}=lpfRegion(currSegTotal,iFarm,1);
        out{iSeg,4}=ttRegion(currSegTotal,iFarm,:);
        out{iSeg,6}=weeks(currSegTotal);
        if sum(ttRegion(currSegTotal,iFarm,3),"omitnan")==0 %only data without cleaner fish
            iNan=find(~isnan(ttRegion(currSegTotal,iFarm,1)),1,"first");
            iNan2=find(~isnan(ttRegion(currSegTotal,iFarm,2)),1,"first"); % medical treatments after mech only
            if isempty(iNan2)
                iNan2=inf;
            end
            if ~isempty(iNan) && iNan<iNan2
                currSeg=currSegTotal(1:iNan); % only use data till first mechanical treatment
                currLPF=lpfRegion(currSeg,iFarm,1);
                if sum(currLPF>0)>10 
                    outP{end+1,1}=currSeg;
                    outP{end,2}=dates(currSeg);
                    outP{end,3}=lpfRegion(currSeg,iFarm,1);
                    outP{end,4}=ttRegion(currSeg,iFarm,:);
                    outP{end,5}=weeks(currSeg);
                    outP{end,6}=sum(ttRegion(currSegTotal,iFarm,1),"omitnan"); % any size of removal
                    % outP{end,6}=sum(ttRegion(currSegTotal,iFarm,4),"omitnan");% only entire farm
                    outP{end,7}=[weeks(currSegTotal),ttRegion(currSegTotal,iFarm,1)]; % pairs of weeks and mechanical treatments
                end
            end
        end
    end
    fullFarmData{iFarm}=out;
    partialFarmData{iFarm}=outP;
end

%% Plot LPF for a specific Farm <- may not be in partial farm
iFarm=6;

fig=newFigure();hold on;
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

out=partialFarmData{iFarm}; % could be empty
for i=1:size(out,1)
    % plot(out{i,1},out{i,3},'x') % sections till first mechanical removal
    area(out{i,1},10.*ones(size(out{i,1})),FaceColor=[144, 238, 144]./255,FaceAlpha=.1,EdgeColor="none")
    legendEntry{end+1}=out{i,2}(1)+"-"+out{i,2}(end);
end
plot(1:length(dates),.5.*ones(length(dates),1),'k-')
legendEntry{end+1}='Threshold';
plot(1:length(dates),[1.1,1.2,1.3,1.4].*squeeze(ttRegion(:,iFarm,:)),'x')
% plot(1:length(dates),squeeze(ttRegion(:,iFarm,2:end)),'.')
% legendEntry=cat(2,legendEntry,treatments');
legendEntry=cat(2,legendEntry,["Mechanical Rem.","Medicine","Cleaner Fish"]);
legend(legendEntry,'Interpreter','none')
% title("Farm "+ farmsRegion(iFarm),'Interpreter','none')
ylim([0,2])
xlabel('weeks')
ylabel('lice per fish')
exportgraphics(fig,['Figures/lpf',num2str(iFarm),'.pdf'])

%%
processedData=[];
processedTime=[];
processedMech=[];
processedTotalTimes=[];
processedMechTimes=[];

for iFarm=1:size(partialFarmData,1)
    outPeriods=partialFarmData{iFarm};
    for jPeriod=1:size(outPeriods,1)
        out=outPeriods(jPeriod,:);
        if isempty(processedData)
            processedData=fillmissing(out{3},'nearest');
            processedTime=out{1}-min(out{1});
            % [~,iMax]=max(processedData);
            % processedData=processedData(1:iMax);
            % processedTime=processedTime(1:iMax);
            processedMech=out{6};

            processedTotalTimes=out{7}(:,1)-min(out{7}(:,1));
            processedMechTimes=out{7}(:,2);
        else
            %% Adjust Lice Per Fish and Total Mechanical Treatments
            t1=processedTime;
            t2=out{1}-min(out{1});
            processedMech=[processedMech,out{6}];

            % s1=mean(processedData,2);

            s2=out{3};
            iNNan=~isnan(s2);
            s2=s2(iNNan);
            t2=t2(iNNan);

            d=0; s2a=s2;
            
            % time adjustment after
            currLen=max(length(processedTime),length(s2a));
            if length(processedTime)<length(s2a)
                processedTime=[processedTime;[length(processedTime)+1:currLen]'];
                processedData=cat(1,processedData,nan.*ones(currLen-size(processedData,1),size(processedData,2)));
            else
                s2a=cat(1,s2a,nan.*ones(currLen-length(s2a),1));
            end
            % processedData=fillmissing([processedData,s2a],'nearest',1);
            processedData=[processedData,s2a];
            % processedData(isnan(processedData))=0;
            
            %% Adjust Mechanical Treatment Times
            T1=processedTotalTimes;
            T2=out{7}(:,1)-min(out{7}(:,1)); % guarantuees start at 0 -> no align
            S1=processedMechTimes;
            S2=out{7}(:,2);
            S2a=S2; % no align
            % time adjustment after
            CurrLen=max(length(processedTotalTimes),length(S2a));
            if length(processedTotalTimes)<length(S2a)
                processedTotalTimes=[processedTotalTimes;[length(processedTotalTimes)+1:CurrLen]'];
                processedMechTimes=cat(1,processedMechTimes,nan.*ones(CurrLen-size(processedMechTimes,1),size(processedMechTimes,2)));
            else
                S2a=cat(1,S2a,nan.*ones(CurrLen-length(S2a),1));
            end
            processedMechTimes=[processedMechTimes,S2a];

        end
    end
end

% figure();hold on;
% plot(processedTime,processedData)
fig=newFigure();hold on;
[X,Y]=meshgrid(processedTime,1:size(processedData,2));
Z=processedData;
% [~,ind]=sort(max(processedData,[],1,"omitnan"));
% Z=processedData(:,ind)';
% surf(X,Y,Z);
plot3(X',Y',Z')
xlabel('weeks')
ylabel('farms')
zlabel('lice per fish')
view(3);
exportgraphics(fig,'Figures/lpfTill1stRemoval.pdf')

%% Histogram of mechanical treatments over time
fig=newFigure();hold on;
histogram(processedMech,'Normalization','probability')
Z=cumsum(processedMechTimes,1,"omitnan");
edges = 1:max(Z,[],'all');
bins  = zeros(size(processedMechTimes,1),length(edges)-1);
for i=1:size(processedMechTimes,1)
    ind=Z(i,:)>0;
    if sum(Z(i,:))>0
        [h,~] = histcounts(Z(i,ind),edges,'Normalization','probability');
        bins(i,:) = h;
    end
end
fig=newFigure();hold on;
% plot histogram bins as 3D bar-chart
b = bar3(bins');
view(60,45);
xlabel('weeks')
ylabel('cumulative number of mech. removal')
zlabel('probability')
exportgraphics(fig,'Figures/evoRemovals.pdf')

%% Save data
tCut=1;

LPF0=mean(processedData(1:tCut,:),'all');
LPF=processedData(tCut+1:end,:);
t=processedTime(tCut+1:end)-processedTime(tCut);
T=processedTotalTimes(tCut+1:end)-processedTotalTimes(tCut);
MechTimes=processedMechTimes(tCut+1:end,:);
save(outStr,"t","LPF","LPF0","processedMech","T","MechTimes");



%% Auxiliary
function Icat=detectSegments(M)
    index=find(~isnan(M));
    idx=find(diff(index)~=1);
    if isempty(idx) 
        A=[1;numel(index)-1];
    else
        A=[idx(1);diff(idx);numel(index)-idx(end)];
    end
    tmp=[1:length(M)]';
    I=mat2cell(tmp(~isnan(M)),A,1);
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
end
function weeks=computeWeeks(dates)
    weeks = double(split(dates,"_",1));
    weeks(1,:) = (weeks(1,:)  - min(weeks(1,:)))*53;
    weeks = sum(weeks,1)'; 
end

function flag=mkDir(dir)
    if ~exist(dir,"dir")
        flag=mkdir(dir);
    else
        flag=1;
    end
end