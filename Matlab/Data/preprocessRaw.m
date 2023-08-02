clear all; close all;
%% Converts RAW data to Dayly data
% 
% data availabe at https://www.barentswatch.no/nedlasting/fishhealth/disease
% Data set 1:
% LPF = Lice per fish
% Slice 1: Adult female lice
% Output format:
% Date / Farm & Region_id & ... & Region_id     & Region_id      & ...
% Week 1       &        &     & LPF          & LPF           & ...
% Week 2       &        &     & LPF          &               & ...
% [...]
% Slice 2: Lice in moving stages
% Output format:
% Date / Farm & Region_id & ... & Region_id     & Region_id      & ...
% Week 1       &        &     & LPF          & LPF           & ...
% Week 2       &        &     & LPF          &               & ...
% [...]
% Slice 3: Stuck lice
% Output format:
% Date / Farm & Region_id & ... & Region_id     & Region_id      & ...
% Week 1       &        &     & LPF          & LPF           & ...
% Week 2       &        &     & LPF          &               & ...
% [...]
% Slice 4: Has fish: 0->no fish, 1->yes=has fish
% Output format:
% Date / Farm & Region_id & ... & Region_id     & Region_id      & ...
% Week 1       &        &     & 0          & 1           & ...
% Week 2       &        &     & 0          &               & ...
% [...]
% If value for week does not exist its NaN
%
% Output will be splitted in 3 components: data matrix, date vector, farm
%
% Data set 2:
% Treatment Type
% Output format:
% Slice 1: Mechanic removal (-1: partial farm, 1: entire farm)
% Date / Farm & Region_id & ... & Region_id     & Region_id      & ...
% Week 1       &        &     & -1          &            & ...
% Week 2       &        &     & 1          &               & ...
% [...]
% Slice 2: Rensefisk type 1
% Date / Farm & Region_id & ... & Region_id     & Region_id      & ...
% Week 1       &        &     & 1000          & 124           & ...
% Week 2       &        &     & 5236          &               & ...
% [...]
% Slice [...]
% If value for week does not exist its NaN
%
% Output will be splitted in 4 components: data array, 
% date vector (same as above), farm (same as above), treatment vector


% Input
fileDir = 'Raw';
fileExt = 'xlsx';
fileNameLPF = 'lakselus_per_fisk';
fileStrLPF = [fileDir,'/',fileNameLPF,'.',fileExt];
fileNameTT = 'tiltak_mot_lakselus';
fileStrTT = [fileDir,'/',fileNameTT,'.',fileExt];

% Output
saveDir = 'Weekly'; mkDir([cd,'\',saveDir]);
saveExt = 'mat';
saveName = 'LicePerFish';
saveStr = [saveDir,'/',saveName,'.',saveExt];

%% Process Raw Data
rawTableLPF = readtable(fileStrLPF,'VariableNamingRule','preserve');
opt = detectImportOptions(fileStrTT,'VariableNamingRule','preserve');
opt = setvartype(opt, 'Rensefisk', 'string');
opt = setvartype(opt, 'Antall','double');
rawTableTT = readtable(fileStrTT,opt);

%%
% LPF
weeksLPF = rawTableLPF{:,1};
yearsLPF = rawTableLPF{:,2};
rawhasFish = rawTableLPF{:,8};
hasFish = zeros(size(rawhasFish),'logical');
indNei=contains(rawhasFish,'Nei');%probably not without fish, so has fish
hasFish(indNei)=true;

lpf = rawTableLPF{:,5:7};
idLPF = string(rawTableLPF{:,3});
regions = string(rawTableLPF{:,13});
rawdatesLPF = string(yearsLPF)+"_"+string(num2str(weeksLPF,'%02d'));

[rawdatesLPF,sInd] = sort(rawdatesLPF);
lpf=lpf(sInd,:);
regions=regions(sInd,:);
idLPF=idLPF(sInd,:);
hasFish=hasFish(sInd,:);
lpf=[lpf,hasFish];

datesLPF = unique(rawdatesLPF);
[uidLPF,indLPF]=unique(idLPF);

% TT
weeksTT = rawTableTT{:,1};
yearsTT = rawTableTT{:,2};
meds=string(rawTableTT{:,7});
cleaner=string(rawTableTT{:,9});
cleaner(ismissing(cleaner))="";
numCleanerFish=rawTableTT{:,10};
type=string(rawTableTT{:,5});
scopeStr=rawTableTT{:,11};
treatments=type+"_"+meds+cleaner+"_"+scopeStr;

utreatments=unique(treatments);
tt=ones(size(treatments));
% indMech=contains(treatments,"mekanisk");
indCleaner=contains(treatments,"rensefisk");
% indMed=contains(treatments,"medikamentell");
% tt(indMech)=1;
tt(indCleaner)=numCleanerFish(indCleaner);
% tt(indMed)=1;
idTT = string(rawTableTT{:,3});
rawdatesTT = string(yearsTT)+"_"+string(num2str(weeksTT,'%02d'));

[rawdatesTT,sInd] = sort(rawdatesTT);
tt=tt(sInd,:);
idTT=idTT(sInd,:);
treatments=treatments(sInd,:);


datesTT = unique(rawdatesTT);
% [uidTT,indTT]=unique(idTT);



minData = 0.01.*length(datesLPF);%at least 1% of dates non-nan

uregions=regions(indLPF);
ufarms=uregions+"_"+uidLPF;

dates=unique([datesLPF;datesTT]);

% % Sanity check, slower than version below
% dataLPF = nan.*ones(length(dates)*length(ufarms),3);
% dataTT = nan.*ones(length(dates)*length(ufarms)*length(utreatments),1);
% 
% indFarmLPF=idLPF==uidLPF';
% indDateLPF=rawdatesLPF==dates';
% sz=[length(dates),length(ufarms)];
% 
% parfor i = 1:length(dates)*length(ufarms)
%     [iDate,iFarm] = ind2sub(sz,i);
%     % indDate=rawdatesLPF==dates(iDate);
%     indDataLPF = indDateLPF(:,iDate) & indFarmLPF(:,iFarm);
%     % iData = ~isnan(lpf(indDataLPF,:));
%     if sum(indDataLPF)>0
%         dataLPF(i,:)=lpf(indDataLPF,:);
%     end
% end
% dataLPF=reshape(dataLPF,sz(1),sz(2),3);
% 
% indDateTT=rawdatesTT==dates';
% indFarmTT=idTT==uidLPF';
% indTreatmentTT=treatments==utreatments';
% sz=[length(dates),length(ufarms),length(utreatments)];
% parfor i = 1:prod(sz)
%     [iDate,iFarm,iTreatment] = ind2sub(sz,i);
%     % indDate=rawdatesLPF==dates(iDate);
%     indDataTT = indDateTT(:,iDate) & indFarmTT(:,iFarm) & indTreatmentTT(:,iTreatment);
%     % iData = ~isnan(lpf(indDataLPF,:));
%     if sum(indDataTT)>0
%         dataTT(i)=sum(tt(indDataTT),1,"omitnan");
%     end
% end
% dataTT=reshape(dataTT,sz);
 
% for this to work both data sets need to be ordered the same, e.g.
% ascending dates, this version is faster than parfor above
dataLPF = nan.*ones(length(dates),length(ufarms),4);
dataTT = nan.*ones(length(dates),length(ufarms),length(utreatments));
for iF = 1:length(ufarms)
    i = uidLPF(iF);
    indFarmLPF = idLPF==i;
    currDatesLPF = rawdatesLPF(indFarmLPF);
    indDateLPF = logical(sum(dates==currDatesLPF',2));%works only if unique
    dataLPF(indDateLPF,iF,:)=lpf(indFarmLPF,:);

    indFarmTT = idTT==i;
    for iT=1:length(utreatments)
        currTreatmentInd = indFarmTT & treatments==utreatments(iT);
        currTmpTreatment = tt(currTreatmentInd);
        currDatesTT = rawdatesTT(currTreatmentInd);
        ucurrDatesTT = unique(currDatesTT);
        currTreatment=zeros(size(ucurrDatesTT));
        for iD=1:length(ucurrDatesTT)
            iU=currDatesTT==ucurrDatesTT(iD);
            currTreatment(iD)=sum(currTmpTreatment(iU));
        end
        indDateTT = logical(sum(dates==ucurrDatesTT',2));
        dataTT(indDateTT,iF,iT)=currTreatment;
    end
end

remFarms=sum(~isnan(dataLPF(:,:,1)),1)>minData;
dataLPF=dataLPF(:,remFarms,:);
dataTT=dataTT(:,remFarms,:);
farms=ufarms(remFarms);
save(saveStr,"dataLPF","dataTT","dates","farms","utreatments")

%% Auxiliary
function flag=mkDir(dir)
    if ~exist(dir,"dir")
        flag=mkdir(dir);
    else
        flag=1;
    end
end