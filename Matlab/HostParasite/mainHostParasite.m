clear all; close all; fclose('all'); rng(0);
try
    num_workers = str2num(getenv('SLURM_CPUS_PER_TASK'));
    old_threads = maxNumCompThreads(num_workers);
    pool=parpool('threads'); % multithreading
    fprintf('Chosen number of workers %d, Number of active workers %d\n',num_workers,pool.NumWorkers)
catch ME
end
pool=gcp('nocreate');
if isempty(pool)
%     pool=parpool('local'); % multiprocessing
    pool=parpool('threads'); % multithreading
end
availableGPUs = gpuDeviceCount('available');
if availableGPUs > 0
    gpuDevice([]); % clears GPU
    gpuDevice(1); % selects first GPU, change for multiple with spmd
end
%%
% Input
loadDir = '../Data/WeeklyAligned';
loadName = 'LicePerFish';
loadExt = 'mat';
loadStr = [loadDir,'/',loadName,'.',loadExt];

data = load(loadStr);

lpf=data.LPF;
% lpf0=data.LPF0; %overestimates the initial lice per fish
lpf0=0.001;
% mechs=data.processedMech;
tData=data.t./53;%weeks to years, till first mechanical treatment

%cut of tData for mean time without treatments:
[~,indMaxLice]=max(lpf,[],1);

tCut=mean(tData(indMaxLice));%+std(tData(indMaxLice))/2;
[~,indCut]=max(tData>=tCut);
tData=tData(1:indCut);
lpf=lpf(1:indCut,:);

mMech=mean(cumsum(data.MechTimes,1,"omitnan"),2,"omitnan");
tMech=data.T./53;%weeks to years, till end of farms
dt=tMech(end)/(length(tMech)-1);
[~,peakMech]=max(diff(mMech,1,1)./dt);%after this time harvesting starts
mechs=cumsum(data.MechTimes,1,"omitnan");

tMech1=tMech(peakMech);
mechs1=mechs(peakMech,:);
tMech2=data.T(end)./53;
mechs2=mechs(end,:);

% select for calibration
% tMech=(tMech1+tMech2)./2;
% mechs=(mechs1+mechs2)./2;
tMech=tMech2;
mechs=mechs2;


H0=10000;
P0=lpf0.*H0;
T=3;
N=3*2*12*10;% every two weeks with 10 intermediate points
tModel=linspace(0,T,N)';


[~,tInd] = max(tModel'>=tData,[],2); 

% pCal = mu, alpha, lambda, b
[pCal,pxCal,Hsim,Psim,mechSim]=calibrate(H0,P0,T,N,tModel,tInd,lpf,mechs,tMech);
[H,P]=hostParasite(pCal,H0,P0,T,N);
disp(mean(Hsim(end,:)))

%%
tM1=find(tModel<tMech1,1,'last');
fig=newFigure();hold on;
histogram(sum(mechSim(1:tM1,:),1),'Normalization','probability')
histogram(mechs1,'Normalization','probability')
xlabel('Cumulative number of treatments')
ylabel('Probability')
legend({'Host-Parasite model','Data'},'Location','southoutside','NumColumns',4,'Interpreter','latex')
title(['Comparison of simulation and data at t=',num2str(tModel(tM1)),' years'])
exportgraphics(fig,['Figures/hostParasiteDist_',num2str(tM1),'.pdf'])

tM2=find(tModel<tMech2,1,'last');
fig=newFigure();hold on;
histogram(sum(mechSim(1:tM2,:),1),'Normalization','probability')
histogram(mechs2,'Normalization','probability')
xlabel('Cumulative number of treatments')
ylabel('Probability')
legend({'Host-Parasite model','Data'},'Location','southoutside','NumColumns',4,'Interpreter','latex')
title(['Comparison of simulation and data at t=',num2str(tModel(tM2)),' years'])
exportgraphics(fig,['Figures/hostParasiteDist_',num2str(tM2),'.pdf'])

%%
for iM=1
fig=newFigure();hold on;
xlabel('time in years')
yyaxis left
plot(tModel,Psim(:,iM)./Hsim(:,iM),'.');
ylabel('female lice per fish')
yyaxis right
plot(tModel,H,'--');
plot(tModel,Hsim(:,iM),'-');
plot(tModel,H0.*exp(-0.05.*tModel),'g--')
plot(tModel,H0.*exp(-0.1.*tModel),'k--')
ylabel('number of fish')
legend({'lice per fish','Fish without treatments','Fish with treatments','Benchmark $H_0\exp(-0.05 t)$','Benchmark $H_0\exp(-0.1 t)$'},'Location','southoutside','NumColumns',4,'Interpreter','latex')
exportgraphics(fig,['Figures/hostParasiteTra_',num2str(iM),'.pdf'])
end

