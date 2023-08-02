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

% lpf=data.LPF;
% lpf0=data.LPF0;
% lpf0=0.0001;
mechs=data.MechTimes;
tData=data.T./53;%weeks to years
dt=tData(end)/(size(tData,1)-1);
m=mean(cumsum(mechs,1,"omitnan"),2,"omitnan");
dm=diff(m,1,1)./dt;


H0=10000;
T=3;
N=(3*2*12)*10;% every two weeks with 10 intermediate points
tModel=linspace(0,T,N)';

dmOld=dm;
[dmMax,iMax]=max(dm,[],'all');
iMax=iMax-5;dmMax=dm(iMax);
dm(iMax+1:end)=mean(dm(iMax));
scale=m(end)./(m(iMax)+dmMax*(tData(end)-tData(iMax)));
dm=dm.*scale;

% p=A,K,C,Q,B,nu. f(t)=A+(K-A)/((C+Q e^(-B t))^(1/nu)), A=0
richardsCurve = @(t,p) 0+(p(1)-0)./((p(2)+p(3).*exp(-p(4).*(t))).^(1/p(5)));
p=1.*ones(5,1);
% [p,err]=fminsearch(@(p) mean((cumsum(richardsCurve(tData(1:end),p),1).*(tData(end)/(length(tData)-1))-m).^2),p,optimset(MaxIter=1e4,MaxFunEvals=1e5));
% [p,err]=fmincon(@(p) mean((cumsum(richardsCurve(tData(1:end),p),1).*(tData(end)/(length(tData)-1))-m).^2),p);
[p,err]=lsqnonlin(@(p) (cumsum(richardsCurve(tData(1:end),p),1).*(tData(end)/(length(tData)-1))-m),p,[],[],[],[],[],[],[],optimoptions("lsqnonlin",'MaxFunctionEvaluations',1e6,'MaxIterations',1e6));

% lambda=griddedInterpolant(tData(2:end),dm,'linear');
lambda=@(t)richardsCurve(t,p);

dt=T/(N-1);
Lambda=cumtrapz(lambda(tModel).*dt);
Lambda=griddedInterpolant(tModel,Lambda,'linear');
%%
fig=newFigure();
tlo=tiledlayout(1,2);
nexttile; hold on;
plot(tData(2:end),dmOld)
plot(tData(2:end),dm)
plot(tModel,richardsCurve(tModel,p))
xlabel('time in years')
title('Intensity')
nexttile; hold on;
xlabel('time in years')
title('Cumulative intensity')
plot(tData(2:end),cumsum(dmOld.*(tData(end)/(length(tData)-2))))
plot(tData(2:end),cumsum(dm.*(tData(end)/(length(tData)-2))))
plot(tModel,cumsum(richardsCurve(tModel,p)).*dt)
lgd=legend({'Raw Data','Scaled Data','Logistic Fit'},'NumColumns',4,'Interpreter','latex');
lgd.Layout.Tile = 'south';
exportgraphics(fig,['Figures/PoissonIntensity','.pdf'])

%%
m=[0;cumsum(dm.*dt,1)];

tic;
Nt=inhomPoisson(@(x)Lambda(x),2^12,tModel);
toc;

dm=lambda(tData(2:end));
csvwrite('tData.csv',tData')
csvwrite('dm.csv',dm')
% csvwrite('pwT.csv',pwTModel')

%%
tInd1=iMax+5;
tInd2=length(tData);
tM1=find(tModel<=tData(tInd1),1,'last');
tM2=find(tModel<=tData(tInd2),1,'last');
%%
figure();
subplot(2,1,1);hold on;
plot(tModel(1:tM1),mean(Nt(1:tM1,:),2),'r-');
plot(tData,m,'b--');
subplot(2,1,2);hold on;
plot(tModel(1:tM1),var(Nt(1:tM1,:),0,2),'m-');
plot(tData,var(cumsum(mechs,1,"omitnan"),0,2,"omitnan"),'c--');

H=hostModel(tModel,H0,0.05,Nt,0.995,1.0);
disp(mean(H(end,:)))
iM=1:10;
fig=newFigure();hold on;
plots=plot(tModel,H(:,iM));
plots=plots(1);
plots(end+1)=plot(tModel,H0.*exp(-0.1*tModel),'k-');
% plot(tModel,mean(H,2),'r-')
ylabel('number of fish')
legend(plots,{'Fish with treatments, 10 trajectories','Benchmark $H_0\exp(-0.1 t)$'},'Location','southoutside','NumColumns',4,'Interpreter','latex')
exportgraphics(fig,['Figures/PoissonTra','.pdf'])
%%

fig=newFigure();hold on;
histogram(Nt(tM1,:),'Normalization','probability')
histogram(sum(mechs(1:tInd1,:),1,"omitnan"),'Normalization','probability')
xlabel('Cumulative number of treatments')
ylabel('Probability')
legend({'Poisson model','Data'},'Location','southoutside','NumColumns',4,'Interpreter','latex')
title(['Comparison of simulation and data at t=',num2str(tModel(tM1)),' years'])
exportgraphics(fig,['Figures/PoissonDist_',num2str(tM1),'.pdf'])

fig=newFigure();hold on;
histogram(Nt(tM2,:),'Normalization','probability')
histogram(sum(mechs(1:tInd2,:),1,"omitnan"),'Normalization','probability')
xlabel('Cumulative number of treatments')
ylabel('Probability')
legend({'Poisson model','Data'},'Location','southoutside','NumColumns',4,'Interpreter','latex')
title(['Comparison of simulation and data at t=',num2str(tModel(tM2)),' years'])
exportgraphics(fig,['Figures/PoissonDist_',num2str(tM2),'.pdf'])