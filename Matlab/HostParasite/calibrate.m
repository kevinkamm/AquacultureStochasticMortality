function [pCal,pxCal,Hout,Pout,mechSim]=calibrate(H0,P0,T,N,tModel,tInd,lpf,mechs,tPeakMech)
    optLsqN=optimoptions('lsqnonlin','Display','iter-detailed');
    
    p0=log([1]);
    
    ticCal=tic;
    [pCal,res]=lsqnonlin(@(params)objective(params,H0,P0,T,N,tInd,lpf),p0,...
                                            [],[],[],[],[],[],[],optLsqN);
    ctimeCal=toc(ticCal);
    fprintf('Elapsed time: %g s.\n',ctimeCal);
    % pCal=exp(pCal);
    p0=[0.05,0.1,exp(p0),0.05]
    pCal=[0.05,0.1,exp(pCal),0.05]

    x=linspace(0.1,.9,10);

    % Discrete Dist
    % optFmin=optimoptions('fmincon','Display','iter-detailed','DiffMinChange',1e-3);
    % px=ones(size(x))./length(x);
    % % px=abs(randn(size(x)));px=px./sum(px);
    % lbpx=zeros(size(x));
    % ubpx=ones(size(x));
    % tM=find(tModel<2,1,'last');
    % M=1000;
    % ticCal=tic;
    % [pxCal,res]=fmincon(@(p)objective2(p,x,pCal,H0,P0,T,tModel,N,M,mechs,tM,1),px,...
    %                                         [],[],[],[],lbpx,ubpx,[],optFmin);
    % ctimeCal=toc(ticCal);
    % fprintf('Elapsed time: %g s.\n',ctimeCal);
    % pxCal=pxCal./sum(pxCal)
    
    % % Beta Dist
    optFmin=optimoptions('fmincon','Display','iter-detailed','DiffMinChange',1e-2);
    px=ones(2,1).*.075;
    lbpx=[];
    ubpx=[];
    tM=find(tModel<tPeakMech,1,'last');
    M=1000;
    ticCal=tic;
    [pxCal,res]=fmincon(@(p)objective2(p,x,pCal,H0,P0,T,tModel,N,M,mechs,tM,1),px,...
                                            [],[],[],[],lbpx,ubpx,[],optFmin);
    ctimeCal=toc(ticCal);
    fprintf('Elapsed time: %g s.\n',ctimeCal);
    disp(pxCal)

    [Hout,Pout,mechSim]=simHostParasite(pxCal,x,pCal,T,tModel,H0,P0,N,M,2);

    % Norm Dist
    % optFmin=optimoptions('fmincon','Display','iter-detailed','DiffMinChange',1e-3);
    % px=ones(2,1);
    % lbpx=[];
    % ubpx=[];
    % tM=find(tModel<2,1,'last');
    % M=1000;
    % ticCal=tic;
    % [pxCal,res]=fmincon(@(p)objective2(p,x,pCal,H0,P0,T,tModel,N,M,mechs,tM,1),px,...
    %                                         [],[],[],[],lbpx,ubpx,[],optFmin);
    % ctimeCal=toc(ticCal);
    % fprintf('Elapsed time: %g s.\n',ctimeCal);
    % pxCal
    % 
    % [Hout,Pout,mechSim]=simHostParasite(pxCal,x,pCal,T,tModel,H0,P0,N,M,2);
end
function res=objective(params,H0,P0,T,N,tInd,lpf)
    params=exp(params);
    [H,P]=hostParasite([0.05,0.1,params,0.05],H0,P0,T,N);
    PH=P./H;
    PH=PH(tInd).*ones(1,size(lpf,2));
    mask=~isnan(lpf);
    res=lpf(mask)-PH(mask);
end
function res=objective2(px,x,pCal,H0,P0,T,tModel,N,M,mechs,tM,seed)
    [~,~,mechSim]=simHostParasite(px,x,pCal,T,tModel,H0,P0,N,M,seed);
    
    mechsModel=sum(mechSim(1:tM,:),1);
    res=1.*(mean(mechsModel)-mean(mechs))^2 + 2.*(std(mechsModel)-std(mechs))^2;% + 1.*(skewness(mechsModel)-skewness(mechs))^2 + 1.*(kurtosis(mechsModel)-kurtosis(mechs))^2;
end