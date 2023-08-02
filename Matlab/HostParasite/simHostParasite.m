function [Hout,Pout,mechs]=simHostParasite(px,x,pCal,T,tModel,H0,P0,N,M,seed)
    if ~isempty(seed)
        rng(seed)
    end
    Hout=zeros(length(tModel),M);
    Pout=zeros(length(tModel),M);
    mechs=zeros(length(tModel),M,'logical');
    % if sum(px)>1
    %     return 
    % end
    X=scaledUnif(.995,1,N,M);
    % Y=calDist(x,px,N,M);
    Y=scaledBeta(x(1),x(end),px,N,M);
    % Y=scaledNorm(x(1),x(end),px,N,M);
    % Y=scaledUnif2(x(1),x(end),px,N,M);
    parfor iM=1:M
        currT=T;
        currt=tModel;
        currH0=H0;
        currP0=P0;
        currN=N;
        iLast=0;
        currP=zeros(N,1);
        currH=zeros(N,1);
        currM=zeros(N,1,'logical');
        for k=1:N
            [H,P]=hostParasite(pCal,currH0,currP0,currT,currN);
            i=find(P./H>=0.5,1,'first');
            if isempty(i)
                break
            end
            % Pout(1+iLast:iLast+i-1,iM)=P(1:i-1);
            % Hout(1+iLast:iLast+i-1,iM)=H(1:i-1);
            % mechs(iLast+i,iM)=true;
            currP(1+iLast:iLast+i-1)=P(1:i-1);
            currH(1+iLast:iLast+i-1)=H(1:i-1);
            currM(iLast+i)=true;
            currH0=H(i).*X(iLast+i,iM);
            currP0=P(i).*Y(iLast+i,iM); % at least 5% removal
            if currP0/currH0>=0.5
                error('step size too large')
            end
            iLast=iLast+i-1;
            currT=T-currt(i);
            currt=currt(i:end);
            currN=length(currt);
        end
        [H,P]=hostParasite(pCal,currH0,currP0,currT,currN);
        % Pout(1+iLast:end,iM)=P;
        % Hout(1+iLast:end,iM)=H;
        currP(1+iLast:end)=P;
        currH(1+iLast:end)=H;
        Pout(:,iM)=currP;
        Hout(:,iM)=currH;
        mechs(:,iM)=currM;
    end
end
function r=scaledUnif(a,b,N,M)
    r=a+(b-a).*rand(N,M);
end
function r=scaledUnif2(a,b,px,N,M)
    a=a.*px(1);
    b=b.*px(2);
    r=a+(b-a).*rand(N,M);
end
% function r=scaledNorm(a,b,px,N,M)
%     r=abs(random('normal',(b-a)./4.*px(1),(b-a)./10.*px(2),[N,M]));
% end
function r=scaledBeta(a,b,px,N,M)
    r=a+(b-a).*random('beta',px(1),px(2),[N,M]);
end
function r=calDist(x,px,N,M)
    r=randsrc(N,M,[x;px]);
end