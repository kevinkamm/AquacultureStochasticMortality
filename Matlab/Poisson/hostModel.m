function H=hostModel(t,H0,m,Nt,a,b)
    [N,M]=size(Nt);
    % H=zeros(size(Nt));
    njumps=sum(Nt(end,:),'all');
    U=ones(size(Nt));
    njumps2=0;
    dNt=diff(Nt,1,1);
    for k=1:max(dNt,[],'all')
        inddNt=cat(1,zeros(1,M,'logical'),dNt==k);
        jumps=sum(inddNt,'all');
        njumps2=njumps2+k*jumps;
        u=a+(b-a)*rand(jumps,k);
        U(inddNt)=prod(u,2);
    end
    if njumps2~=njumps
        error('Impossible')
    end
    U=cumprod(U,1);
    H=H0.*exp(-m.*t).*U;
end
% function H=hostModel(t,H0,m,Nt,a,b)
%     [N,M]=size(Nt);
%     % H=zeros(size(Nt));
%     njumps=sum(Nt(end,:),'all');
%     U=zeros(size(Nt));
%     njumps2=0;
%     dNt=diff(Nt,1,1);
%     for k=1:max(dNt,[],'all')
%         inddNt=cat(1,zeros(1,M,'logical'),dNt==k);
%         jumps=sum(inddNt,'all');
%         njumps2=njumps2+k*jumps;
%         u=(a+(b-a)*rand(jumps,k))-1;
%         U(inddNt)=prod(u,2);
%     end
%     if njumps2~=njumps
%         error('Impossible')
%     end
%     dt=t(end)/(length(t)-1);
%     U=cumsum(U,1).*dt;
%     H=H0.*exp(-m.*(t+U));
% end