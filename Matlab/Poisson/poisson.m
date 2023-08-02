function [H,P]=hostParasite(params,H0,P0,T,N)
    mu=params(1);
    alpha=params(2);
    lambda=params(3);
    b=params(4);
    
    dt=T/(N-1);
    H=zeros(N,1);
    P=zeros(N,1);
    H(1,:)=H0;
    P(1,:)=P0;
    for i=1:N-1
        Htmp=H(i,:);
        Ptmp=P(i,:);
        PHtmp=Ptmp./Htmp;

        H(i+1,:)=Htmp-(mu.*Htmp+alpha.*Ptmp).*dt;
        P(i+1,:)=Ptmp+(lambda.*Htmp./(H0)-(b+mu)-alpha.*PHtmp).*Ptmp .*dt;
    end

end