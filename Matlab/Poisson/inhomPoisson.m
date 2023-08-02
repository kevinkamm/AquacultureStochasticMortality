% function Nt=inhomPoisson(lambda,pwT,t,M)
% %% Ross Simulations Ed. 4 p. 85 https://shop.elsevier.com/books/simulation/ross/978-0-12-598063-0
% 
% %% Critical for this algorithm: we assume piecewise constant intensity
% T=t(end);
% pwT=pwT(:);%[t_0,t_1,t_2,...]-> intevals [t_0,t_1],[t_1,t_2],...
% t0=pwT(1:end-1);
% t1=pwT(2:end);
% mesh=linspace(0,1,100);
% tgrid=t0+(t1-t0).*mesh;
% lambdaM=max(lambda(tgrid),[],2); %majorant of lambda in piecewise intervals
% 
% Nt=zeros(length(t),M); % the results will be approximated in this time grid
% for wi=1:M
%     ti=0;J=1;
%     s=[];
%     while ti<T
%         X=-1/lambdaM(J)*log(rand());
%         if ti+X>pwT(J+1)
%             if J>=length(pwT)-1
%                 break
%             end
%             X=(X-pwT(J+1)+ti)*lambdaM(J)/lambdaM(J+1);
%             ti=pwT(J+1);
%             J=J+1;
%         else
%             ti=ti+X;
%             U=rand();
%             if U<=lambda(ti)/lambdaM(J)
%                 s(end+1)=ti;
%             end
%         end
%     end
%     for i=1:length(s)
%         currJ=find(t>=s(i),1,'first');
%         Nt(currJ,wi)=Nt(currJ,wi)+1;%if time grid too small multiple jumps
%     end
% end
% Nt=cumsum(Nt,1);
% 
% end
function Nt=inhomPoisson(Lambda,M,t)
    T=t(end);
    Nt=zeros(length(t),M);
    LT=Lambda(T);
    r = poissrnd(LT,M,1);
    njumps=sum(r);
    u=rand(njumps,1);

    a=zeros(size(u));
    b=T.*ones(size(u));
    % Ftinv=zeros(size(u));
    for j=1:10 %10 is enough
        tmp=Lambda((a+b)./2)./LT;
        less=tmp<=u;
        greater=~less;
        a(less)=(a(less)+b(less))./2;
        b(greater)=(a(greater)+b(greater))./2;
    end
    Ftinv=(a+b)./2;

    lastJumps=0;
    [~,tI]=max(Ftinv<t',[],2);%gives first index, basically "find" for each row
    for wi=1:M
        currJumps=tI(lastJumps+1:lastJumps+r(wi));
        for ji=1:length(currJumps)
            ti=currJumps(ji);
        % for ti=currJumps %somehow slower and less accurate
            Nt(ti,wi)=Nt(ti,wi)+1;
        end
        lastJumps=lastJumps+r(wi);
    end
    Nt=cumsum(Nt,1);
end