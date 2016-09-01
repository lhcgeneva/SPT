function [msd,tau,pB]=getmsd(x,y,fps)
% if track is 200 frames long, msd(tau) only has 100 elements (look at
% nOpti)

nOpti=length(x)-10;
lenx=length(x);
msd=zeros(1,nOpti);

for i=1:nOpti
    x1=x(1:lenx-i);
    x2=x(1+i:lenx);
    delx=x2-x1;
    
    y1=y(1:lenx-i);
    y2=y(1+i:lenx);
    dely=y2-y1;
    
    msd(i)=nanmean(delx.^2+dely.^2);
end