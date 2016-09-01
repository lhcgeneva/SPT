function msdArr=getwormMSD(fname,fps)

load(fname)
N=length(posArr);
msdArr(N)=makemstruct([],[],[],[],[]);

figure()
hold on
for i=1:N
    x=posArr(i).x;
    y=posArr(i).y;
    [msd,tau,pB]=getmsd(x,y,fps);
    plot(tau,msd)
    mstruct=makemstruct(msd,tau,pB,[],[]);
    msdArr(i)=mstruct;
end
ylim([0.001 30])
set(gca,'xscale','log','yscale','log')
xlabel('Lag time \tau (s)')
ylabel('MSD (\mum^2 s^-^1)')
title([fname ' MSD'])
box on

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
    
    msd(i)=mean(delx.^2+dely.^2);
end

tau=(1:length(msd))/fps;

[pB,xfitB,yfitB]=mylogfit(tau(1:10),msd(1:10));

% figure()
% hold on 
% plot(tau,msd)
% plot(xfitB,yfitB,'r')
% set(gca,'xscale','log','yscale','log')
% pause()

function mstruct=makemstruct(msd,tau,pB,pA,p)
% make mstruct
% msd, tau are self explanatory
% p is the parameter vec:
%    msdfit=10^p(2)*tau.^p(1)+10^p(4)*tau.^p(3);
% equivalently
%    msdfit=10^b0*tau.^beta+10^a0*tau.^alpha;

mstruct=struct('msd',msd,'tau',tau,'pB',pB,'pA',pA,'p',p);

function [p,xfit,yfit]=mylogfit(xin,yin)

idx=find(yin>0);
x=xin(idx);
y=yin(idx);

lx=log10(x);
ly=log10(y);

p=polyfit(lx,ly,1);

lyfit=p(1)*lx+p(2);
yfit=10.^lyfit;
xfit=x;

% figure()
% hold on
% plot(xin,yin)
% plot(xfit,yfit)
% box on
% set(gca,'xscale','log','yscale','log')