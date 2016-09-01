function plotmsddist(fname)

load(['msd' fname])

N=length(msdArr);
vAlpha=zeros(1,N);
vD=zeros(1,N);

for i=1:N
    msdi=msdArr(i);
    pB=msdi.pB;
    vAlpha(i)=pB(1);
    vD(i)=(10^pB(2))/4;
end

figure()
scatter(vAlpha,vD)
box on
xlabel('Exponent \alpha')
ylabel('D (um^2/s)')