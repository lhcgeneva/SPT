function tracksstats(fname)

load(fname)

N=length(posArr);
vlen=zeros(1,N);

for i=1:N
    vlen(i)=length(posArr(i).x);
end

figure()
hist(vlen)
xlabel('Track length (frame)')
ylabel('Number of tracks')
