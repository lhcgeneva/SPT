function portC2Arr(fname,fac)

load(fname)

clear posArr
N=length(c);
tri=struct('x',[],'y',[],'fstart',0,'I',[]);
posArr(N)=tri;
for i=1:N
    tri=struct('x',c{i}(:,1)*fac,'y',-c{i}(:,2)*fac,'fstart',c{i}(1,6),...
        'I',c{i}(:,3));
    posArr(i)=tri;
end

save(fname,'c','posArr')


