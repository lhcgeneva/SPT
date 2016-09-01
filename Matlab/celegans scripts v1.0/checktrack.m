function checktrack(fname)

load(fname)
fac=0.1049;
tstr=fname(1:4);

N=length(posArr);
[Xmat,Ymat]=getbigmats(fname,1);

figure()
for t=1:N
    clf
    imt=imread([tstr '\' tstr '_' sprintf('%4.4d', t-1) '.tif']);
    imshow(imt)
    hold on
    
    tstart=max(1,t-9);
    xt=Xmat(:,tstart:t);
    yt=Ymat(:,tstart:t);
    
    for i=1:N
        xti=xt(i,:);
        yti=yt(i,:);
        idx=find(xti~=0);
        vx=xti(idx); vy=-yti(idx);
        plot(vx,vy,'r-')
        if ~isempty(vx)
            plot(vx(end),vy(end),'ro','LineWidth',2,'MarkerSize',12)
        end
    end
    xlim([100 300])
    ylim([100 300])
    title(t)
    pause()
end