function plottracks(fname)

load(fname)

N=length(posArr);

figure()
hold on
for i=1:N
    tri=posArr(i);
    xi=tri.x;
    yi=tri.y;
    plot(xi,yi)
end
axis equal
box on