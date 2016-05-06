function x = fit_offRate(fitTimes, fitData)
kOffStart = 0.001;
kPhStart = 0.01;
NssStart = 1;
x = fminsearch(@(x) objFunc(x, fitTimes, fitData), [kOffStart, NssStart, kPhStart]);
[t,y] = ode45(@(t,y) x(1)*x(2) - (x(1)+x(3))*y, [min(fitTimes) max(fitTimes)], x(2));
plot(t,y,'-', 'LineWidth', 3)

function f = objFunc(x, fitTimes, fitData)
    [t,y] = ode45(@(t,y) x(1)*x(2) - (x(1)+x(3))*y, [min(fitTimes) max(fitTimes)], x(2));    
    y_interp = interp1(t, y, fitTimes);
    f = sum((y_interp-fitData).^2);
        