function [a, D] = msd_analysis(fovn, logtable, result_cell, x_y_cell,...
                               PLOTTING, h_msd, h_aD)
%MSD_ANALYSIS Calculate and analyse mean square displacement
%% Calculate msd from tracks
msd  = cell(1, length(result_cell));
for j = 1:length(x_y_cell)
    for tau = 1:length(x_y_cell{j})-1
        temp=nan(length(x_y_cell{j})-tau, 1);
        for i =  1:length(x_y_cell{j})-tau
            temp(i) = (x_y_cell{j}(i+tau, 1)-x_y_cell{j}(i, 1))^2+...
                          (x_y_cell{j}(i+tau,2)-x_y_cell{j}(i, 2))^2;
        end
        msd{j}(tau) = nanmean(temp);
    end
    msd{j} = [0,msd{j}];
end

% Fit MSD = 4Dt^a, plot D(a)
% % % frame_interval = logtable.frame_interval(fovn);
% % % numFramesFit = 10;
% % % D = zeros(1,length(msd));
% % % a = zeros(1,length(msd));
% % % for i = 1:length(msd)
% % %     lx = log10((frame_interval):(frame_interval):(10*frame_interval))';
% % %     ly = log10(msd{i}(2:numFramesFit+1))';
% % %     p = polyfit(lx, ly, 1);
% % %     D(i) = 10^(p(2))/4;
% % %     a(i) = p(1);
% % % end

frame_interval = logtable.frame_interval(fovn);
startFrameFit = 1; % Which lag time to set as first frame for fitting
numFramesFit = 10;
D = zeros(1,length(msd));
a = zeros(1,length(msd));
for i = 1:length(msd)
    lx = log10((startFrameFit*frame_interval):...
               (startFrameFit*frame_interval):...
               (numFramesFit*startFrameFit*frame_interval))';
%     ly = log10(msd{i}(2:numFramesFit+1))';
    ly = log10(msd{i}(1+startFrameFit:startFrameFit:...
                      startFrameFit*numFramesFit+1))';
    p = polyfit(lx, ly, 1);
    D(i) = 10^(p(2))/4;
    a(i) = p(1);
end

if PLOTTING
    % Plot all msd over time
%         figure; hold on;
    axes(h_msd); hold on;
    for i = 1:length(result_cell)
        plot(msd{i}, 'b');
    end
    set(gca,'XScale','log');
    set(gca,'YScale','log');
    xlabel('\tau [s]', 'FontSize', 16);
    ylabel('$$\overline{x^2} [\mu m^2]$$', 'FontSize', 16,...
           'interpreter','latex');
    title('MSD vs lag time', 'FontSize', 14);
    % Plot Diffusion coefficient versus anomalous diffusion exponent
%         figure();
    axes(h_aD); hold on;
    scatter(a, D);
    title('Diffusion characterization \langle x^2 \rangle = 4Dt^\alpha',...
          'FontSize', 14);
    xlabel('\alpha', 'FontSize', 16);
    ylabel('D [$\frac{\mu m^2}{s}$]', 'FontSize', 16, 'Interpreter',...
           'latex');
%     axis([-0.5 2 0 /0.06]);
end