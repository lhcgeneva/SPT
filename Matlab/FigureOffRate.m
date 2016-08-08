%PAR2
BIG1 = [0.0025, 0.0026, 0.0042, 0.0042, 0.0057, 0.0045];
TH120 = [0.0076, 0.0019	0.0028, 0.0042, 0.0027	0.0046	0.0049];
kOffs = [mean(BIG1), mean(TH120)];
kOffsStds = [std(BIG1), std(TH120)];
        
figure
bar(kOffs); hold on;
errorbar(1:length(kOffs),kOffs,kOffsStds./sqrt([length(BIG1), length(TH120)]),'.');
specs = {'Big-1 P0', 'WT P0'}; 
set(gca,'xtick',[1 2], 'XTickLabel',specs, 'FontSize', 18);
ylabel('$k_{off}$ [$\frac{1}{s}$]', 'Interpreter', 'Latex', 'FontSize', 24);
