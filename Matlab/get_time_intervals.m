function time_diff = get_time_intervals(file)
%GET_TIME_INTERVALS Takes a .stk file and output frame intervals 
%% More data as recorded into WF_timingAnalysis
    d = dir(file);
    t = tiffread2(d.name);
    temp = [t.MM_stack];
    time_diff = diff(temp(4, :));
end