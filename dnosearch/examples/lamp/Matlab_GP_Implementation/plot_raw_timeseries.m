function [ outcode ] = plot_raw_timeseries( a_par,  zz, tt)
%PLOT_RAW_TIMESERIES Summary of this function goes here
%   Detailed explanation goes here


    if (nargin == 2)
        tt = linspace(1, size(zz, 1), size(zz, 1));
    end

    figure(31);
    clf;
    hold on
    %k_off = 12*25;
    k_off = 0;
    for k = k_off + 1*(1:20)
        plot(tt, zz(:, k));
    end
    title('Sample time series');
    
    figure(32);
    clf;
    hold on
    plot(tt, var(zz, [], 2));
    title('time series variance');
    
%     figure(33);
%     clf;
%     hold on
%     plot(tt, mean(zz, 2));
%     title('time series mean');
    
    
    outcode = 1;
end

