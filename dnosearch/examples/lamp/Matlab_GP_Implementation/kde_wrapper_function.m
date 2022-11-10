function [ff] = kde_wrapper_function(xx, ww, alpha)
%KEW_WRAPPER_FUNCTION Summary of this function goes here
%   Detailed explanation goes here

    [a, ~] = ksdensity(xx, alpha, 'Weights', ww);

    %d = size(xx, 2);
    %n = size(xx, 1);
    %bw = std(xx).*(4/((d+2)*n)).^(1./(d+4));  % Silverman's rule
    %a = mvksdensity(xx,alpha,...
	%    'Bandwidth',bw, 'Weights', ww);
    
    ff = a;
end

