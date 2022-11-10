function [ pp ] = weighted_histogram(yy, ww, bb)
%WEIGHTED_HISTOGRAM Summary of this function goes here
%   Detailed explanation goes here

    ii = discretize(yy, bb);
    ii(isnan(ii)) = 1; 

    pp = zeros(length(bb)-1, 1);
    for j = 1:length(yy)
        pp(ii(j)) = pp(ii(j)) + ww(j);
    end

    pp = pp./sum(pp(:));

end

