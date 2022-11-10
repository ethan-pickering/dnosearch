function [ f_likelihood ] = build_likelihood(f, aa3_grid, ww3, bbq)
%BUILD_LIKELIHOOD Summary of this function goes here
%   Currently, we build the likelihood transform using only the surrogate
%   mean, and not the surrogate uncertainty.  This makes a certain
%   importance-weighted histogram easier to build, but might lead to some
%   issues

    %[ qq3, ~] = f(aa3_grid);
    [ qq3 ] = f(aa3_grid);
    qq3 = qq3(:, 1);
    [ pp3 ] = weighted_histogram(qq3, ww3, bbq);
    
    eps0 = 1e-9;

    f_likelihood = @(q) max(interp1(bbq(1:(end-1)), pp3, q, 'linear', 'extrap'), eps0);
end

