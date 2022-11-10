function [ u ] = f_acq_lw_us_multi_out(alpha, f_input, f_likelihood, f_blackbox, k_out)
%F_ACQ_LW_US Summary of this function goes here
%   Detailed explanation goes here

    pa = f_input(alpha);
    [ mu, std ] = f_blackbox(alpha);

    mu = mu(:, k_out);
    sigma2 = std(:, k_out).^2;
    eps0 = 1e-9;

    pq = f_likelihood( mu );
    w = pa./(pq + eps0);
    u = sigma2.*w;

end

