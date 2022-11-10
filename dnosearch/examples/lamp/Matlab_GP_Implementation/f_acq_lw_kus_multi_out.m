function [ u ] = f_acq_lw_kus_multi_out(alpha, f_input, f_likelihood, f_blackbox, sigma2n, k_out)
%F_ACQ_LW_US Summary of this function goes here
%   Detailed explanation goes here

    pa = f_input(alpha);
    [ mu, std ] = f_blackbox(alpha);

    mu = mu(:, k_out);
    sigma2adj = std(:, k_out).^2 - sigma2n;
    eps0 = 1e-9;

    pq = f_likelihood( mu );
    w = pa./(pq + eps0);
    u = sigma2adj.*w;

end

