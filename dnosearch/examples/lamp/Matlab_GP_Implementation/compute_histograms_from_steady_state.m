function [ pq, pz] = compute_histograms_from_steady_state(a_par, as_par, zz, beta, V_out)
%COMPUTE_HISTOGRAMS_FROM_GPR_PROTOCOL Summary of this function goes here
%   Detailed explanation goes here

    bbq = linspace(-as_par.q_max, as_par.q_max, as_par.nqb+1);

    bbz = linspace(-7*beta, 7*beta, as_par.nqb+1);

    pz = histcounts(zz(:)*beta, bbz, 'Normalization', 'pdf');

    pq = ones(1, length(bbq)-1);

end

