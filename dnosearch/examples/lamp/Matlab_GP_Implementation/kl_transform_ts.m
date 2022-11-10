function [ QQ ] = kl_transform_ts(a_par, ZZ, V_basis, lambda, ts_mu)
%SJ1D_CALC_BASIS_COEFFS Summary of this function goes here
%   Detailed explanation goes here

    n_exp = size(ZZ, 2);
    QQ = zeros(n_exp, a_par.n_modes );
    
    ZZ_norm = ZZ - repmat(ts_mu, [1, n_exp]);

    for k_exp = 1:n_exp
        for k_f = 1:a_par.n_modes
          QQ(k_exp, k_f) = sum(V_basis(:, k_f).*ZZ_norm(:, k_exp))./sqrt(lambda(k_f));
        end
    end

end

