function [ zz ] = ts_transform_kl( a_par, qq, V_basis, lambda, ts_mu )
%TS_TRANSFORM_KL Summary of this function goes here
%   Detailed explanation goes here


    zz_cur = zeros(size(V_basis, 1), 1);
    
    for k_mode = 1:size(qq, 2)
        zz_cur = zz_cur + qq(:, k_mode)*V_basis(:, k_mode).*sqrt(lambda(k_mode));
    end

    zz = (zz_cur + ts_mu);
    
end

