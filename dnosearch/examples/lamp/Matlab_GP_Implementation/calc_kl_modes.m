function [ V_out , D_out, zz_mu ] = calc_kl_modes(zz)
%CALC_KL_MODES Summary of this function goes here
%   Detailed explanation goes here



    sig_vmbg = std(zz(:));   % important for normalization!
    mu_vbmg = mean(zz(:));
    
    %ZZ_klmc_vbmg_normed = zz/sig_vmbg;
    ZZ_klmc_vbmg_normed = zz;


    n_exp = size(ZZ_klmc_vbmg_normed, 2);
    zz_mu = mean(ZZ_klmc_vbmg_normed, 2);
    zz_res = ZZ_klmc_vbmg_normed - repmat(zz_mu, [1, n_exp]);

    RR = (zz_res)*(zz_res') / n_exp;

    [V,D] = eig(RR, 'vector');

    V_out = fliplr(V);
    D_out = flipud(D);

end

