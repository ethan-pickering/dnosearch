function [ RR ] = calc_spectra_transform(H1, H2, T_m, fixed_T)
%EK_REBALANCE_SPECTRA Summary of this function goes here
%   Detailed explanation goes here

%
% Spectra rebalancing
%
    
    fprintf('Calculating energy ratio between J1 and J2.\n');
    
    %fixed_T = 32;
    
    H_s = H1;
    j_par = JONSWAP_Parameters();
    j_par.update_significant_wave_height( H_s );
    j_par.update_modal_period( T_m );               % close to what I was using before?
    amp_of_cosine = @(S, w, dw) sqrt(2*S(w).*dw);
    
    WW_kl = linspace(j_par.omega_min, j_par.omega_max, j_par.n_W)';
    dW = WW_kl(2) - WW_kl(1);
    AA_kl = amp_of_cosine(j_par.S, WW_kl, dW);
    
    T_max_kl = fixed_T;
    n_t_kl = 512;
    TT_kl = linspace(0, T_max_kl, n_t_kl);
    dt_kl = TT_kl(2) - TT_kl(1);
    
    [ V_1, D_1 ] = calc_direct_kl_modes(AA_kl, WW_kl, TT_kl);
    
    H_s = H2;
    j_par = JONSWAP_Parameters();
    j_par.update_significant_wave_height( H_s );
    j_par.update_modal_period( T_m );               % close to what I was using before?
    amp_of_cosine = @(S, w, dw) sqrt(2*S(w).*dw);
    
    WW_kl = linspace(j_par.omega_min, j_par.omega_max, j_par.n_W)';
    dW = WW_kl(2) - WW_kl(1);
    AA_kl = amp_of_cosine(j_par.S, WW_kl, dW);
    
    T_max_kl = fixed_T;
    n_t_kl = 512;
    TT_kl = linspace(0, T_max_kl, n_t_kl);
    dt_kl = TT_kl(2) - TT_kl(1);
    
    [ V_2, D_2 ] = calc_direct_kl_modes(AA_kl, WW_kl, TT_kl);
    
    
    RR = real(sqrt(D_2./D_1));
    
    %disp(RR(1:3));

end

