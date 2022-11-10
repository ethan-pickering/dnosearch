function [ maxf, minf, xx ] = compute_extrema_pdf( protocol )
%COMPUTE_EXTREMA_PDF Summary of this function goes here
%   Detailed explanation goes here



    a_par = protocol.a_par;
    gpr_surrogate = protocol.gpr_obj;

    %
    % sample from the surrogates
    %

    n_samples = a_par.n_hist_resample;
    %n_samples = 20;

    xx_test = randn(n_samples, gpr_surrogate.n_inputs);



    switch a_par.gpr_resampling_strat 
        case 'normally-distributed'
            [ yprd, ysd ] = gpr_surrogate.predict(xx_test);

            bb = randn(size(ysd));

            yy_guess_nd = yprd + bb.*ysd;
    
        case 'vector-resample'
            [ qq_sample, qq_pred_mu, ~ ] = gpr_surrogate.sample(xx_test);
            
            yy_guess_nd = qq_sample;
            
        case 'list-only'
            [ qq_sample ] = gpr_surrogate.sample(xx_test);
            
            yy_guess_nd = qq_sample;
            
    end



    V_out = gpr_surrogate.V_out;
    lambda = gpr_surrogate.D_out; %rescale by KL eigenweights
    beta = gpr_surrogate.overall_norm_factor; % final rescaling
    ts_mu = gpr_surrogate.ts_mu;
    
    zz_list_nd = zeros(n_samples, size(V_out, 1));

    local_max_list = cell(n_samples, 1);
    local_min_list = cell(n_samples, 1);
    
    for k_sample = 1:n_samples
        zz_list_nd(k_sample, :) = ts_transform_kl( a_par, yy_guess_nd(k_sample, :), V_out, lambda, ts_mu );
        local_max_list{k_sample} = findpeaks(zz_list_nd(k_sample, :), 'MinPeakDistance', 15 );
        local_min_list{k_sample} = -findpeaks(-zz_list_nd(k_sample, :), 'MinPeakDistance', 15);
    end

    local_max_total = cell2mat(local_max_list');
    local_min_total = cell2mat(local_min_list');

    xx_zz = linspace(-10*beta, 10*beta, a_par.n_hist_bins);
    xx = 1/2*(xx_zz(2:end) + xx_zz(1:end-1));

    maxf = histcounts(local_max_total*beta, xx_zz, 'Normalization', 'pdf');
    minf = histcounts(local_min_total*beta, xx_zz, 'Normalization', 'pdf');


    
end

