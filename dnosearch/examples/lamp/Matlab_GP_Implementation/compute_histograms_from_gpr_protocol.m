function [ pq, pz] = compute_histograms_from_gpr_protocol(a_par, as_par, cur_model_protocol)
%COMPUTE_HISTOGRAMS_FROM_GPR_PROTOCOL Summary of this function goes here
%   Detailed explanation goes here

    bbq = linspace(-as_par.q_max, as_par.q_max, as_par.nqb+1);

    beta = cur_model_protocol.gpr_obj.overall_norm_factor;
    %zstar = 7;
    zstar = 10;
    bbz = linspace(-zstar*beta, zstar*beta, as_par.nqb+1);
    

    switch as_par.q_pdf_rule
        case 'likelihood-transform'
            %pq_list{k} = zz;
            warning('This (%s) is bad these days!\n', as_par.q_pdf_rule)
        case 'MC'
            aa_q = randn(as_par.nq_mc, as_par.n_dim_in);
            [ qq, yprd, ysd ] = cur_model_protocol.gpr_obj.sample(aa_q);
            pq = histcounts(qq(:, as_par.q_plot), bbq, ...
                'Normalization', 'pdf');
    end

    %
    % full pdf
    %

    bb = randn(size(ysd));
    yy_guess_nd = yprd + bb.*ysd;

    V_out = cur_model_protocol.gpr_obj.V_out;
    lambda = cur_model_protocol.gpr_obj.D_out; %rescale by KL eigenweights
    beta = cur_model_protocol.gpr_obj.overall_norm_factor; % final rescaling
    ts_mu = cur_model_protocol.gpr_obj.ts_mu;

    zz_list_nd = zeros(as_par.nq_mc, length(ts_mu));

    for k_sample = 1:as_par.nq_mc
        zz_list_nd(k_sample, :) = ts_transform_kl( a_par, yy_guess_nd(k_sample, :), V_out, lambda, ts_mu );
    end

    %zz_list_nd = zz_list_nd*beta;

    pz = histcounts(zz_list_nd(:), bbz, 'Normalization', 'pdf');
end

