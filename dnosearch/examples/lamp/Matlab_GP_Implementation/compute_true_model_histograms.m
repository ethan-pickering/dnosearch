function [true_pq, true_pz] = compute_true_model_histograms(true_model_as_par, true_model_protocol)
%COMPUTE_TRUE_MODEL_HISTOGRAMS Summary of this function goes here
%   Detailed explanation goes here

    bbq = linspace(-true_model_as_par.q_max, true_model_as_par.q_max, true_model_as_par.nqb+1);
    
    true_model_as_par.nq_mc = 1e6;
    switch true_model_as_par.true_q_pdf_rule
        case 'likelihood-transform'
            [ f_likelihood ] = build_likelihood(true_model_protocol.gpr_obj, aa3_grid, ww3, bbq);
            true_pq = f_likelihood(qq_interval);
            warning('this is bad!\n')
        case 'MC'
            aa_q = randn(true_model_as_par.nq_mc, 3);
            [ qq, ~, ~ ] = true_model_protocol.gpr_obj.sample(aa_q);
            true_pq = histcounts(qq(:, true_model_as_par.q_plot), bbq, 'Normalization', 'pdf');
    end

end

