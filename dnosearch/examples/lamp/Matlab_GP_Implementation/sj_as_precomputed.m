function [ outcode ] = sj_as_precomputed(a_par, as_par, data_aa, data_zz, true_pz )

    %
    % Initialize stuff
    %
    
    f_input = @(alpha) prod(1/sqrt(2*pi)*exp(-alpha.^2/2), 2);
    
    max_n_data = as_par.n_init + as_par.n_iter + 10;
    %a_par.kl_transformation_rule = 'no-transform';
    
    
    aa_train = zeros(max_n_data, as_par.n_dim_in);
    zz_train = zeros(max_n_data, size(data_zz, 2));
    
    switch as_par.initial_samples_rule
        case 'random-sample'
            [~, ii] = datasample(1:size(data_aa, 1), as_par.n_init);

            aa_train(1:as_par.n_init, :) = data_aa(ii, :);
            zz_train(1:as_par.n_init, :) = data_zz(ii, :);

        otherwise
            warning('%s not recognized!\n', as_par.initial_samples_rule);
    end
        
    protocol_list = cell(as_par.n_iter, 1);
    
    options = optimoptions('fmincon','Display','off');
    
    %
    % Main active search loop
    %
    
    tic;
    
    for k = 1:as_par.n_iter
        fprintf('Starting round k=%d.\n', k);
        
        switch as_par.mode_choice_rule
            case 'fixed-mode'
                cur_opt_mode = 1;
            case 'round-robin'
                cur_opt_mode = mod(k-1, as_par.n_rr_rondel_size)+1;
            otherwise
                warning('%s not recognized!\n', as_par.mode_choice_rule);
        end
        cur_aa_train = aa_train(1:(as_par.n_init+k-1), :);
        cur_zz_train = zz_train(1:(as_par.n_init+k-1), :);
    
        cur_model_protocol = LAMP_Protocol(a_par);
        cur_model_protocol.exp_name = sprintf('six-60');
        cur_model_protocol.overall_norm = as_par.overall_norm_factor;
        cur_model_protocol.load_training_data(cur_aa_train, cur_zz_train');
        cur_model_protocol.load_testing_data(data_aa, data_zz');
        cur_model_protocol.transform_data();
        cur_model_protocol.train_gpr();
    
%         cur_model_protocol.gpr_obj.V_out = true_model_protocol.gpr_obj.V_out;
%         cur_model_protocol.gpr_obj.D_out = true_model_protocol.gpr_obj.D_out;
%         cur_model_protocol.gpr_obj.overall_norm_factor = ...
%             true_model_protocol.gpr_obj.overall_norm_factor; % final rescaling
%         cur_model_protocol.gpr_obj.ts_mu = true_model_protocol.gpr_obj.ts_mu;
    
        %
        % Build acquisition function
        %
    
        fprintf('Building acquisition function with rule: %s.\n', as_par.acq_rule);
        
        f_blackbox = @(alpha) cur_model_protocol.gpr_obj.predict(alpha);
        [ f_likelihood ] =  build_likelihood_function(as_par, f_input, f_blackbox, ...
            cur_opt_mode);
    
        switch as_par.acq_rule
            case 'lw-kus'
                sigma_n_list= cur_model_protocol.gpr_obj.get_sigma_n_list();
                sigma2n = sigma_n_list(cur_opt_mode).^2;
                f_acq = @(alpha) -f_acq_lw_kus_multi_out(alpha, f_input, ...
                    f_likelihood, f_blackbox, sigma2n, cur_opt_mode);   
    
            case 'lw-us'
                f_acq = @(alpha) -f_acq_lw_us_multi_out(alpha, f_input, ...
                    f_likelihood, f_blackbox, cur_opt_mode); 
                
            case 'uniform'
                f_acq = @(alpha) 1;
    
            otherwise
                warning('%s not recognized\n', as_par.acq_rule);
        end
    
        %
        % Choose next point
        %
    
        fprintf('Evaluating acquisition function to choose next point.\n');
    
        switch as_par.opt_rule
            case 'uniform'
                %new_aa = as_par.z_max*(ones(1, as_par.n_dim_in) - 2*rand(1, as_par.n_dim_in));
                new_ii = ceil(size(data_aa, 1)*rand(1,1));
                new_aa = data_aa(new_ii, :);
                new_zz = data_zz(new_ii, :);

            case 'as'
                uu = f_acq(data_aa);
                [~, new_ii] = min(uu);
                new_aa = data_aa(new_ii, :);
                new_zz = data_zz(new_ii, :);

            otherwise
                warning('%s not recognized\n', as_par.acq_rule);
        end
    
        %
        % Evaluate next point
        %
    
    
        aa_train(as_par.n_init+k, :) = new_aa;
        zz_train(as_par.n_init+k, :) = new_zz;
    
        protocol_list{k} = cur_model_protocol;
    end
    
    fprintf('Main active search loop over after %0.2f seconds\n', toc);
    
    %
    % Plots!
    %
    
    fprintf('Starting plots!\n');
    
    %draw_true_model_plots(a_par, as_par, true_f_mean);
    %draw_movie_plots(a_par, as_par, protocol_list, true_f_mean, true_pq);
    draw_sample_point_plots(a_par, as_par, aa_train);

    f_fake = @(a) 1;
    [ err_struct  ] = draw_error_plots( a_par, as_par, protocol_list, ...
        f_fake, 0, true_pz);
    
    err_filename = sprintf('%serr_struct.mat', a_par.fig_path);
    save(err_filename, 'err_struct', '-mat');


    gamma =  8.9224e-10;
    figure(101);
    clf;
    plot(1:length(err_struct.pz_log_mae_trunc_list), gamma*err_struct.pz_log_mae_trunc_list);
    set(gca, 'YScale', 'log');
    
    
    
    outcode = 1;

end