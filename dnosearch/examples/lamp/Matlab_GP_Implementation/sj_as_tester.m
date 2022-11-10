function [ outcode ] = sj_as_tester(a_par, as_par, true_f_sample, true_f_mean, ...
    aa_fixed_initial, zz_fixed_initial, true_pq, true_pz, ...
    true_testing_aa, true_testing_zz, true_model_protocol)

    %
    % Initialize stuff
    %
    
    f_input = @(alpha) prod(1/sqrt(2*pi)*exp(-alpha.^2/2), 2);
    
    max_n_data = as_par.n_init + as_par.n_iter + 10;
    a_par.kl_transformation_rule = 'no-transform';
    
    
    aa_train = zeros(max_n_data, as_par.n_dim_in);
    zz_train = zeros(max_n_data, a_par.n_modes);
    
    switch as_par.initial_samples_rule
        case 'uniform'
            aa_train(1:as_par.n_init, :) = as_par.z_max*...
                (ones(as_par.n_init, as_par.n_dim_in) - 2*rand(as_par.n_init, as_par.n_dim_in));
    
            [yy] = true_f_sample(aa_train(1:as_par.n_init, :));
            zz_train(1:as_par.n_init, :) = yy;
        case 'fixed-lhs'
            aa_train(1:as_par.n_init, :) = aa_fixed_initial;
            zz_train(1:as_par.n_init, :) = zz_fixed_initial;
        otherwise
            warning('%s not recognized!\n', as_par.initial_samples_rule);
    end
    
    [yy] = true_f_sample(aa_train(1:as_par.n_init, :));
    zz_train(1:as_par.n_init, :) = yy;
    
    %a3_grid = linspace(-as_par.z_max, as_par.z_max, as_par.n_grid_likelihood);
    %[aa13, aa23, aa33] = meshgrid(a3_grid, a3_grid, a3_grid);
    %aa3_grid = [aa13(:), aa23(:), aa33(:)];
    %ww3 = f_input(aa3_grid);
    
    %bbq = linspace(-as_par.q_max, as_par.q_max, as_par.nqb+1);
    
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
                cur_opt_mode = mod(k-1, 6)+1;
            otherwise
                warning('%s not recognized!\n', as_par.mode_choice_rule);
        end
        cur_aa_train = aa_train(1:(as_par.n_init+k-1), :);
        cur_zz_train = zz_train(1:(as_par.n_init+k-1), :);
    
        cur_model_protocol = LAMP_Protocol(a_par);
        cur_model_protocol.exp_name = sprintf('three-60');
        cur_model_protocol.overall_norm = 1;
        cur_model_protocol.load_training_data(cur_aa_train, cur_zz_train');
        cur_model_protocol.load_testing_data(true_testing_aa, true_testing_zz');
        cur_model_protocol.transform_data();
        cur_model_protocol.train_gpr();
    
        cur_model_protocol.gpr_obj.V_out = true_model_protocol.gpr_obj.V_out;
        cur_model_protocol.gpr_obj.D_out = true_model_protocol.gpr_obj.D_out;
        cur_model_protocol.gpr_obj.overall_norm_factor = ...
            true_model_protocol.gpr_obj.overall_norm_factor; % final rescaling
        cur_model_protocol.gpr_obj.ts_mu = true_model_protocol.gpr_obj.ts_mu;
    
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
                new_aa = as_par.z_max*(ones(1, as_par.n_dim_in) - 2*rand(1, as_par.n_dim_in));
    
            case 'as'
                A = [eye(as_par.n_dim_in); -eye(as_par.n_dim_in)];
                b = [as_par.z_max*ones(as_par.n_dim_in, 1); as_par.z_max*ones(as_par.n_dim_in, 1)];
                %ub = as_par.z_max*ones(1, 3);
    
                a_opt_list = zeros(as_par.n_acq_restarts, as_par.n_dim_in);
                f_opt_list = zeros(as_par.n_acq_restarts, 1);
                a0_list = as_par.z_max*(ones(as_par.n_acq_restarts, as_par.n_dim_in) - ...
                    2*lhsdesign(as_par.n_acq_restarts, as_par.n_dim_in));
    
                for j = 1:as_par.n_acq_restarts
                    fprintf('%d-', j);
                    a0 = a0_list(j, :);
                    
                    [x,fval,~,~] = fmincon(f_acq, a0, A, b, [], [], [], [], [], ...
                        options);
    
                    a_opt_list(j, :) = x;
                    f_opt_list(j) = fval;
                end
    
                [~, i ] = min(f_opt_list);
    
                new_aa = a_opt_list(i, :);
                %fprintf('Next point at alpha = (%0.2f, 0.2f, 0.2f).\n', new_aa(1), ...
                %    new_aa(2), new_aa(3))
    
            otherwise
                warning('%s not recognized\n', as_par.acq_rule);
        end
    
        %
        % Evaluate next point
        %
    
        [yy] = true_f_sample(new_aa);
        new_zz = yy;
    
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
    [ err_struct  ] = draw_error_plots( a_par, as_par, protocol_list, ...
        true_f_mean, true_pq, true_pz);
    
    
    
    
    
    outcode = 1;

end