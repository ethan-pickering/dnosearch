function [ outcode ] = sj_as_tester_toy(a_par, as_par, true_f, ...
    aa_fixed_initial, true_pq)

%
% Initialize stuff
%

f_input = @(alpha) prod(1/sqrt(2*pi)*exp(-alpha.^2/2), 2);

max_n_data = as_par.n_init + as_par.n_iter + 10;


aa_train = zeros(max_n_data, as_par.n_dim_in);
zz_train = zeros(max_n_data, a_par.n_modes);

switch as_par.initial_samples_rule
    case 'uniform'
        aa_train(1:as_par.n_init, :) = as_par.z_max*...
            (ones(as_par.n_init, as_par.n_dim_in) - 2*rand(as_par.n_init, as_par.n_dim_in));
    case 'fixed-lhs'
        aa_train(1:as_par.n_init, :) = aa_fixed_initial;
    otherwise
        warning('%s not recognized!\n', as_par.initial_samples_rule);
end

[yy] = true_f(aa_train(1:as_par.n_init, :));
zz_train(1:as_par.n_init, :) = yy;


model_list = cell(as_par.n_iter, 1);

options = optimoptions('fmincon','Display','off');

%
% Main active search loop
%

tic;

for k = 1:as_par.n_iter
    fprintf('Starting round k=%d.\n', k);
    tic;

    cur_aa_train = aa_train(1:(as_par.n_init+k-1), :);
    cur_zz_train = zz_train(1:(as_par.n_init+k-1), :);


    switch as_par.fixed_sigma_for_optimization
        case true
            cur_gpr = fitrgp(cur_aa_train, cur_zz_train, ...
                'BasisFunction', a_par.gpr_explicit_basis_class, ...
                'Sigma', 1e-5, 'ConstantSigma', true, 'SigmaLowerBound', 1e-6);

        case false
            cur_gpr = fitrgp(cur_aa_train, cur_zz_train);
    end



    %
    % Build acquisition function
    %

    fprintf('Building acquisition function with rule: %s.\n', as_par.acq_rule);

    f_blackbox = @(alpha) cur_gpr.predict(alpha);
    [ f_likelihood ] =  build_likelihood_function(as_par, f_input, f_blackbox);

    switch as_par.acq_rule
        case 'lw-kus'
            sigma_n= cur_gpr.Sigma;
            sigma2n = sigma_n.^2;
            f_acq = @(alpha) -f_acq_lw_kus(alpha, f_input, f_likelihood, f_blackbox, sigma2n);   

        case 'lw-us'
            f_acq = @(alpha) -f_acq_lw_us(alpha, f_input, f_likelihood, f_blackbox);            

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
                fprintf('Restart round %d.\n', j);
                a0 = a0_list(j, :);
                %disp(f_acq(a0));
                %[x,fval,~,~] = fmincon(f_acq, a0, A, b);

                [x,fval,~,~] = fmincon(f_acq, a0, A, b, [], [], [], [], [], ...
                    options);
                
                %[x,fval,~,~] = fmincon(f_acq, a0, [], [], [], [], -ub, ub, ...
                %    'Display', 'off');

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

    [yy] = true_f(new_aa);
            new_zz = yy;

    aa_train(as_par.n_init+k, :) = new_aa;
    zz_train(as_par.n_init+k, :) = new_zz;

    model_list{k} = cur_gpr;
end

fprintf('Main active search loop over after %0.2f seconds\n', toc);

%
% Plots!
%

fprintf('Starting plots!\n');

draw_plots_toy(a_par, as_par, model_list, true_f, true_pq);

if as_par.draw_plots
    fprintf('Drawing sample locations.\n');

    sz = 25;
    figure(1);
    clf;
    hold on
    scatter(aa_train(1:as_par.n_init, 1) , zeros([as_par.n_init, 1]), sz, 'red')
    scatter(aa_train((as_par.n_init+1):(as_par.n_init+as_par.n_iter), 1), ...
        (1:as_par.n_iter), sz, 'blue');
    xlabel('$x$', 'Interpreter', 'Latex')
    title('training samples', 'Interpreter', 'Latex')
    legend({'initial', 'active'}, 'Interpreter', 'Latex');
    set(gca, 'FontSize', 9);
    set(gcf,'units','inches','position', a_par.plot_pos);
    set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    if a_par.save_figs
        filename = sprintf('%straining_data_locs_a12', a_par.fig_path);
        print(filename,'-dpdf');
        savefig(filename);
    end

end





outcode = 1;

end