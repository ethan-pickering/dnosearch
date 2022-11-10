% %
% % Spectra rebalancing
% %
% 
% fprintf('Calculating energy ratio between J1 and J2.\n');
% 
% fixed_T = 32;
% 
% H_s = 5.5;
% T_m = 8;
% j_par = JONSWAP_Parameters();
% j_par.update_significant_wave_height( H_s );
% j_par.update_modal_period( T_m );               % close to what I was using before?
% amp_of_cosine = @(S, w, dw) sqrt(2*S(w).*dw);
% 
% WW_kl = linspace(j_par.omega_min, j_par.omega_max, j_par.n_W)';
% dW = WW_kl(2) - WW_kl(1);
% AA_kl = amp_of_cosine(j_par.S, WW_kl, dW);
% 
% T_max_kl = fixed_T;
% n_t_kl = 512;
% TT_kl = linspace(0, T_max_kl, n_t_kl);
% dt_kl = TT_kl(2) - TT_kl(1);
% 
% [ V_1, D_1 ] = calc_direct_kl_modes(AA_kl, WW_kl, TT_kl);
% 
% H_s = 13;
% T_m = 8;
% j_par = JONSWAP_Parameters();
% j_par.update_significant_wave_height( H_s );
% j_par.update_modal_period( T_m );               % close to what I was using before?
% amp_of_cosine = @(S, w, dw) sqrt(2*S(w).*dw);
% 
% WW_kl = linspace(j_par.omega_min, j_par.omega_max, j_par.n_W)';
% dW = WW_kl(2) - WW_kl(1);
% AA_kl = amp_of_cosine(j_par.S, WW_kl, dW);
% 
% T_max_kl = fixed_T;
% n_t_kl = 512;
% TT_kl = linspace(0, T_max_kl, n_t_kl);
% dt_kl = TT_kl(2) - TT_kl(1);
% 
% [ V_2, D_2 ] = calc_direct_kl_modes(AA_kl, WW_kl, TT_kl);
% 
% 
% RR = real(sqrt(D_2./D_1));
% 
% disp(RR(1:3));

%
% adjust the training data
% it's all in j1 basis
%

[ RR ] = ek_rebalance_spectra();

rr = RR(1:3)';
aa_training_adj = aa_training./rr;

%
% back to regularly scheduled GPR fitting
%



%a_par.kl_transformation_rule = 'structured-sampling';
a_par.kl_transformation_rule = 'restricted-mc';

cur_prot = LAMP_Protocol(a_par);
cur_prot.exp_name = 'openFoam--april-as';
cur_prot.overall_norm = overall_norm_factor_f;
cur_prot.load_training_data(aa_training_adj, ff_training);
cur_prot.load_testing_data(aa_training_adj(1:26, :), ff_training(:, 1:26));
cur_prot.transform_data();
cur_prot.train_gpr();

cur_prot.plot_basis();
cur_prot.plot_surrogate(1);
%[ p_foam_1.RR_res ] = draw_reconstruction_scatterplots( p_foam_1 );
draw_recon_pdf( cur_prot );
[ rmse_list_1, frac_rmse_list_1, env_rmse_list_1, env_frac_rmse_list_1 ] = ...
    compute_reconstruction_error( cur_prot );
compare_wavegroup_histograms( cur_prot );


figure(61);
clf;
plot(cur_prot.D_kl./cur_prot.D_kl(1));
set(gca, 'YScale', 'log')
xlim([1, 10])
title('Output Mode spectral Decay', 'Interpreter', 'Latex');



as_par = Active_Search_Parameters();
as_par.draw_plots = true;
as_par.opt_rule = 'as';
as_par.n_acq_restarts = 100;
as_par.save_intermediate_plots = true;

as_par.n_dim_in = 3;
as_par.q_min = -6;
as_par.q_max = 22;

as_par.z_max = 5.5;
%as_par.z_max = 7.5;

%
% Adjust this value depending on which mode we want to opt for! 
%

as_par.acq_active_output_mode = 3;

%

as_par.acq_rule = 'lw-kus';
as_par.likelihood_alg = 'kde';

options = optimoptions('fmincon','Display','off');

as_par.video_path = a_par.fig_path;
if ~exist(a_par.fig_path, 'dir')
    mkdir(a_par.fig_path);
end





fprintf('Building acquisition function with rule: %s.\n', as_par.acq_rule);

alpha_space = 'j2';
switch alpha_space
    case 'j1'
        rr = RR(1:3)';
        f_input = @(alpha) prod(1/sqrt(2*pi)*exp(-(alpha./(rr)).^2/2), 2);
    case 'j2'
        f_input = @(alpha) prod(1/sqrt(2*pi)*exp(-alpha.^2/2), 2);
end

f_blackbox = @(alpha) cur_prot.gpr_obj.predict(alpha);
[ f_likelihood ] =  build_likelihood_function(as_par, f_input, f_blackbox);


sigma_n_list= cur_prot.gpr_obj.get_sigma_n_list();
sigma2n = sigma_n_list(as_par.acq_active_output_mode).^2;
f_acq = @(alpha) -f_acq_lw_kus_multi_out(alpha, f_input, f_likelihood, ...
    f_blackbox, sigma2n, as_par.acq_active_output_mode);



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


fprintf('New acquired sample point (in J2 space):\n');
disp(new_aa);

fprintf('New acquired sample point (in J1 space):\n');
disp(new_aa.*rr);