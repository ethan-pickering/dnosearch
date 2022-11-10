% aa_train = aa_list_set{3, 3, 1};
% aa_test = aa_list_set{3, 3, 2};
% 
% zz_train = ZZ_list_set{3, 3, 1};
% zz_test = ZZ_list_set{3, 3, 2};
% 
% base_name = 'four-40';

aa_train = aa_list_set{2, 4, 1};
aa_test = aa_list_set{2, 4, 2};

zz_train = ZZ_list_set{2, 4, 1};
zz_test = ZZ_list_set{2, 4, 2};

base_name = 'three-60';


p_scalar = LAMP_Protocol(a_par);
p_scalar.exp_name = sprintf('%s-scalar', base_name);
p_scalar.overall_norm = vbmg_norm_factor;
p_scalar.load_training_data(aa_train, zz_train);
p_scalar.load_testing_data(aa_test, zz_test);
p_scalar.transform_data();
p_scalar.train_gpr();

p_scalar.plot_basis();
p_scalar.plot_surrogate(1);
[ p_scalar.RR_res ] = draw_reconstruction_scatterplots( p_scalar );
draw_recon_pdf( p_scalar );

p_scalar.gpr_obj.g_fit_list{1}.KernelInformation.KernelParameters
p_scalar.gpr_obj.g_fit_list{1}.Sigma
p_scalar.gpr_obj.g_fit_list{1}.Beta




p_vektor = LAMP_Protocol(a_par);
p_vektor.exp_name = sprintf('%s-vektor', base_name);
p_vektor.overall_norm = vbmg_norm_factor;
p_vektor.load_training_data(aa_train, zz_train);
p_vektor.load_testing_data(aa_test, zz_test);
p_vektor.transform_data();
p_vektor.vector_pair_list = [1, 2];
p_vektor.rho_list = [p_scalar.RR_res(1, 2)];
p_vektor.train_gpr();

p_vektor.plot_basis();
p_vektor.plot_surrogate(1);
[ p_vektor.RR_res ] = draw_reconstruction_scatterplots( p_vektor );
a_par.n_hist_resample = 2000;
draw_recon_pdf( p_vektor );

p_vektor.gpr_obj.vector_gpr_list{1}.g_fit.KernelInformation.KernelParameters
p_vektor.gpr_obj.vector_gpr_list{1}.rho
p_vektor.gpr_obj.vector_gpr_list{1}.sigma0

% 
% qq = zeros(100, 2);
% for k = 1:100
%     [ qq_sample, qq_pred_mu, qq_pred_cov ] = p_vektor.gpr_obj.sample(aa_test(1, :));
%     qq(k, :) = qq_sample(1:2);
% end
% 
% [ qq_pred_mu, qq_pred_cov ] = p_vektor.gpr_obj.vector_gpr_list{1}.predict(aa_test(1, :));
% 
% figure(11);
% clf;
% scatter(qq(:, 1), qq(:, 2));




% [rot_mat, ~] = eig(p_vektor.RR_res);
% 
% 
% p_vektor_rot = LAMP_Protocol(a_par);
% p_vektor_rot.exp_name = 'four-40-rot';
% p_vektor_rot.overall_norm = vbmg_norm_factor;
% p_vektor_rot.load_training_data(aa_list_set{3, 3, 1}, ZZ_list_set{3, 3, 1});
% p_vektor_rot.load_testing_data(aa_list_set{3, 3, 2}, ZZ_list_set{3, 3, 2});
% p_vektor_rot.rot_mat = rot_mat;
% p_vektor_rot.transform_data();
% p_vektor_rot.train_gpr();
% 
% p_vektor_rot.plot_basis();
% p_vektor_rot.plot_surrogate(1);
% [ p_vektor_rot.RR_res ] = draw_reconstruction_scatterplots( p_vektor_rot );
% draw_recon_pdf( p_vektor_rot );
% 
% p_vektor_rot.gpr_obj.g_fit_list{1}.KernelInformation.KernelParameters
% p_vektor_rot.gpr_obj.g_fit_list{1}.Sigma
% p_vektor_rot.gpr_obj.g_fit_list{1}.Beta


% 
% p_vektor2 = LAMP_Protocol(a_par);
% p_vektor2.exp_name = 'four-80';
% p_vektor2.overall_norm = vbmg_norm_factor;
% p_vektor2.load_training_data(aa_list_set{3, 5, 1}, ZZ_list_set{3, 5, 1});
% p_vektor2.load_testing_data(aa_list_set{3, 5, 2}, ZZ_list_set{3, 5, 2});
% p_vektor2.transform_data();
% p_vektor2.train_gpr();
% 
% p_vektor2.plot_basis();
% p_vektor2.plot_surrogate(1);
% [ p_vektor2.RR_res ] = draw_reconstruction_scatterplots( p_vektor2 );
% draw_recon_pdf( p_vektor2 );
% 
% p_vektor2.gpr_obj.g_fit_list{1}.KernelInformation.KernelParameters
% p_vektor2.gpr_obj.g_fit_list{1}.Sigma
% p_vektor2.gpr_obj.g_fit_list{1}.Beta