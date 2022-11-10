% cur_prot = LAMP_Protocol(a_par);
% cur_prot.exp_name = sprintf('two-60');
% cur_prot.overall_norm = vbmg_norm_factor;
% cur_prot.load_training_data(aa_list_set{1, 4, 1}(:, :), ZZ_list_set{1, 4, 1}(:, :));
% cur_prot.load_testing_data(aa_list_set{1, 4, 2}(:, :), ZZ_list_set{1, 4, 2});
% cur_prot.transform_data();
% cur_prot.train_gpr();
% 
% cur_prot.plot_basis();
% cur_prot.plot_surrogate(1);
% [ cur_prot.RR_res ] = draw_reconstruction_scatterplots( cur_prot );
% draw_recon_pdf( cur_prot );
% 
% save_path = '../../../Data/GPR/Two-60-no-basis';
% if ~exist(save_path, 'dir')
%     mkdir(save_path);
% end
% cur_prot.save_to_text(save_path);





cur_prot = LAMP_Protocol(a_par);
cur_prot.exp_name = sprintf('three-60');
cur_prot.overall_norm = vbmg_norm_factor;
cur_prot.load_training_data(aa_list_mar_bonus{1, 1}(:, :), ZZ_list_mar_bonus{1, 1}(:, :));
%cur_prot.load_testing_data(aa_list_set{2, 4, 2}(:, :), ZZ_list_set{2, 4, 2});
cur_prot.load_testing_data(aa_list_mar_bonus{1, 1}(:, :), ZZ_list_mar_bonus{1, 1}(:, :));
cur_prot.transform_data();
cur_prot.train_gpr();

%cur_prot.plot_basis();
%cur_prot.plot_surrogate(1);
%[ cur_prot.RR_res ] = draw_reconstruction_scatterplots( cur_prot );
%draw_recon_pdf( cur_prot );

save_path = '../../../Data/GPR/Mar-three-60';
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
cur_prot.save_to_text(save_path);