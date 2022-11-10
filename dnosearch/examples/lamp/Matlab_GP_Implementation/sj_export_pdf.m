II = [2, 3, 4, 5, 6];
TT = [20, 30, 40, 60, 80, 100, 120];

pdf_save_path = '../../../Data/LAMP/vbm_pdf_for_fatigue/';
if ~exist(pdf_save_path, 'dir')
    mkdir(pdf_save_path);
end


p_set_list = cell(length(II), length(TT));

XXs_list = cell(length(II), length(TT));
FFs_list = cell(length(II), length(TT));

for ki = 1:length(II)
    for kt = 1:length(TT)
        %II = randperm(size(aa_list_set{ki, kt}, 1), 300);
        
        p_set_list{ki, kt} = LAMP_Protocol(a_par);
        p_set_list{ki, kt}.exp_name = sprintf('%d-%d', II(ki), TT(kt));
        p_set_list{ki, kt}.overall_norm = vbmg_norm_factor;
        p_set_list{ki, kt}.load_training_data(aa_list_set{ki, kt, 1}(:, :), ZZ_list_set{ki, kt, 1}(:, :));
        p_set_list{ki, kt}.load_testing_data(aa_list_set{ki, kt, 2}(:, :), ZZ_list_set{ki, kt, 2});
        p_set_list{ki, kt}.transform_data();
        p_set_list{ki, kt}.train_gpr();
        
        %p_set_list{ki, kt}.plot_basis();
        %p_set_list{ki, kt}.plot_surrogate(1);
        %[ p_set_list{ki, kt}.RR_res ] = draw_reconstruction_scatterplots( p_set_list{ki, kt} );
        [ XX, FF] = draw_recon_pdf( p_set_list{ki, kt} );
        XXs_list{ki, kt} = XX;
        FFs_list{ki, kt} = FF;

        xx_filename = sprintf('%s/xx_t_%dn_%d.txt', pdf_save_path, TT(kt), II(ki));
        save(xx_filename, 'XX');
        ff_filename = sprintf('%s/ff_t_%dn_%d.txt', pdf_save_path, TT(kt), II(ki));
        save(ff_filename, 'FF');        
    end
end



xx_zz = linspace(-3e9, 3e9, a_par.n_hist_bins);
PP_zz_klmc = histcounts(ZZ_nov_klmc(:)*vbmg_norm_factor, xx_zz, 'Normalization', 'pdf');
PP_zz_ssmc = histcounts(ZZ_nov_ss(:)*vbmg_norm_factor, xx_zz, 'Normalization', 'pdf');
xx_plot_1 = 1/2*(xx_zz(2:end) + xx_zz(1:end-1));



xx_filename = sprintf('%sxx_klmc.txt', pdf_save_path);
save(xx_filename, 'xx_plot_1');
ff_filename = sprintf('%sff_klmc.txt', pdf_save_path);
save(ff_filename, 'PP_zz_klmc');

xx_filename = sprintf('%sxx_ssmc.txt', pdf_save_path);
save(xx_filename, 'xx_plot_1');
ff_filename = sprintf('%sff_ssmc.txt', pdf_save_path);
save(ff_filename, 'PP_zz_ssmc');  