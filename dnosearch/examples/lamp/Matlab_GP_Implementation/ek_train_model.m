%ii = [1:6, 8:size(aa_training, 1)];
%ii = [1:6, 8:20];
ii = [1:size(aa_training, 1)];

%
% adjust the training data
% it's all in j1 basis
%

target_spectrum = 'j2';

switch target_spectrum 
    case 'j1'
        SS = ss_ff;

        aa_training_adj = aa_training;
        aa_testing_adj = aa_testing;

    case 'j2'
        [ RR ] = ek_rebalance_spectra();
        
        rr = RR(1:3)';
        aa_training_adj = aa_training./rr;
        aa_testing_adj = aa_testing./rr;

        SS = ss_table2(:, f_col_index);
end


p_foam_1 = LAMP_Protocol(a_par);
p_foam_1.exp_name = 'openFoam--april';
p_foam_1.overall_norm = overall_norm_factor_f;
p_foam_1.load_training_data(aa_training_adj(ii, :), ff_training(:, ii));
p_foam_1.load_testing_data(aa_testing_adj, ff_testing);
p_foam_1.transform_data();
p_foam_1.train_gpr();

p_foam_1.plot_basis();
p_foam_1.plot_surrogate(1);
%[ p_foam_1.RR_res ] = draw_reconstruction_scatterplots( p_foam_1 );
[ XX, FF ] = draw_recon_pdf( p_foam_1 );
%[ rmse_list_1, frac_rmse_list_1, env_rmse_list_1, env_frac_rmse_list_1 ] = ...
%    compute_reconstruction_error( p_foam_1 );
%compare_wavegroup_histograms( p_foam_1 );


switch f_col_index
    case 2
        f_name = '$F_x$';
        filename_tag = 'fx';
    case 3
        f_name = '$F_y$';
        filename_tag = 'fy';
    case 4
        f_name = '$F_z$';
        filename_tag = 'fz';
    case 5
        f_name = '$F_x^P$';
        filename_tag = 'fpx';
    case 6
        f_name = '$F_y^P$';
        filename_tag = 'fpy';
    case 7
        f_name = '$F_z^P$';
        filename_tag = 'fpz';
    case 8
        f_name = '$F_x^\nu$';
        filename_tag = 'fnx';
    case 9
        f_name = '$F_y^\nu$';
        filename_tag = 'fny';
    case 10
        f_name = '$F_z^\nu$';
        filename_tag = 'fnz';
end





figure(101);
clf;
hold on
plot(XX, FF, 'LineWidth', 3);
%[pf_ss, bb_ss] = histcounts(ss_ff, 'Normalization', 'pdf');
[pf_ss, bb_ss] = histcounts(SS, 'Normalization', 'pdf');
xx_ss = 1/2*(bb_ss(1:end-1) + bb_ss(2:end));
maxP = max(pf_ss(:));
plot(xx_ss, pf_ss, 'LineWidth', 3);
set(gca, 'YScale', 'log');
%ylim([1e-10, 1e-6])
ylim([5*maxP*10^-6, maxP*5])
xlabel(f_name, 'Interpreter', 'Latex');
ylabel('$f_F(f)$', 'Interpreter', 'Latex');
title(sprintf('Comparison of steady state and GPR %s', f_name), 'Interpreter', 'Latex');
legend({'GPR', 'MC'}, 'Location', 'best');
grid on
set(gca, 'FontSize', 9);
%set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
filename = sprintf('%spdf_hist_%s', a_par.fig_path, filename_tag);
print(filename,'-dpdf');
savefig(filename);






figure(201);
clf
plot(p_foam_1.D_kl./p_foam_1.D_kl(1), 'LineWidth', 3);
xlim([0, 10])
set(gca, 'YScale', 'log');
set(gca, 'FontSize', 9);
%set(gcf,'units','inches','position', a_par.plot_pos);
title(sprintf('PCA eigenvalues -- %s', f_name), 'Interpreter', 'Latex');
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
filename = sprintf('%sfx_output_pca_eignevalues_%s', a_par.fig_path, filename_tag);
print(filename,'-dpdf');
savefig(filename);



figure(202);
clf;
hold on
for k = 1:4
    subplot(2, 2, k);
    hold on
    plot(TT, p_foam_1.V_kl(:, 2*k-1), 'LineWidth', 1.5);
    plot(TT, p_foam_1.V_kl(:, 2*k), 'LineWidth', 1.5);
    xlim([0, max(TT)])
    set(gca, 'FontSize', 9);
    %set(gcf,'units','inches','position', a_par.plot_pos);
    title(sprintf('modes %d \\& %d', 2*k-1, 2*k), 'Interpreter', 'Latex');
end
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
filename = sprintf('%sfx_output_pca_eignemodes_%s', a_par.fig_path, filename_tag);
print(filename,'-dpdf');
savefig(filename);




% 
% TT = linspace(0, 32, 512);
% [zz, mu, ss] = p_foam_1.sample(aa_training(end, :));
% figure(103);
% clf;
% hold on
% plot(TT, ff_training(:, end), 'LineWidth', 3, 'Color', 'Red');
% plot(TT, zz, 'LineWidth', 3, 'Color', 'Blue');
% plot(TT, mu+ss, 'LineWidth', 3, 'Color', 'Cyan', 'LineStyle', ':');
% plot(TT, mu-ss, 'LineWidth', 3, 'Color', 'Cyan', 'LineStyle', ':');
% legend('openFOAM', 'surrogate', 'Interpreter', 'Latex');
% title(sprintf('Comparison of steady state and GPR %s', f_name), 'Interpreter', 'Latex');
% set(gca, 'FontSize', 9);
% %set(gcf,'units','inches','position', a_par.plot_pos);
% set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
% filename = sprintf('%sts_recovery_unblind%s', a_par.fig_path, filename_tag);
% print(filename,'-dpdf');
% savefig(filename);
% 
% figure(104);
% clf;
% hold on
% plot(TT, mu, 'LineWidth', 3, 'Color', 'Red');
% plot(TT, ss, 'LineWidth', 3, 'Color', 'Blue');

% figure(102);
% clf;
% hold on
% plot(XX, FF, 'LineWidth', 3);
% %[pf_ss, xx_ss] = ksdensity(ss_ff);
% [pf_ss, xx_ss] = ksdensity(ss_table2(:, f_col_index));
% plot(xx_ss, pf_ss, 'LineWidth', 3);
% set(gca, 'YScale', 'log');
% %ylim([1e-10, 1e-6])
% ylim([5*maxP*10^-6, maxP*5])
% xlabel('$F_x$', 'Interpreter', 'Latex');
% ylabel('$f_F(f)$', 'Interpreter', 'Latex');
% title(sprintf('Comparison of steady state and GPR %s', f_name), 'Interpreter', 'Latex');
% legend({'GPR', 'MC'})
% grid on
% set(gca, 'FontSize', 9);
% %set(gcf,'units','inches','position', a_par.plot_pos);
% set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
% filename = sprintf('%spdf_kde_%s', a_par.fig_path, filename_tag);
% print(filename,'-dpdf');
% savefig(filename);



% 
% 
% p_foam_2 = LAMP_Protocol(a_par);
% p_foam_2.exp_name = 'openFoam--jan--recover-coeff';
% p_foam_2.overall_norm = overall_norm_factor_f;
% p_foam_2.load_training_data(aa_training_recovered, ff_training);
% p_foam_2.load_testing_data(aa_testing_recovered, ff_testing);
% p_foam_2.transform_data();
% p_foam_2.train_gpr();
% 
% p_foam_2.plot_basis();
% p_foam_2.plot_surrogate(1);
% %[ p_foam_2.RR_res ] = draw_reconstruction_scatterplots( p_foam_2 );
% %draw_recon_pdf( p_foam_2 );
% [ rmse_list_2, frac_rmse_list_2, env_rmse_list_2, env_frac_rmse_list_2 ] = ...
%     compute_reconstruction_error( p_foam_2 );
% compare_wavegroup_histograms( p_foam_2 );
% 
% 
% 
% p_foam_3 = LAMP_Protocol(a_par);
% p_foam_3.exp_name = 'openFoam--jan--coeff2wavegroup';
% p_foam_3.overall_norm = overall_norm_factor_z;
% p_foam_3.load_training_data(aa_training_recovered, zz_training);
% p_foam_3.load_testing_data(aa_testing_recovered, zz_testing);
% p_foam_3.transform_data();
% p_foam_3.train_gpr();
% 
% p_foam_3.plot_basis();
% p_foam_3.plot_surrogate(1);
% %[ p_foam_3.RR_res ] = draw_reconstruction_scatterplots( p_foam_3 );
% %draw_recon_pdf( p_foam_3 );
% [ rmse_list_3, frac_rmse_list_3, env_rmse_list_3, env_frac_rmse_list_3 ] = ...
%     compute_reconstruction_error( p_foam_3 );
% compare_wavegroup_histograms( p_foam_3 );
% 
% 


% 
% 
% 
% a1 = 1;
% a2 = 2;
% q1 = 1;
% 
% figure(101);
% clf;
% scatter3(p_foam_1.aa_train(:, a1), p_foam_1.aa_train(:, a2), p_foam_1.qq_train(:, q1));
% title(sprintf('$q_%d$ surrogate, resampled training points -- %s', q1, p_foam_1.exp_name), 'Interpreter', 'Latex');
% xlabel(sprintf('$\\alpha_%d$', a1), 'Interpreter', 'Latex')
% ylabel(sprintf('$\\alpha_%d$', a2), 'Interpreter', 'Latex')
% zlabel(sprintf('$q_%d$', q1), 'Interpreter', 'Latex')
% 
% 
% [ ~, qq_hat, ~] = p_foam_1.gpr_obj.sample(p_foam_1.aa_train);
% 
% figure(102);
% clf;
% scatter3(p_foam_1.aa_train(:, a1), p_foam_1.aa_train(:, a2), qq_hat(:, q1));
% title(sprintf('$q_%d$ surrogate, resampled testing points -- %s', q1, p_foam_1.exp_name), 'Interpreter', 'Latex');
% xlabel(sprintf('$\\alpha_%d$', a1), 'Interpreter', 'Latex')
% ylabel(sprintf('$\\alpha_%d$', a2), 'Interpreter', 'Latex')
% zlabel(sprintf('$q_%d$', q1), 'Interpreter', 'Latex')
% 
% 
% 
% a1_list = [1, 1];
% a2_list = [2, 3];
% q1_list = [1, 2, 3];
% 
% ng = 65;
% a_grid = linspace(-4, 4, ng);
% [aa1, aa2] = meshgrid(a_grid, a_grid);
% 
% LL = linspace(-4, 4, 10);
% 
% for ka = 1:2
% figure(110 + ka);
% clf;
% 
% for kq = 1:length(q1_list)
% 
%     a1 = a1_list(ka);
%     a2 = a2_list(ka);
%     q1 = q1_list(kq);
% 
%     aa_grid =  zeros(ng^2, 3);
%     aa_grid(:, [a1, a2]) = [aa1(:), aa2(:)];
%     [ ~, qq_grid, ~] = p_foam_1.gpr_obj.sample(aa_grid);
% 
%     zz = reshape(qq_grid(:, q1), [ng, ng]);
%     figure(103);
%     clf;
%     mesh(a_grid, a_grid, zz);
%     title(sprintf('$q_%d$ surrogate, resampled grid -- %s', q1, p_foam_1.exp_name), 'Interpreter', 'Latex');
%     xlabel(sprintf('$\\alpha_%d$', a1), 'Interpreter', 'Latex')
%     ylabel(sprintf('$\\alpha_%d$', a2), 'Interpreter', 'Latex')
%     zlabel(sprintf('$q_%d$', q1), 'Interpreter', 'Latex')
% 
%     figure(104);
%     clf;
%     pcolor(a_grid, a_grid, zz);
%     shading flat
%     title(sprintf('$q_%d$ surrogate, resampled grid -- %s', q1, p_foam_1.exp_name), 'Interpreter', 'Latex');
%     xlabel(sprintf('$\\alpha_%d$', a1), 'Interpreter', 'Latex')
%     ylabel(sprintf('$\\alpha_%d$', a2), 'Interpreter', 'Latex')
%     %zlabel(sprintf('$q_%d$', q1), 'Interpreter', 'Latex')
%     colorbar();
% 
%     figure(110 + ka);
%     subplot(2, 2, kq)
%     hold on
%     pcolor(a_grid, a_grid, zz);
%     caxis([-3.5, 3.5])
%     contour(a_grid, a_grid, zz, LL, 'Color', 'Black');
%     shading flat
%     title(sprintf('$q_%d$', q1), 'Interpreter', 'Latex');
%     xlabel(sprintf('$\\alpha_%d$', a1), 'Interpreter', 'Latex')
%     ylabel(sprintf('$\\alpha_%d$', a2), 'Interpreter', 'Latex')
% end
% 
%     figure(110 + ka);
%     set(gca, 'FontSize', 9);
% 
%     set(gcf,'units','inches','position', a_par.plot_pos);
%     set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
% 
% 
%     filename = sprintf('%ssurrogate_model_%s_%d', a_par.fig_path, p_foam_1.exp_name, ka);
%     print(filename,'-dpdf');
%     savefig(filename);
% 
% end
% 
% 
% 



n_recoveries = 4;

XX4_raw = cell(n_recoveries, 1);
XX4_cooked = cell(n_recoveries, 1);

V_out = p_foam_1.gpr_obj.V_out;
lambda = p_foam_1.gpr_obj.D_out; %rescale by KL eigenweights
beta = p_foam_1.gpr_obj.overall_norm_factor; % final rescaling
ts_mu = p_foam_1.gpr_obj.ts_mu;

for k = 1:n_recoveries
    qq = p_foam_1.gpr_obj.predict(aa_testing(k, :));
    XX4_cooked{k} = ts_transform_kl( a_par, qq, V_out, lambda, ts_mu )*...
        p_foam_1.gpr_obj.overall_norm_factor;

    XX4_raw{k} = ff_testing(:, k)*p_foam_1.gpr_obj.overall_norm_factor;
end



n_reps = 1000;
XX4_cooked_mean = cell(n_recoveries, 1);
XX4_cooked_std = cell(n_recoveries, 1);

for k = 1:n_recoveries
    zz_cooked  = zeros(n_reps, length(ts_mu));
    for j = 1:n_reps
        qq = p_foam_1.gpr_obj.sample(aa_testing(k, :));
        zz_cooked(j, :) = ts_transform_kl( a_par, qq, V_out, lambda, ts_mu )*...
            p_foam_1.gpr_obj.overall_norm_factor;
    end

    XX4_cooked_mean{k} = mean(zz_cooked, 1);
    XX4_cooked_std{k} = std(zz_cooked, 0, 1);
end


zstar = 3;
lw = 1;


TT_plot = linspace(0, 32, length(XX4_raw{1}));

figure(131);
clf;
for k = 1:n_recoveries
    subplot(2, 2, k);
    hold on
    plot(TT_plot, XX4_raw{k}, 'LineWidth', lw)
    plot(TT_plot, XX4_cooked{k}, 'LineWidth', lw)
    xlabel('$t$', 'Interpreter', 'Latex');
    ylabel('$F_x^{\mbox{tot}}$', 'Interpreter', 'Latex');
    if (k == 1)
        legend({'openFOAM', 'reconstruct'}, 'Interpreter', 'Latex', 'Location', 'Northwest');
    end
    title(sprintf('openFOAM vs GP surrogate -- wave %d', 400 + k), 'Interpreter', 'Latex')

    xlim([0, max(TT_plot(:))]);
    %ylim([-zstar, zstar]);
    z_max = 5e6;
    ylim([-z_max, z_max]);
end

%subplot(2, 2, 2);
%legend({'LAMP', 'recon'}, 'Interpreter', 'Latex');

set(gca, 'FontSize', 9);

set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.full_paper_pos, 'PaperSize', a_par.full_paper_size);

if a_par.save_figs
    filename = sprintf('%sreconstructed_time_series_openFOAM-%s', a_par.fig_path, p_foam_1.exp_name);
    print(filename,'-dpdf');
    savefig(filename);
end



figure(132);
clf;
for k = 1:n_recoveries
    subplot(2, 2, k);
    hold on

    x2 = [TT_plot, fliplr(TT_plot) ];
    inBetween = [ XX4_cooked_mean{k} + XX4_cooked_std{k}, fliplr(XX4_cooked_mean{k} - XX4_cooked_std{k}) ];
    fill(x2', inBetween', 'cyan');
    h2 = plot(TT_plot, XX4_cooked_mean{k}, 'LineWidth', lw, 'Color', 'blue');

    h1 = plot(TT_plot, XX4_raw{k}, 'LineWidth', lw, 'Color', 'Red');

    if (k == 1)
        legend([h1, h2], {'openFOAM', 'reconstruct'}, 'Interpreter', 'Latex', 'Location', 'Northwest');
    end
    %title(sprintf('openFOAM vs GP surrogate -- wave %d', 400 + k), 'Interpreter', 'Latex')
    title(sprintf('wave %d', 400 + k), 'Interpreter', 'Latex')

    xlim([0, max(TT_plot(:))]);
    %ylim([-zstar, zstar]);
    z_max = 5e6;
    ylim([-z_max, z_max]);
    aa = gca;
    set(gca, 'YTickLabel', aa.YTickLabel())
    
end

set(gca, 'FontSize', 9);

set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', [0, 0, 4.5, 3.5], 'PaperSize', [4.5, 3.5]);

if a_par.save_figs
    filename = sprintf('%sreconstructed_time_series_openFOAM-%s_spread', a_par.fig_path, p_foam_1.exp_name);
    print(filename,'-dpdf');
    savefig(filename);
end