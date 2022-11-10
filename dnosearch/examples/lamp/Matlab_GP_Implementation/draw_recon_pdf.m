function [ XX, FF ] = draw_recon_pdf( protocol )
%DRAW_RECON_PDF Summary of this function goes here
%   Detailed explanation goes here

    a_par = protocol.a_par;
    gpr_surrogate = protocol.gpr_obj;
    yy_true = protocol.qq_test;
    zz_true = protocol.zz_test;

    %
    % sample from the surrogates
    %

    n_samples = a_par.n_hist_resample;
    
    xx_test = randn(n_samples, gpr_surrogate.n_inputs);
    
    
    switch a_par.gpr_resampling_strat 
        case 'normally-distributed'
            [ yprd, ysd ] = gpr_surrogate.predict(xx_test);

            %bb = randn(n_samples, a_par.n_modes);
            bb = randn(size(ysd));

            yy_guess_mo = yprd;
            yy_guess_nd = yprd + bb.*ysd;
    
        case 'vector-resample'
            [ qq_sample, qq_pred_mu, ~ ] = gpr_surrogate.sample(xx_test);
            
            %yy_guess_mo = qq_pred_mu;
            yy_guess_nd = qq_sample;
            
        case 'list-only'
            [ qq_sample ] = gpr_surrogate.sample(xx_test);
            
            yy_guess_nd = qq_sample;
            
    end
    
    
    

    %
    % histogram the mode coefficients
    %

    x_max = 4.5;

    xx_coeff_surrogate = linspace(-x_max, x_max, a_par.n_hist_bins);

    %PP_coeff_mo = zeros(a_par.n_modes, a_par.n_hist_bins-1);
    PP_coeff_nd = zeros(protocol.n_output_modes, a_par.n_hist_bins-1);

    for k_mode = 1:protocol.n_output_modes
        %PP_coeff_mo(k_mode, :) = histcounts(yy_guess_mo(:, k_mode), xx_coeff_surrogate, ...
        %    'Normalization', 'pdf');
        PP_coeff_nd(k_mode, :) = histcounts(yy_guess_nd(:, k_mode), xx_coeff_surrogate, ...
            'Normalization', 'pdf');
    end

    %
    % histogram the 'true' coefficients
    %

    n_bins_true = 65;
    xx_coeff_true = linspace(-x_max, x_max, n_bins_true);
    PP_coeff_true = zeros(a_par.n_modes, n_bins_true-1);

    for k_mode = 1:protocol.n_output_modes
        PP_coeff_true(k_mode, :) = histcounts(yy_true(:, k_mode), xx_coeff_true, ...
            'Normalization', 'pdf');
    end

    %
    % plot the mode coefficient histograms!
    %

    n_plot = min(12, protocol.n_output_modes);

    names = cell(n_plot, 1);
    for k_mode = 1:n_plot
        names{k_mode} = sprintf('n=%d', k_mode);
    end
    
    xx_plot_1 = 1/2*(xx_coeff_surrogate(2:end) + xx_coeff_surrogate(1:end-1));
    xx_plot_2 = 1/2*(xx_coeff_true(2:end) + xx_coeff_true(1:end-1));

    CC = parula(n_plot);
    
    figure(8);
    clf;
    hold on
    hh = zeros(n_plot, 1);
    for k_mode = 1:n_plot
        hh(k_mode) = plot(xx_plot_1,  PP_coeff_nd(k_mode, :), 'Color', CC(k_mode, :));
        plot(xx_plot_2,  PP_coeff_true(k_mode, :), 'Color', CC(k_mode, :), 'LineStyle', ':');
    end
    xlabel('KL coefficient value -- $q_i$', 'Interpreter', 'Latex');
    ylabel('$f_Q(q)$', 'Interpreter', 'Latex');
    title(sprintf('resampled coefficient pdf -- %s', gpr_surrogate.exp_name), 'Interpreter', 'Latex');
    set(gca, 'YScale', 'log');
    legend(hh, names, 'Interpreter', 'Latex');
    grid on
    set(gca, 'FontSize', 9);

    set(gcf,'units','inches','position', a_par.plot_pos);
    set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);

    if a_par.save_figs
        filename = sprintf('%s%s-recon-modes-pdf-combined', a_par.fig_path, gpr_surrogate.exp_name);
        print(filename,'-dpdf');
        savefig(filename);
    end

%     figure(9);
%     clf;
%     for k_mode = 1:n_plot
%         subplot(2, 3, k_mode)
%         hold on
%         plot(xx_plot_1,  PP_coeff_mo(k_mode, :), 'Color', 'Blue', 'LineStyle', '-');
%         plot(xx_plot_1,  PP_coeff_nd(k_mode, :), 'Color', 'Cyan', 'LineStyle', '--');
%         plot(xx_plot_2,  PP_coeff_true(k_mode, :), 'Color', 'Red', 'LineStyle', ':');
%         xlabel(sprintf('KL coefficient value -- $q_%d$', k_mode), 'Interpreter', 'Latex');
%         ylabel(sprintf('$f_{Q_%d}(q_%d)$', k_mode, k_mode), 'Interpreter', 'Latex');
%         title(sprintf('mode $%d$ -- %s', k_mode, gpr_surrogate.exp_name), 'Interpreter', 'Latex');
%         set(gca, 'YScale', 'log');
%         legend('gpr-mean', 'gpr-spread', 'mc', 'Interpreter', 'Latex', 'Location', 'South');
%         %xlim([-1.2, 1.2]);
%         grid on
% 
%         set(gca, 'FontSize', 9);
% 
%     end
% 
%     set(gcf,'units','inches','position', a_par.plot_pos);
%     set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.full_paper_pos, 'PaperSize', a_par.full_paper_size);
% 
%     if a_par.save_figs
%         filename = sprintf('%s%s-recon-modes-pdf', a_par.fig_path, gpr_surrogate.exp_name);
%         print(filename,'-dpdf');
%         savefig(filename);
%     end
    
    
    
    figure(10);
    clf;
    for k_mode = 1:n_plot
        subplot(3, 4, k_mode)
        hold on
        plot(xx_plot_1,  PP_coeff_nd(k_mode, :), 'Color', 'Blue', 'LineStyle', '-', 'LineWidth', 2);
        plot(xx_plot_2,  PP_coeff_true(k_mode, :), 'Color', 'Black', 'LineStyle', '-', 'LineWidth', 2);
        %xlabel(sprintf('KL coefficient value -- $q_%d$', k_mode), 'Interpreter', 'Latex');
        xlabel(sprintf('$q_{%d}$', k_mode), 'Interpreter', 'Latex');
        ylabel(sprintf('$f_{Q_%d}(q_{%d})$', k_mode, k_mode), 'Interpreter', 'Latex');
        %title(sprintf('mode $%d$ -- %s', k_mode, gpr_surrogate.exp_name), 'Interpreter', 'Latex');
        %title(sprintf('$n_{\\mbox{out}} = %d$', k_mode), 'Interpreter', 'Latex')
        set(gca, 'YScale', 'log');
        %legend('gpr', 'mc', 'Interpreter', 'Latex', 'Location', 'South');
        xlim([-3.5, 3.5]);
        ylim([1e-3, 1]);
        grid on

        set(gca, 'FontSize', 9);

    end

    set(gcf,'units','inches','position', a_par.plot_pos);
    set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.full_paper_pos, 'PaperSize', a_par.full_paper_size);

    if a_par.save_figs
        filename = sprintf('%s%s-recon-modes-pdf-no-mean', a_par.fig_path, gpr_surrogate.exp_name);
        print(filename,'-dpdf');
        savefig(filename);
    end

    %
    % histogram the recovered time series
    %
    
    V_out = gpr_surrogate.V_out;
    lambda = gpr_surrogate.D_out; %rescale by KL eigenweights
    beta = gpr_surrogate.overall_norm_factor; % final rescaling
    ts_mu = gpr_surrogate.ts_mu;
    
    zz_list_nd = zeros(n_samples, size(V_out, 1));
    %zz_list_mo = zeros(n_samples, size(V_out, 1));
    
    for k_sample = 1:n_samples
        zz_list_nd(k_sample, :) = ts_transform_kl( a_par, yy_guess_nd(k_sample, :), V_out, lambda, ts_mu );
        %zz_list_mo(k_sample, :) = ts_transform_kl( a_par, yy_guess_mo(k_sample, :), V_out, lambda, ts_mu );
    end
    
    %zz_list_nd = zeros(n_samples, size(V_out, 1));
    %zz_list_mo = zeros(n_samples, size(V_out, 1));

%     for k_sample = 1:n_samples
%         zz_cur_nd = zeros(size(V_out, 1), 1);
%         zz_cur_mo = zeros(size(V_out, 1), 1);
%         for k_mode = 1:a_par.n_modes
%             zz_cur_nd = zz_cur_nd + yy_guess_nd(k_sample, k_mode)*V_out(:, k_mode).*lambda(k_mode);
%             zz_cur_mo = zz_cur_mo + yy_guess_mo(k_sample, k_mode)*V_out(:, k_mode).*lambda(k_mode);
%         end
% 
%         zz_cur_nd = (zz_cur_nd + gpr_surrogate.ts_mu)*beta;
%         zz_cur_mo = (zz_cur_mo + gpr_surrogate.ts_mu)*beta; % remove normalizations
% 
%         zz_list_nd(k_sample, :) = zz_cur_nd;
%         zz_list_mo(k_sample, :) = zz_cur_mo;
%     end

    
    %xx_zz = linspace(-3e9, 3e9, a_par.n_hist_bins);
    xx_zz = linspace(-10*beta, 10*beta, a_par.n_hist_bins);

    %PP_zz_mo = histcounts(zz_list_mo(:)*beta, xx_zz, 'Normalization', 'pdf');
    PP_zz_nd = histcounts(zz_list_nd(:)*beta, xx_zz, 'Normalization', 'pdf');

    %
    % histogram the true time series
    %


    PP_zz_true = histcounts(zz_true(:)*beta, xx_zz, 'Normalization', 'pdf');

    %
    % plot time series histogram
    %

%     figure(11);
%     clf;
%     hold on
%     xx_plot_1 = 1/2*(xx_zz(2:end) + xx_zz(1:end-1));
%     hold on
%     plot(xx_plot_1,  PP_zz_mo, 'Color', 'Blue', 'LineWidth', 3);
%     plot(xx_plot_1,  PP_zz_nd, 'Color', 'Cyan', 'LineWidth', 3);
%     plot(xx_plot_1,  PP_zz_true, 'Color', 'Red', 'LineStyle', ':', 'LineWidth', 3);
%     xlabel('$z$', 'Interpreter', 'Latex');
%     ylabel('$f_Z(z)$', 'Interpreter', 'Latex');
%     title(sprintf('resampled global vbm pdf -- %s', gpr_surrogate.exp_name), 'Interpreter', 'Latex');
%     %legend({'recovered', 'true'}, 'Interpreter', 'Latex');
%     legend('gpr-mean', 'gpr-spread', 'mc', 'Interpreter', 'Latex', 'Location', 'South');
%     set(gca, 'YScale', 'log');
%     xlim([-2e9, 2e9]);
%     grid on
% 
%     set(gca, 'FontSize', 9);
% 
%     set(gcf,'units','inches','position', a_par.plot_pos);
%     set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
% 
%     if a_par.save_figs
%         filename = sprintf('%s%s-recon-vbmg-pdf', a_par.fig_path, gpr_surrogate.exp_name);
%         print(filename,'-dpdf');
%         savefig(filename);
%     end
    
    
    
    figure(12);
    clf;
    hold on
    xx_plot_1 = 1/2*(xx_zz(2:end) + xx_zz(1:end-1));
    hold on
    plot(xx_plot_1,  PP_zz_nd, 'Color', 'Blue', 'LineStyle', '-', 'LineWidth', 3);
    plot(xx_plot_1,  PP_zz_true, 'Color', 'Black', 'LineStyle', '-', 'LineWidth', 3);
    xlabel('$z$', 'Interpreter', 'Latex');
    ylabel('$f_Z(z)$', 'Interpreter', 'Latex');
    title(sprintf('resampled global vbm pdf -- %s', gpr_surrogate.exp_name), 'Interpreter', 'Latex');
    %legend({'recovered', 'true'}, 'Interpreter', 'Latex');
    legend('gpr', 'mc', 'Interpreter', 'Latex', 'Location', 'South');
    set(gca, 'YScale', 'log');
    %xlim([-2e9, 2e9]);
    %ylim([5e-14, 2e-9])
    grid on

    set(gca, 'FontSize', 9);

    set(gcf,'units','inches','position', a_par.plot_pos);
    set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);

    if a_par.save_figs
        filename = sprintf('%s%s-recon-vbmg-pdf-no-mean', a_par.fig_path, gpr_surrogate.exp_name);
        print(filename,'-dpdf');
        savefig(filename);
    end
    
    
    
    XX = xx_plot_1;
    FF = PP_zz_nd;

end

