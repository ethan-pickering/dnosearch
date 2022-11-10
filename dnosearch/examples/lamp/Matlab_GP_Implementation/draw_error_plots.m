function [ err_struct  ] = draw_error_plots( a_par, as_par, protocol_list, ...
    true_f_mean, true_pq, true_pz)
%DRAW_ERROR_PLOTS Summary of this function goes here
%   Detailed explanation goes here

    tic;
    fprintf('Beginning error plot calculations.\n')

    f_input = @(alpha) prod(1/sqrt(2*pi)*exp(-alpha.^2/2), 2);

%     a3_grid = linspace(-as_par.z_max, as_par.z_max, as_par.n_grid_likelihood);
%     [aa13, aa23, aa33] = meshgrid(a3_grid, a3_grid, a3_grid);
%     aa3_grid = [aa13(:), aa23(:), aa33(:)];

    aa3_grid = as_par.z_max*(1-2*lhsdesign(1e4, as_par.n_dim_in));
    ww3 = f_input(aa3_grid);
    dww3 = ww3./sum(ww3(:));

    bbq = linspace(-as_par.q_max, as_par.q_max, as_par.nqb+1);
    qq_interval = 1/2*(bbq(1:end-1) + bbq(2:end));

    beta = protocol_list{1}.gpr_obj.overall_norm_factor;
    bbz = linspace(-7*beta, 7*beta, as_par.nqb+1);
    qqz_interval = 1/2*(bbz(1:end-1) + bbz(2:end));

    pq_list = cell(length(protocol_list), 1);
    pz_list = cell(length(protocol_list), 1);

    
    fprintf('Computing q pdf using rules:  %s / %s.\n', ...
        as_par.q_pdf_rule, as_par.true_q_pdf_rule);

    for k = 1:length(protocol_list)
        fprintf('Starting k=%d. (%0.2f seconds elapsed).\n', k, toc);

        [ pq, pz] = compute_histograms_from_gpr_protocol(a_par, as_par, ...
            protocol_list{k});

        pq_list{k} = pq;
        pz_list{k} = pz;
    end


    surr_mu_mae_list = zeros(length(protocol_list), 1);
    surr_mu_rmse_list = zeros(length(protocol_list), 1);

    if as_par.compute_surr_errors
        true_surr_mu = true_f_mean(aa3_grid);

        fprintf('Calculating intermediate pdf errors --- %d total rounds.\n', length(protocol_list));
        for k = 1:length(protocol_list)
            fprintf('Starting k=%d. (%0.2f seconds elapsed).\n', k, toc);
    
            %
            % surrogate estimate
            %
    
            [ cur_surr_mu, ~] = protocol_list{k}.gpr_obj.predict(aa3_grid);
            delta = cur_surr_mu(:, as_par.q_plot) - true_surr_mu(:, as_par.q_plot);
            surr_mu_mae_list(k) = sum(abs(delta).*dww3);
            surr_mu_rmse_list(k) = sqrt(sum(delta.^2.*dww3));
    
        end
    end




    NN_plot = (1:length(pq_list)) + as_par.n_init;
    
    %
    % calculate errors for particular mode coeff
    %

    qp_mae_list = zeros(length(pq_list), 1);
    qp_rmse_list = zeros(length(pq_list), 1);
    qp_log_mae_list = zeros(length(pq_list), as_par.n_kl_bounds);
    qp_log_rmse_list = zeros(length(pq_list), as_par.n_kl_bounds);
    qp_kl_div_forward_list = zeros(length(pq_list), as_par.n_kl_bounds);
    qp_kl_div_backward_list = zeros(length(pq_list), as_par.n_kl_bounds);

    if as_par.compute_mode_errors
        
        for k = 1:length(pq_list)
            cur_qp = pq_list{k};
    
            delta = cur_qp - true_pq;
            qp_mae_list(k) = mean(abs(delta));
            qp_rmse_list(k) = sqrt(mean(delta.^2));
    
            log_delta = log(cur_qp) - log(true_pq);
            for j = 1:as_par.n_kl_bounds
                kl_lim = as_par.kl_bound_list(j);
                ii = find((qq_interval > -kl_lim) & (qq_interval < kl_lim));
    
                qp_log_mae_list(k, j) = mean(abs(log_delta(ii)));
                qp_log_rmse_list(k, j) = sqrt(mean(log_delta(ii).^2));
                
                qp_kl_div_forward_list(k, j) = sum(true_pq(ii).*log(true_pq(ii)./cur_qp(ii)));
                qp_kl_div_backward_list(k, j) = sum(cur_qp(ii).*log(cur_qp(ii)./true_pq(ii)));
            end
        end
    end

    %
    % calculate errors for total vbm
    %

    pz_mae_list = zeros(length(pz_list), 1);
    pz_rmse_list = zeros(length(pz_list), 1);
    pz_log_mae_list = zeros(length(pz_list), as_par.n_kl_bounds);
    pz_log_rmse_list = zeros(length(pz_list), as_par.n_kl_bounds);
    pz_kl_div_forward_list = zeros(length(pz_list), as_par.n_kl_bounds);
    pz_kl_div_backward_list = zeros(length(pz_list), as_par.n_kl_bounds);
    pz_log_mae_trunc_list = zeros(length(pz_list), 1);
    pz_log_mae_trunc_list2 = zeros(length(pz_list), 1);

    for k = 1:length(pz_list)
        cur_pz = pz_list{k};

        delta = cur_pz - true_pz;
        pz_mae_list(k) = mean(abs(delta));
        pz_rmse_list(k) = sqrt(mean(delta.^2));

        log_delta = log(cur_pz) - log(true_pz);
        for j = 1:as_par.n_kl_bounds
            kl_upper_lim = as_par.kl_bound_list_vbm_upper(j);
            kl_lower_lim = as_par.kl_bound_list_vbm_lower(j);
            ii = find((qqz_interval > kl_lower_lim) & (qqz_interval < kl_upper_lim));

            pz_log_mae_list(k, j) = mean(abs(log_delta(ii)));
            pz_log_rmse_list(k, j) = sqrt(mean(log_delta(ii).^2));
            
            pz_kl_div_forward_list(k, j) = sum(true_pz(ii).*log(true_pz(ii)./cur_pz(ii)));
            pz_kl_div_backward_list(k, j) = sum(cur_pz(ii).*log(cur_pz(ii)./true_pz(ii)));
        end

        bbz = linspace(-10*beta, 10*beta, as_par.nqb+1);
        pz_log_mae_trunc_list(k) = calc_log_pdf_errors(true_pz, cur_pz, bbz, 1e-13);
        pz_log_mae_trunc_list2(k) = calc_log_pdf_errors(true_pz, cur_pz, bbz, 1e-12);
    end

    fprintf('Plotting recovered q pdf and error metrics.\n');


    if as_par.draw_plots
        lkk = [5:5:length(protocol_list), length(protocol_list)+1];
        CC = colormap(parula(length(protocol_list)));

        figure(18);
        clf;
        hold on
        hh = zeros(length(protocol_list)+1, 1);
        names = cell(length(protocol_list)+1, 1);
        for k = 1:length(protocol_list)
            hh(k) = plot(qqz_interval, pz_list{k}, 'LineWidth', 1, 'Color', CC(k, :));
            names{k} = sprintf('n = %d', k+as_par.n_init);
        end
        hh(length(protocol_list)+1) = plot(qqz_interval, true_pz, 'LineWidth', 3, 'Color', 'Black');
        names{length(protocol_list)+1} = 'truth';
        xlabel('$M_y$', 'Interpreter', 'Latex')
        ylabel('$p_M(m_y)$', 'Interpreter', 'Latex');
        legend(hh(lkk), names(lkk), 'Interpreter', 'Latex', 'Location', 'South');
        set(gca, 'YScale', 'log')
        title(sprintf('full VBM pdf'), 'Interpreter', 'Latex');
        ylim([1e-16, 1e-9])
        set(gca, 'FontSize', 9);
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
        if a_par.save_figs
            filename = sprintf('%svbm-pdf_total', a_par.fig_path);
            print(filename,'-dpdf');
            savefig(filename);
        end
    
        if as_par.compute_mode_errors
            figure(13);
            clf;
            hold on
            hh = zeros(length(protocol_list)+1, 1);
            names = cell(length(protocol_list)+1, 1);
            for k = 1:length(protocol_list)
                hh(k) = plot(qq_interval, pq_list{k}, 'LineWidth', 1, 'Color', CC(k, :));
                names{k} = sprintf('n = %d', k+as_par.n_init);
            end
            hh(length(pq_list)+1) = plot(qq_interval, true_pq, 'LineWidth', 3, 'Color', 'Black');
            names{length(pq_list)+1} = 'truth';
            xlabel('$q_1$', 'Interpreter', 'Latex')
            ylabel('$p_Q(q_1)$', 'Interpreter', 'Latex');
            legend(hh(lkk), names(lkk), 'Interpreter', 'Latex', 'Location', 'South');
            set(gca, 'YScale', 'log')
            title(sprintf('q pdf, mode %d', as_par.q_plot), 'Interpreter', 'Latex');
            ylim([1e-7, 1e0])
            set(gca, 'FontSize', 9);
            set(gcf,'units','inches','position', a_par.plot_pos);
            set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
        
            if a_par.save_figs
                filename = sprintf('%sq-pdf_total_mode_%d', a_par.fig_path, as_par.q_plot);
                print(filename,'-dpdf');
                savefig(filename);
            end

            figure(14);
            clf;
            hold on
            plot(NN_plot, qp_mae_list);
            plot(NN_plot, qp_rmse_list);
            xlabel('$n$', 'Interpreter', 'Latex')
            ylabel('$\epsilon$', 'Interpreter', 'Latex');
            legend({'MAE', 'RMSE'}, 'Interpreter', 'Latex')
            set(gca, 'YScale', 'log')
            title(sprintf('q-pdf error q=%d', as_par.q_plot), 'Interpreter', 'Latex');
            set(gca, 'FontSize', 9);
            set(gcf,'units','inches','position', a_par.plot_pos);
            set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
        
            if a_par.save_figs
                filename = sprintf('%sq-pdf-error_q=%d', a_par.fig_path, as_par.q_plot);
                print(filename,'-dpdf');
                savefig(filename);
            end
    
            figure(15);
            clf;
            hold on
            hh = zeros(2*as_par.n_kl_bounds, 1);
            for j = 1:as_par.n_kl_bounds
                hh(2*j-1) = plot(NN_plot, qp_log_mae_list(:, j), 'Color', 'Red');
                hh(2*j) = plot(NN_plot, qp_log_rmse_list(:, j), 'Color', 'Blue');
            end
            xlabel('$n$', 'Interpreter', 'Latex')
            ylabel('$\epsilon$', 'Interpreter', 'Latex');
            legend(hh(1:2), {'MAE', 'RMSE'}, 'Interpreter', 'Latex')
            set(gca, 'YScale', 'log')
            title(sprintf('q-pdf log error q=%d', as_par.q_plot), 'Interpreter', 'Latex');
            set(gca, 'FontSize', 9);
            set(gcf,'units','inches','position', a_par.plot_pos);
            set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
        
            if a_par.save_figs
                filename = sprintf('%sq-pdf-log_error_q=%d', a_par.fig_path, as_par.q_plot);
                print(filename,'-dpdf');
                savefig(filename);
            end
    
            figure(16);
            clf;
            hold on
            hh = zeros(2*as_par.n_kl_bounds, 1);
            for j = 1:as_par.n_kl_bounds
                hh(2*j-1) = plot(NN_plot, qp_kl_div_forward_list(:, j), 'Color', 'Red');
                hh(2*j) = plot(NN_plot, qp_kl_div_backward_list(:, j), 'Color', 'Blue');
            end
            xlabel('$n$', 'Interpreter', 'Latex')
            ylabel('$D_{KL}$', 'Interpreter', 'Latex');
            legend(hh(1:2), {'$D_{KL}(\mbox{true} || \mbox{model})$', '$D_{KL}(\mbox{model} || \mbox{true})$'}, ...
                'Interpreter', 'Latex')
            set(gca, 'YScale', 'log')
            title(sprintf('q-pdf KL divergence q=%d', as_par.q_plot), 'Interpreter', 'Latex');
            set(gca, 'FontSize', 9);
            set(gcf,'units','inches','position', a_par.plot_pos);
            set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
        
            if a_par.save_figs
                filename = sprintf('%sq-pdf-kl-div_q=%d', a_par.fig_path, as_par.q_plot);
                print(filename,'-dpdf');
                savefig(filename);
            end
        end
    
        if as_par.compute_surr_errors
            figure(17);
            clf;
            hold on
            plot(NN_plot, surr_mu_mae_list);
            plot(NN_plot, surr_mu_rmse_list);
            xlabel('$n$', 'Interpreter', 'Latex')
            ylabel('$\epsilon$', 'Interpreter', 'Latex');
            legend({'MAE', 'RMSE'}, 'Interpreter', 'Latex')
            set(gca, 'YScale', 'log')
            title(sprintf('surrogate mean expected error q=%d', as_par.q_plot), 'Interpreter', 'Latex');
            set(gca, 'FontSize', 9);
            set(gcf,'units','inches','position', a_par.plot_pos);
            set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
        
            if a_par.save_figs
                filename = sprintf('%ssurrogate-mean-expected-error_q=%d', a_par.fig_path, as_par.q_plot);
                print(filename,'-dpdf');
                savefig(filename);
            end
        end



        figure(19);
        clf;
        hold on
        plot(NN_plot, pz_mae_list);
        plot(NN_plot, pz_rmse_list);
        xlabel('$n$', 'Interpreter', 'Latex')
        ylabel('$\epsilon$', 'Interpreter', 'Latex');
        legend({'MAE', 'RMSE'}, 'Interpreter', 'Latex')
        set(gca, 'YScale', 'log')
        title(sprintf('VBM-pdf error'), 'Interpreter', 'Latex');
        set(gca, 'FontSize', 9);
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
        if a_par.save_figs
            filename = sprintf('%sVBM-pdf-error', a_par.fig_path);
            print(filename,'-dpdf');
            savefig(filename);
        end

        figure(20);
        clf;
        hold on
        hh = zeros(2*as_par.n_kl_bounds, 1);
        for j = 1:as_par.n_kl_bounds
            hh(2*j-1) = plot(NN_plot, pz_log_mae_list(:, j), 'Color', 'Red');
            hh(2*j) = plot(NN_plot, pz_log_rmse_list(:, j), 'Color', 'Blue');
        end
        xlabel('$n$', 'Interpreter', 'Latex')
        ylabel('$\epsilon$', 'Interpreter', 'Latex');
        legend(hh(1:2), {'MAE', 'RMSE'}, 'Interpreter', 'Latex')
        set(gca, 'YScale', 'log')
        title(sprintf('VBM-pdf log error'), 'Interpreter', 'Latex');
        set(gca, 'FontSize', 9);
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
        if a_par.save_figs
            filename = sprintf('%sVBM-pdf-log_error', a_par.fig_path);
            print(filename,'-dpdf');
            savefig(filename);
        end

        figure(21);
        clf;
        hold on
        hh = zeros(2*as_par.n_kl_bounds, 1);
        for j = 1:as_par.n_kl_bounds
            hh(2*j-1) = plot(NN_plot, pz_kl_div_forward_list(:, j), 'Color', 'Red');
            hh(2*j) = plot(NN_plot, pz_kl_div_backward_list(:, j), 'Color', 'Blue');
        end
        xlabel('$n$', 'Interpreter', 'Latex')
        ylabel('$D_{KL}$', 'Interpreter', 'Latex');
        legend(hh(1:2), {'$D_{KL}(\mbox{true} || \mbox{model})$', '$D_{KL}(\mbox{model} || \mbox{true})$'}, ...
            'Interpreter', 'Latex')
        set(gca, 'YScale', 'log')
        title(sprintf('VBM-pdf KL divergence'), 'Interpreter', 'Latex');
        set(gca, 'FontSize', 9);
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
        if a_par.save_figs
            filename = sprintf('%sVBM-pdf-kl-div', a_par.fig_path);
            print(filename,'-dpdf');
            savefig(filename);
        end
    end



    if as_par.save_errors
        filename = sprintf('%s/error_data.m', a_par.fig_path);
        save(filename, 'surr_mu_mae_list', 'surr_mu_rmse_list', ...
            'qp_mae_list', 'qp_rmse_list',  'qp_kl_div_forward_list', ...
            'qp_kl_div_backward_list', 'qp_log_mae_list', 'qp_log_rmse_list', ...
            'pz_mae_list', 'pz_rmse_list',  'pz_kl_div_forward_list', ...
            'pz_kl_div_backward_list', 'pz_log_mae_list', 'pz_log_rmse_list');
    end

    err_struct = struct();
    err_struct.pz_list = pz_list;
    err_struct.surr_mu_mae_list = surr_mu_mae_list;
    err_struct.surr_mu_rmse_list = surr_mu_rmse_list;
    err_struct.qp_mae_list = qp_mae_list;
    err_struct.qp_rmse_list = qp_rmse_list;
    err_struct.qp_kl_div_forward_list = qp_kl_div_forward_list;
    err_struct.qp_kl_div_backward_list = qp_kl_div_backward_list;
    err_struct.qp_log_mae_list = qp_log_mae_list;
    err_struct.qp_log_rmse_list = qp_log_rmse_list;
    err_struct.pz_mae_list = pz_mae_list;
    err_struct.pz_rmse_list = pz_rmse_list;
    err_struct.pz_kl_div_forward_list = pz_kl_div_forward_list;
    err_struct.pz_kl_div_backward_list = pz_kl_div_backward_list;
    err_struct.pz_log_mae_list = pz_log_mae_list;
    err_struct.pz_log_rmse_list = pz_log_rmse_list;
    err_struct.pz_log_mae_trunc_list = pz_log_mae_trunc_list;
    err_struct.pz_log_mae_trunc_list2 = pz_log_mae_trunc_list2;

    fprintf('Error calculation and plotting stuff done after %0.2f seconds.\n', toc);

end

