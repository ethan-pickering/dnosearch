function [ outcode ] = draw_plots_toy( a_par, as_par, model_list, true_f, true_pq)
%DRAW_PLOTS Summary of this function goes here
%   Detailed explanation goes here

    tic;

    %z_max = 4.5;
    %n_init = 10;

    %q_plot = 1;
    a_grid = linspace(-as_par.z_max, as_par.z_max, as_par.na);
    %[aa1, aa2] = meshgrid(a_grid, a_grid);
    %aa_grid = [aa1(:), aa2(:), zeros(size(aa1(:)))];
    aa_grid = a_grid';

    f_input = @(alpha) prod(1/sqrt(2*pi)*exp(-alpha.^2/2), 2);

    a3_grid = linspace(-as_par.z_max, as_par.z_max, as_par.n_grid_likelihood);
    %[aa13, aa23, aa33] = meshgrid(a3_grid, a3_grid, a3_grid);
    %aa3_grid = [aa13(:), aa23(:), aa33(:)];
    aa3_grid = a3_grid';
    ww3 = f_input(aa3_grid);
    dww3 = ww3./sum(ww3(:));
    
    %nqb= 65;
    %q_max = 6.5;
    bbq = linspace(as_par.q_min, as_par.q_max, as_par.nqb+1);

    %qq_interval = linspace(-q_max, q_max, nqb);
    qq_interval = 1/2*(bbq(1:end-1) + bbq(2:end));

    %save_intermediate_plots = false;

    pq_list = cell(length(model_list), 1);
    %nq_mc = 5e6;
    %q_pdf_rule = 'MC';
    %true_q_pdf_rule = 'MC';
    fprintf('Computing q pdf using rules:  %s / %s.\n', ...
        as_par.q_pdf_rule, as_par.true_q_pdf_rule);

    fprintf('Drawing true model stuff.\n');

   

    true_f_noiseless = @(x) x.^1.5.*(x>0) + x.*(x<0).*(x>=-2) -2.*(x<-2);
    true_surr_mu = true_f_noiseless(aa3_grid);
    surr_mu_mae_list = zeros(length(model_list), 1);
    surr_mu_rmse_list = zeros(length(model_list), 1);

    zz = true_f_noiseless(aa_grid);
    zz_plot = reshape(zz, size(zz));
    
    if as_par.draw_plots
        figure(4);
        clf;
        %pcolor(aa1, aa2, zz_plot)
        plot(aa3_grid, zz_plot, 'LineWidth', 3)
        shading flat
        xlabel('$x$', 'Interpreter', 'Latex')
        ylabel('$\mu_y$', 'Interpreter', 'Latex')
        title(sprintf('true function'), 'Interpreter', 'Latex');
        %colorbar();
        set(gca, 'FontSize', 9);
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.third_paper_pos, 'PaperSize', a_par.third_paper_size);
        
        if a_par.save_figs
            filename = sprintf('%ssurrogate_true_q_%d', a_par.fig_path, as_par.q_plot);
            print(filename,'-dpdf');
            savefig(filename);
        end
    end

    max_surr_abs = max(abs(zz_plot(:)));
    
    zz = f_input(aa_grid);
    

    if as_par.draw_plots
        zz_plot = reshape(zz, size(zz));
        figure(11);
        clf;
        %pcolor(aa1, aa2, zz_plot)
        plot(aa_grid, zz_plot, 'LineWidth', 3);
        shading flat
        xlabel('$x$', 'Interpreter', 'Latex')
        ylabel('$f_X(x)$', 'Interpreter', 'Latex')
        title(sprintf('input distribution'), 'Interpreter', 'Latex');
        %colorbar();
        set(gca, 'FontSize', 9);
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.third_paper_pos, 'PaperSize', a_par.third_paper_size);
        
        if a_par.save_figs
            filename = sprintf('%sinput_density', a_par.fig_path);
            print(filename,'-dpdf');
            savefig(filename);
        end
    end

    if as_par.save_videos
        filename = sprintf('%ssurrogate-q-likelihood-evolution', as_par.video_path);
        if isfile(filename)
            delete(filename)
        end
        q_likelihood_vid_file = VideoWriter(filename, as_par.vid_profile);
        q_likelihood_vid_file.FrameRate = as_par.video_frame_rate;
        open(q_likelihood_vid_file);
    
        filename = sprintf('%ssurrogate-mean-evolution', as_par.video_path);
        if isfile(filename)
            delete(filename)
        end
        surr_mean_vid_file = VideoWriter(filename, as_par.vid_profile);
        surr_mean_vid_file.FrameRate = as_par.video_frame_rate;
        open(surr_mean_vid_file);
    
        filename = sprintf('%sacq-function-evolution', as_par.video_path);
        if isfile(filename)
            delete(filename)
        end
        lw_us_acq_vid_file = VideoWriter(filename, as_par.vid_profile);
        lw_us_acq_vid_file.FrameRate = as_par.video_frame_rate;
        open(lw_us_acq_vid_file);

        filename = sprintf('%slw-evolution', as_par.video_path);
        if isfile(filename)
            delete(filename)
        end
        lw_vid_file = VideoWriter(filename, as_par.vid_profile);
        lw_vid_file.FrameRate = as_par.video_frame_rate;
        open(lw_vid_file);

        filename = sprintf('%ssr-std-evolution', as_par.video_path);
        if isfile(filename)
            delete(filename)
        end
        surr_std_vid_file = VideoWriter(filename, as_par.vid_profile);
        surr_std_vid_file.FrameRate = as_par.video_frame_rate;
        open(surr_std_vid_file);
    end


    fprintf('Drawing iterated reconstruction stuff --- %d total rounds.\n', length(model_list));
    for k = 1:length(model_list)
        fprintf('Starting k=%d. (%0.2f seconds elapsed).\n', k, toc);

        cur_model = model_list{k};

        %[ f_likelihood ] = build_likelihood(@(alpha) cur_model.predict(alpha), ...
        %    aa3_grid, ww3, bbq);
        [ f_likelihood ] = build_likelihood_function(as_par, f_input, true_f);
        zz = f_likelihood(qq_interval);


        switch as_par.q_pdf_rule
            case 'likelihood-transform'
                pq_list{k} = zz;
            case 'MC'
                aa_q = randn(as_par.nq_mc, 1);
                [ mm, ss ] = cur_model.predict(aa_q);
                qq = mm + ss.*randn(size(mm));
                pq_list{k} = histcounts(qq, bbq, ...
                    'Normalization', 'pdf');
        end
%         figure(10)
%         clf;
%         plot(qq_interval, zz, 'LineWidth', 3);
%         xlabel('$q_1$', 'Interpreter', 'Latex')
%         ylabel('$p_Q(q_1)$', 'Interpreter', 'Latex')
%         set(gca, 'YScale', 'log')
%         title(sprintf('q-likelihood %d', q_plot), 'Interpreter', 'Latex');
%         ylim([1e-7, 1e0])
%         set(gca, 'FontSize', 9);
%         set(gcf,'units','inches','position', a_par.plot_pos);
%         set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
% 
%         if a_par.save_figs
%             filename = sprintf('%sq-likelihood_transform_n_%d_q_%d', a_par.fig_path, k, q_plot);
%             print(filename,'-dpdf');
%             savefig(filename);
%         end


        [qq, ss] = cur_model.predict(aa_grid);
        zz = f_likelihood(qq);

        if as_par.draw_plots
            zz_plot = reshape(zz, size(zz));
            figure(6);
            clf;
            %pcolor(aa1, aa2, zz_plot)
            plot(qq, zz_plot)
            set(gca, 'YScale', 'log');
            %shading flat
            xlabel('$y$', 'Interpreter', 'Latex')
            ylabel('$f_Y(y)$', 'Interpreter', 'Latex')
            title(sprintf('output-likelihood'), 'Interpreter', 'Latex');
            %colorbar();
            set(gca, 'FontSize', 9);
            set(gcf,'units','inches','position', a_par.plot_pos);
            set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
            if as_par.save_intermediate_plots
                filename = sprintf('%sq-likelihood_n_%d', a_par.fig_path, k);
                print(filename,'-dpdf');
                savefig(filename);
            end
    
            if as_par.save_videos
                F = getframe(gcf); 
                writeVideo(q_likelihood_vid_file,F);
            end
        end


        [ cur_surr_mu, ~] = cur_model.predict(aa3_grid);
        delta = cur_surr_mu - true_surr_mu;
        surr_mu_mae_list(k) = sum(abs(delta).*dww3);
        surr_mu_rmse_list(k) = sqrt(sum(delta.^2.*dww3));

        if as_par.draw_plots
            zz = f_input(aa_grid)./f_likelihood(qq);
            zz_plot = reshape(zz, size(aa_grid));
            figure(8);
            clf;
            %pcolor(aa1, aa2, zz_plot)
            plot(aa_grid, zz_plot)
            %shading flat
            xlabel('$x$', 'Interpreter', 'Latex')
            ylabel('$w(x)$', 'Interpreter', 'Latex')
            title(sprintf('likelihood ratio'), 'Interpreter', 'Latex');
            %colorbar();
            set(gca, 'FontSize', 9);
            set(gcf,'units','inches','position', a_par.plot_pos);
            set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
            if as_par.save_intermediate_plots
                filename = sprintf('%slikelihood-ratio_n_%d', a_par.fig_path, k);
                print(filename,'-dpdf');
                savefig(filename);
            end
    
            if as_par.save_videos
                F = getframe(gcf); 
                writeVideo(lw_vid_file,F);
            end
        end


        zz = f_input(aa_grid)./f_likelihood(qq).*ss.^2;

        if as_par.draw_plots
            zz_plot = reshape(zz, size(zz));
            figure(9);
            clf;
            %pcolor(aa1, aa2, zz_plot)
            plot(aa_grid, zz_plot)
            %shading flat
            xlabel('$x$', 'Interpreter', 'Latex')
            ylabel('$a_{LW-US}$', 'Interpreter', 'Latex')
            title(sprintf('lw-us-direct %d', as_par.q_plot), 'Interpreter', 'Latex');
            %colorbar();
            set(gca, 'FontSize', 9);
            set(gcf,'units','inches','position', a_par.plot_pos);
            set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
            if as_par.save_intermediate_plots
                filename = sprintf('%sacq-direct_n_%d', a_par.fig_path, k);
                print(filename,'-dpdf');
                savefig(filename);
            end
    
            if as_par.save_videos
                F = getframe(gcf); 
                writeVideo(lw_us_acq_vid_file,F);
            end
        end


%         ss_adj = ss - cur_model_protocol.gpr_obj.g_fit_list{1}.Sigma;
%         zz = f_input(aa_grid)./f_likelihood(qq(:, 1)).*ss_adj;
%         zz_plot = reshape(zz(:, q_plot), size(aa1));
%         figure(12);
%         clf;
%         pcolor(aa1, aa2, zz_plot)
%         shading flat
%         xlabel('$\alpha_1$', 'Interpreter', 'Latex')
%         ylabel('$\alpha_2$', 'Interpreter', 'Latex')
%         title(sprintf('lw-us-direct-adj %d', q_plot), 'Interpreter', 'Latex');
%         colorbar();
%         set(gca, 'FontSize', 9);
%         set(gcf,'units','inches','position', a_par.plot_pos);
%         set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
% 
%         if a_par.save_intermediate_plots
%             filename = sprintf('%sacq-direct-adj_n_%d_q_%d', a_par.fig_path, k, q_plot);
%             print(filename,'-dpdf');
%             savefig(filename);
%         end



%         zz_plot = reshape(ss(:, q_plot), size(aa1));
%         figure(7);
%         clf;
%         pcolor(aa1, aa2, zz_plot)
%         shading flat
%         xlabel('$\alpha_1$', 'Interpreter', 'Latex')
%         ylabel('$\alpha_2$', 'Interpreter', 'Latex')
%         title(sprintf('gpr uncertainty mode %d', q_plot), 'Interpreter', 'Latex');
%         colorbar();
%         set(gca, 'FontSize', 9);
%         set(gcf,'units','inches','position', a_par.plot_pos);
%         set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
% 
%         if a_par.save_intermediate_plots
%             filename = sprintf('%suncertainty_n_%d_q_%d', a_par.fig_path, k, q_plot);
%             print(filename,'-dpdf');
%             savefig(filename);
%         end


%        f_blackbox = @(alpha) cur_model_protocol.gpr_obj.predict(alpha);
%        f_acq = @(alpha) -f_acq_lw_us(alpha, f_input, f_likelihood, f_blackbox);
%        zz = f_acq(aa_grid);
%        zz_plot = reshape(zz(:, q_plot), size(aa1));
%         figure(5);
%         clf;
%         pcolor(aa1, aa2, zz_plot)
%         shading flat
%         xlabel('$\alpha_1$', 'Interpreter', 'Latex')
%         ylabel('$\alpha_2$', 'Interpreter', 'Latex')
%         title(sprintf('lw-us mode %d', q_plot), 'Interpreter', 'Latex');
%         colorbar();
%         set(gca, 'FontSize', 9);
%         set(gcf,'units','inches','position', a_par.plot_pos);
%         set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
% 
%         if a_par.save_figs
%             filename = sprintf('%slw-us-acq_n_%d_q_%d', a_par.fig_path, k, q_plot);
%             print(filename,'-dpdf');
%             savefig(filename);
%         end



        [ zz, ss ] = model_list{k}.predict(aa_grid);

        if as_par.draw_plots
            zz_plot = reshape(zz, size(zz));
            figure(3);
            clf;
            %pcolor(aa1, aa2, zz_plot)
            plot(aa_grid, zz_plot);
            shading flat
            xlabel('$x$', 'Interpreter', 'Latex')
            ylabel('$\mu_y$', 'Interpreter', 'Latex')
            title(sprintf('recovered surrogate -- n = %d', k+as_par.n_init), ...
                'Interpreter', 'Latex');
            %caxis([-1.25*max_surr_abs, 1.25*max_surr_abs]);
            %colorbar();
            set(gca, 'FontSize', 9);
            set(gcf,'units','inches','position', a_par.plot_pos);
            set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
            
            if as_par.save_intermediate_plots
                filename = sprintf('%ssurrogate_n_%d', a_par.fig_path, k);
                print(filename,'-dpdf');
                savefig(filename);
            end
    
            if as_par.save_videos
                F = getframe(gcf); 
                writeVideo(surr_mean_vid_file,F);
            end

        end


        if as_par.draw_plots
            zz_plot = reshape(ss, size(ss));
            figure(23);
            clf;
            %pcolor(aa1, aa2, zz_plot)
            plot(aa_grid, zz_plot);
            %shading flat
            xlabel('$x$', 'Interpreter', 'Latex')
            ylabel('$\sigma_y$', 'Interpreter', 'Latex')
            title(sprintf('surrogate uncertainty -- n = %d', k+as_par.n_init), ...
                'Interpreter', 'Latex');
            %caxis([-1.25*max_surr_abs, 1.25*max_surr_abs]);
            %colorbar();
            set(gca, 'FontSize', 9);
            set(gcf,'units','inches','position', a_par.plot_pos);
            set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
            
            if as_par.save_intermediate_plots
                filename = sprintf('%ssur_uncertainty_n_%d', a_par.fig_path, k);
                print(filename,'-dpdf');
                savefig(filename);
            end
    
            if as_par.save_videos
                F = getframe(gcf); 
                writeVideo(surr_std_vid_file,F);
            end
        end

    end


    if as_par.save_videos
        close(q_likelihood_vid_file);
        close(surr_mean_vid_file);
        close(lw_us_acq_vid_file);
        close(lw_vid_file);
        close(surr_std_vid_file);
    end
    



    %cur_model_protocol.plot_surrogate(1);
    %true_model_protocol.plot_surrogate(1);

    fprintf('Plotting recovered q pdf and error metrics.\n');

    
    

    if as_par.draw_plots
        lkk = [5:5:length(model_list), 51];
        CC = colormap(parula(length(pq_list)));
        figure(13);
        clf;
        hold on
        hh = zeros(length(pq_list)+1, 1);
        names = cell(length(pq_list)+1, 1);
        for k = 1:length(pq_list)
            hh(k) = plot(qq_interval, pq_list{k}, 'LineWidth', 1, 'Color', CC(k, :));
            names{k} = sprintf('n = %d', k+as_par.n_init);
        end
        hh(max(lkk)) = plot(qq_interval, true_pq, 'LineWidth', 3, 'Color', 'Black');
        names{max(lkk)} = 'truth';
        xlabel('$y$', 'Interpreter', 'Latex')
        ylabel('$p_Y(y)$', 'Interpreter', 'Latex');
        legend(hh(lkk), names(lkk), 'Interpreter', 'Latex', 'Location', 'Northeast');
        set(gca, 'YScale', 'log')
        title(sprintf('y pdf'), 'Interpreter', 'Latex');
        ylim([1e-7, 1e0])
        set(gca, 'FontSize', 9);
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
        if a_par.save_figs
            filename = sprintf('%sq-pdf_total_mode_%d', a_par.fig_path, as_par.q_plot);
            print(filename,'-dpdf');
            savefig(filename);
        end
    end




    %kl_bound_list = [2, 2.25, 2.5, 2.75, 3];

    %n_kl_bounds = length(kl_bound_list);
    qp_mae_list = zeros(length(pq_list), 1);
    qp_rmse_list = zeros(length(pq_list), 1);
    qp_log_mae_list = zeros(length(pq_list), as_par.n_kl_bounds);
    qp_log_rmse_list = zeros(length(pq_list), as_par.n_kl_bounds);
    qp_kl_div_forward_list = zeros(length(pq_list), as_par.n_kl_bounds);
    qp_kl_div_backward_list = zeros(length(pq_list), as_par.n_kl_bounds);
    NN_plot = (1:length(pq_list)) + as_par.n_init;

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


    if as_par.draw_plots
        figure(14);
        clf;
        hold on
        plot(NN_plot, qp_mae_list);
        plot(NN_plot, qp_rmse_list);
        xlabel('$n$', 'Interpreter', 'Latex')
        ylabel('$\epsilon$', 'Interpreter', 'Latex');
        legend({'MAE', 'RMSE'}, 'Interpreter', 'Latex')
        set(gca, 'YScale', 'log')
        title(sprintf('y-pdf error'), 'Interpreter', 'Latex');
        set(gca, 'FontSize', 9);
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
        if a_par.save_figs
            filename = sprintf('%sq-pdf-error_q=%d', a_par.fig_path, as_par.q_plot);
            print(filename,'-dpdf');
            savefig(filename);
        end
    end

    if as_par.draw_plots
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
        title(sprintf('y-pdf log error'), 'Interpreter', 'Latex');
        set(gca, 'FontSize', 9);
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
        if a_par.save_figs
            filename = sprintf('%sq-pdf-log_error_q=%d', a_par.fig_path, as_par.q_plot);
            print(filename,'-dpdf');
            savefig(filename);
        end
    end

    if as_par.draw_plots
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
        title(sprintf('pdf KL divergence'), 'Interpreter', 'Latex');
        set(gca, 'FontSize', 9);
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
        if a_par.save_figs
            filename = sprintf('%sq-pdf-kl-div_q=%d', a_par.fig_path, as_par.q_plot);
            print(filename,'-dpdf');
            savefig(filename);
        end
    end


    if as_par.draw_plots
        figure(17);
        clf;
        hold on
        plot(NN_plot, surr_mu_mae_list);
        plot(NN_plot, surr_mu_rmse_list);
        xlabel('$n$', 'Interpreter', 'Latex')
        ylabel('$\epsilon$', 'Interpreter', 'Latex');
        legend({'MAE', 'RMSE'}, 'Interpreter', 'Latex')
        set(gca, 'YScale', 'log')
        title(sprintf('surrogate mean expected error'), 'Interpreter', 'Latex');
        set(gca, 'FontSize', 9);
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
        if a_par.save_figs
            filename = sprintf('%ssurrogate-mean-expected-error_q=%d', a_par.fig_path, as_par.q_plot);
            print(filename,'-dpdf');
            savefig(filename);
        end
    end



    if as_par.save_errors
        filename = sprintf('%s/error_data.m', a_par.fig_path);
        save(filename, 'qp_mae_list', 'qp_rmse_list', 'surr_mu_mae_list', ...
            'surr_mu_rmse_list', 'qp_kl_div_forward_list', 'qp_kl_div_backward_list', ...
            'qp_log_mae_list', 'qp_log_rmse_list');
    end

    fprintf('Plotting stuff done after %0.2f seconds.\n', toc);

    
    outcode = 1;

end

